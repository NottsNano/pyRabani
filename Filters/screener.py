# -*- coding: utf-8 -*-

import pycroscopy as scope
import matplotlib.pyplot as plt
import h5py
import pyUSID
from scipy import ndimage, signal, stats
import numpy as np
from os import listdir
from sys import exit

# Toggles showing progress statements
p_s = 1

print(p_s*'Locating file...')

files = os.listdir('thres_img/tp')
files_ibw = [i for i in files if i.endswith('.ibw')]

# Create an object capable of translating .ibw files
TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)
path2file = f'thres_img/tp/000TEST.ibw'
# path2file = f'thres_img/tn/SiO2_t12_ring5_1mgmL_0000.ibw' # Image for testing dud line finder

# Translate the requisite file
Output = TranslateObj.translate(
    file_path=path2file, verbose=False)

print(p_s*'Found', Output)
print(p_s*'Opening file and saving file as array...')

# Opening this file to read in sections as a numpy array
Read_Path = Output
h5_File = h5py.File(Output, mode='r')

data_Trace = h5_File['Measurement_000/Channel_000/Raw_Data']
phase_Trace = h5_File['Measurement_000/Channel_002/Raw_Data']
data_Trace_Array = np.array(data_Trace[:])
phase_Trace_Array = np.array(phase_Trace[:])

if data_Trace_Array.shape[0] == 65536:
    row_num = 256
elif data_Trace_Array.shape[0] == 262144:
    row_num = 512
elif data_Trace_Array.shape[0] == 1048576:
    row_num = 1024
else:
    print('Rejected! ibw file is non-standard size.')
    # continue

shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))
shaped_phase_Trace_Array = np.reshape(phase_Trace_Array, (row_num, row_num))

print(p_s*'Closing file...')

h5_File.close()

# ---------------Next step is to apply median difference to rows--------------
# Create a function to take two adjacent rows and return the alignment required to
# move the second row in line with the first


norm_data_Trace_Array = (flattened_data_Trace_Array-np.min(flattened_data_Trace_Array))\
                        / (np.max(flattened_data_Trace_Array)-np.min(flattened_data_Trace_Array))

print(p_s*'Applying median aligner...')

def line_align(row1, row2):
    diff = row1 - row2
    bins = np.linspace(np.min(diff), np.max(diff), 1000)
    binned_indices = np.digitize(diff, bins, right=True)
    np.sort(binned_indices)
    median_index = np.median(binned_indices)
    return bins[int(median_index)]

row_fit_data_Trace_Array = shaped_data_Trace_Array
row_fit_data_Trace_Array[1, :] = shaped_data_Trace_Array[1, :] - np.mean(shaped_data_Trace_Array[1, :])

aligned_med_data_Trace_Array = row_fit_data_Trace_Array
aligned_med_phase_Trace_Array = shaped_phase_Trace_Array
dud_row = 0

for i in range(1, row_num):
    row_iless1 = aligned_med_data_Trace_Array[i - 1, :]
    row_i = aligned_med_data_Trace_Array[i, :]
    Offset = line_align(row_iless1, row_i)
    aligned_med_data_Trace_Array[i, :] = aligned_med_data_Trace_Array[i, :] + Offset

    row_iless1 = aligned_med_phase_Trace_Array[i - 1, :]
    row_i = aligned_med_phase_Trace_Array[i, :]
    Offset = line_align(row_iless1, row_i)
    aligned_med_phase_Trace_Array[i, :] = aligned_med_phase_Trace_Array[i, :] + Offset
    # Extra line that tells the code to count rows where the mode is the same as 95% of the values in the row or ie a dead scan line
    row_mode = stats.mode(row_i)
    # counter = np.count_nonzero(row_i == (stats.mode(row_i))[0])
    counter = np.count_nonzero(0.95*(stats.mode(row_i))[0] < row_i < 1.05(stats.mode(row_i))[0])  # Change line so its 0.95*mode<val<1.05*mode
    dud_row = dud_row + (counter > 0.95*row_num)

if dud_row/row_num > 0.05:
    print('Rejected!', "%.1f" % (100*dud_row/row_num), '% of the rows were corrupted.')
    # continue

# ----------------Next step is to flatten the surface-------------------------

print(p_s*'Applying planar flattening...')

horizon_left = aligned_med_data_Trace_Array[:, 0]
horizon_right = aligned_med_data_Trace_Array[:, row_num - 1]
vertical_top = aligned_med_data_Trace_Array[row_num - 1, :]
vertical_bottom = aligned_med_data_Trace_Array[0, :]

# Finding the approximate direction of the gradient of the plane
hor_gradient = np.mean(horizon_left - horizon_right) / row_num
ver_gradient = np.mean(vertical_top - vertical_bottom) / row_num

# The options for gradients of the plane
hor_grad_array = np.linspace(0 * hor_gradient, 1.5 * hor_gradient, 10)
ver_grad_array = np.linspace(-1.5 * ver_gradient, 1.5 * ver_gradient, 10)

square_differences = np.zeros([10, 10])
centroid = ndimage.measurements.center_of_mass(aligned_med_data_Trace_Array)
centroid_mass = aligned_med_data_Trace_Array[int(np.round(centroid[0])), int(np.round(centroid[1]))]
test_line_x = np.ones([row_num, row_num]) * range(-int(np.round(centroid[0])), row_num - int(np.round(centroid[0])))
test_line_y = np.ones([row_num, row_num]) * range(-int(np.round(centroid[1])), row_num - int(np.round(centroid[1])))

for i in range(0, 10):
    for j in range(0, 10):
        hor_gradient = hor_grad_array[i]
        ver_gradient = ver_grad_array[j]

        hor_array = test_line_x * - hor_gradient
        ver_array = np.transpose(test_line_y * - ver_gradient)

        test_plane = hor_array + ver_array + centroid_mass

        square_differences[i, j] = np.sum(np.square(aligned_med_data_Trace_Array - test_plane))

        # print(i, j, square_differences[i, j])

best_indices = np.unravel_index(np.argmin(square_differences, axis=None), square_differences.shape)

hor_gradient = hor_grad_array[best_indices[0]]
ver_gradient = ver_grad_array[best_indices[1]]

# Test gradients are 1.2e-10 and 0.3e-10
# print(hor_gradient)
# print(ver_gradient)

hor_array = test_line_x * - hor_gradient
ver_array = np.transpose(test_line_y * - ver_gradient)
centroid = ndimage.measurements.center_of_mass(aligned_med_data_Trace_Array)
test_plane = hor_array + ver_array + centroid_mass
# print(centroid_mass)

# The plane is removed
plt.subplot(2, 2, 1)
plt.imshow(aligned_med_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

flattened_data_Trace_Array = aligned_med_data_Trace_Array - test_plane

# -------Next step is to calculate an optimal threshold for image binarising------

# Normalise the array such that all values lie between 0 and 1
norm_data_Trace_Array = (flattened_data_Trace_Array-np.min(flattened_data_Trace_Array))\
                        / (np.max(flattened_data_Trace_Array)-np.min(flattened_data_Trace_Array))

plt.subplot(2, 2, 2)
plt.imshow(norm_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

# Consider all possible threshold values
n = 1000
thres = np.linspace(0, 1, n)
pix = np.zeros((n,))
for i, t in enumerate(thres):
    pix[i] = np.sum(norm_data_Trace_Array < t)

plt.subplot(2,2,3)
threshold_plot = plt.plot(thres, pix)
plt.grid(True)

pix_gauss_grad = ndimage.gaussian_gradient_magnitude(pix,10)
peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=1)
troughs, properties = signal.find_peaks(-pix_gauss_grad, prominence=1)

plt.subplot(2,2,4)
dif_threshold_plot = plt.plot(thres, pix_gauss_grad)
dif_threshold_scatter = plt.scatter(thres[peaks], pix_gauss_grad[peaks])
dif_threshold_scatter = plt.scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
plt.grid(True)


# Print the image
opt_thres = thres[troughs[0]]
plt.figure()
plt.imshow(norm_data_Trace_Array < opt_thres, extent=(0, row_num, 0, row_num), origin='lower', cmap='RdGy')

#Use this to force a plot
#plt.ion()
