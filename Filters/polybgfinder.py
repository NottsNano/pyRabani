# -*- coding: utf-8 -*-

import pycroscopy as scope
import matplotlib.pyplot as plt
import h5py
import pyUSID
from scipy import ndimage, signal
import numpy as np
from sys import exit

from sklearn.preprocessing import PolynomialFeatures

# Create an object capable of translating .ibw files
TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)

# Translate the requisite file
# Output = TranslateObj.translate(
#     file_path=r'ibw_test.ibw', verbose=False)
Output = TranslateObj.translate(
    file_path=r'CurveTest2.ibw', verbose=False)

print(Output)

# Opening this file to read in sections as a numpy array
Read_Path = Output
h5_File = h5py.File(Output, mode='r')

# Various commands for accessing information of the file
# pyUSID.hdf_utils.print_tree(h5_File)
# for key, val in pyUSID.hdf_utils.get_attributes(h5_File).items():
#    print('{} : {}'.format(key, val))

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

shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))
shaped_phase_Trace_Array = np.reshape(phase_Trace_Array, (row_num, row_num))

h5_File.close()

# ---------------Next step is to apply median difference to rows--------------
# Create a function to take two adjacent rows and return the alignment required to
# move the second row in line with the first

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

for i in range(1, row_num):
    row_iless1 = aligned_med_data_Trace_Array[i - 1, :]
    row_i = aligned_med_data_Trace_Array[i, :]
    Offset = line_align(row_iless1, row_i)
    aligned_med_data_Trace_Array[i, :] = aligned_med_data_Trace_Array[i, :] + Offset

    row_iless1 = aligned_med_phase_Trace_Array[i - 1, :]
    row_i = aligned_med_phase_Trace_Array[i, :]
    Offset = line_align(row_iless1, row_i)
    aligned_med_phase_Trace_Array[i, :] = aligned_med_phase_Trace_Array[i, :] + Offset

#Attempt to polyfit the 2d image array

def normalise(array): #Set an image numpy array's values to be between 0 and 1
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array


#METHOD 1

# Finds the mean of every row combined into a 1d array to summarise the image in a vertical direction, and the same
# is done for every column for the horizontal direction.  An n-th degree polynomial is fitted over the array in each
# direction, then a product-meshgrid is formed, by addition of the resulting polynomials spread over
# the range of the image size, and subtracted from the original image.

horz_mean = np.mean(aligned_med_data_Trace_Array, axis = 0) # averages all the columns into a x direction array
vert_mean = np.mean(aligned_med_data_Trace_Array, axis = 1) # averages all the rows into a y direction array

line_array = np.arange(0, row_num)
degree = 5

horz_fit = np.polyfit(line_array, horz_mean, degree)
vert_fit = np.polyfit(line_array, vert_mean, degree)

horz_polyval = -np.poly1d(horz_fit)
vert_polyval = -np.poly1d(vert_fit)
# plot meshgrid of x, y, and xy
xv, yv = np.meshgrid(horz_polyval(line_array), vert_polyval(line_array))

_min, _max = np.amin([np.amin(xv), np.amin(yv)]), np.amax([np.amax(xv), np.amax(yv)])

#-----------------
plt.rc('font', size = 4)

plt.subplot(3, 5, 1)
plt.imshow(aligned_med_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

# flattened_data_Trace_Array = aligned_med_data_Trace_Array - test_plane

# -------Next step is to calculate an optimal threshold for image binarising------



# Normalise the array such that all values lie between 0 and 1
# norm_data_Trace_Array = (flattenedC-np.min(flattened_data_Trace_Array))\
#                         / (np.max(flattened_data_Trace_Array)-np.min(flattened_data_Trace_Array))

plt.subplot(3, 5, 2)
plt.imshow(xv, extent=(0, row_num, 0, row_num), origin='lower', vmin = _min, vmax = _max,
           cmap='RdGy')
# plt.colorbar()

plt.subplot(3, 5, 3)
plt.imshow(yv, extent=(0, row_num, 0, row_num), origin='lower', vmin = _min, vmax = _max,
           cmap='RdGy')

plt.subplot(3, 5, 4)
plt.imshow(yv+xv, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

flattened_data_Trace_Array = normalise(aligned_med_data_Trace_Array + yv+xv)

plt.subplot(3, 5, 5)
plt.imshow(flattened_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

# METHOD 2
# Find the polynomial that fits over each individual row and column, form a product-meshgrid in the horizontal and
# vertical direction, apply a strong Gaussian filter over the meshgrid then subtract it from the original image.


# Make 2 zeros arrays the same size as the image

horz_array = np.zeros((row_num, row_num))
vert_array = np.zeros((row_num, row_num))
line_array = np.arange(0, row_num)
degree = 5

# Fit each row and column of the image to a polynomial of n-th degree, store the resulting plots in the above arrays

for i in range(0, row_num-1):
    row_i = aligned_med_data_Trace_Array[i, :]
    column_i = aligned_med_data_Trace_Array[:, i]
    horz_fit = np.polyfit(line_array, row_i, degree)
    vert_fit = np.polyfit(line_array, column_i, degree)
    horz_polyval2 = -np.poly1d(horz_fit)
    vert_polyval2 = -np.poly1d(vert_fit)
    horz_array[i, :] = horz_polyval2(line_array)
    vert_array[:, i] = vert_polyval2(line_array)

_min, _max = np.amin([np.amin(horz_array), np.amin(vert_array)]), np.amax([np.amax(horz_array), np.amax(vert_array)])

plt.subplot(3, 5, 6)
plt.imshow(horz_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
               cmap='RdGy')

plt.subplot(3, 5, 7)
plt.imshow(vert_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
               cmap='RdGy')

# Add the resulting arrays together

plt.subplot(3, 5, 8)
plt.imshow(horz_array+vert_array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')

# Blur the product array
gauss_mesh = ndimage.gaussian_filter(horz_array+vert_array, sigma=10)

plt.subplot(3, 5, 9)
plt.imshow(gauss_mesh, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')

# Substract the resulting plane and normalise

flattened_data_Trace_Array = normalise(aligned_med_data_Trace_Array + gauss_mesh)

plt.subplot(3, 5, 10)
plt.imshow(flattened_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')

# METHOD 3
# Similar to Method 2, but the polyfitter is augmented by nearest neighbours data using a kernel and no gaussian blur
# is applied

conv_data_Trace_Array = ndimage.gaussian_filter(aligned_med_data_Trace_Array, sigma=5)  # Gauss kernel for now
# Try this using kernels that ignore the rows/columns, so 2 convoluted data arrays

plt.subplot(3, 5, 11)
plt.imshow(conv_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')

# Make 2 zeros arrays the same size as the image

horz_array = np.zeros((row_num, row_num))
vert_array = np.zeros((row_num, row_num))
line_array = np.arange(0, row_num)
degree = 5

# Fit each row and column of the image to a polynomial of n-th degree, store the resulting plots in the above arrays

for i in range(0, row_num-1):
    row_i = conv_data_Trace_Array[i, :]
    column_i = conv_data_Trace_Array[:, i]
    horz_fit = np.polyfit(line_array, row_i, degree)
    vert_fit = np.polyfit(line_array, column_i, degree)
    horz_polyval2 = -np.poly1d(horz_fit)
    vert_polyval2 = -np.poly1d(vert_fit)
    horz_array[i, :] = horz_polyval2(line_array)
    vert_array[:, i] = vert_polyval2(line_array)

_min, _max = np.amin([np.amin(horz_array), np.amin(vert_array)]), np.amax([np.amax(horz_array), np.amax(vert_array)])

plt.subplot(3, 5, 12)
plt.imshow(horz_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
               cmap='RdGy')

plt.subplot(3, 5, 13)
plt.imshow(vert_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
               cmap='RdGy')

# Add the resulting arrays together
mesh = horz_array+vert_array

plt.subplot(3, 5, 14)
plt.imshow(mesh, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')

# Substract the resulting plane and normalise

flattened_data_Trace_Array = normalise(aligned_med_data_Trace_Array + mesh)

plt.subplot(3, 5, 15)
plt.imshow(flattened_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')


# # Consider all possible threshold values
# n = 1000
# thres = np.linspace(0, 1, n)
# pix = np.zeros((n,))
# for i, t in enumerate(thres):
#     pix[i] = np.sum(norm_data_Trace_Array < t)
#
# plt.subplot(2,2,3)
# threshold_plot = plt.plot(thres, pix)
# plt.grid(True)
#
# pix_gauss_grad = ndimage.gaussian_gradient_magnitude(pix,10)
# peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=1)
# troughs, properties = signal.find_peaks(-pix_gauss_grad, prominence=1)
#
# plt.subplot(2,2,4)
# dif_threshold_plot = plt.plot(thres, pix_gauss_grad)
# dif_threshold_scatter = plt.scatter(thres[peaks], pix_gauss_grad[peaks])
# dif_threshold_scatter = plt.scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
# plt.grid(True)
#
#
# # Print the image
# opt_thres = thres[troughs[0]]
# plt.figure()
# plt.imshow(norm_data_Trace_Array < opt_thres, extent=(0, row_num, 0, row_num), origin='lower', cmap='RdGy')

#Use this to force a plot
#plt.ion()
