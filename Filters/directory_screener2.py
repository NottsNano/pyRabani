# -*- coding: utf-8 -*-

import pycroscopy as scope
import matplotlib.pyplot as plt
import h5py
import pyUSID
from scipy import ndimage, signal, stats, misc
import numpy as np
import os
import sys

# # Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
#
# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__

# Toggles showing progress statements
# 1 is yes, 0 is no
p_s = 1
# Toggles choosing a folder picking manually instead of the one with true positives (tp)
# 1 is yes to choose manually, 0 uses default (tp)
f_p = 0

if f_p == 1:  # Directory chooser dialogue
    fol = input('Which folder?')
    if fol not in os.listdir('thres_img'):
        print('Defaulting to /tp/')
        fol = 'tp'
    dir = str('thres_img/' + fol)
else:
    dir = 'thres_img/tp/'

print('Locating ibw files in ' + dir + '...')

files = os.listdir(dir)
files_ibw = [i for i in files if i.endswith('.ibw')]

# Make a bunch of empty arrays for the screening tables
row_width = np.zeros(files_ibw.__len__())
row_pure = np.zeros(files_ibw.__len__())
opt_peak = np.zeros(files_ibw.__len__())
successes = 0

for k in files_ibw:
    print('\n Processing ibw file ' + str(files_ibw.index(k) + 1) + ' of ' + str(files_ibw.__len__()) + ': ' + k)

    if p_s == 0:
        sys.stdout = open(os.devnull, 'w')

    print('* Translating h5 file of ' + k + '...')

    # Create an object capable of translating .ibw files
    TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)
    # path2file = f'thres_img/tp/000TEST.ibw'
    path2file = dir + k
    # path2file = f'thres_img/tp/C10_0000.ibw' # No clue whats wrong with this one
    # path2file = f'thres_img/tn/SiO2_t12_ring5_1mgmL_0000.ibw' # Image for testing dud line finder

    # Translate the requisite file
    Output = TranslateObj.translate(
        file_path=path2file, verbose=False)

    print('* Translated file ' + Output + '. Saving the hd5 file contents to an array', end='... ')

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
        print('* Rejected! ibw file is non-standard size. Moving to next file')
        row_num = np.nan
        row_pure[files_ibw.index(k)] = np.nan
        opt_peak[files_ibw.index(k)] = np.nan
        continue

    row_width[files_ibw.index(k)] = row_num

    # Shape and normalise the arrays between 0 and 1
    shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))
    norm_data_Trace_Array = (shaped_data_Trace_Array - np.min(shaped_data_Trace_Array)) \
                            / (np.max(shaped_data_Trace_Array) - np.min(shaped_data_Trace_Array))

    shaped_phase_Trace_Array = np.reshape(phase_Trace_Array, (row_num, row_num))
    norm_phase_Trace_Array = (shaped_phase_Trace_Array - np.min(shaped_phase_Trace_Array)) \
                             / (np.max(shaped_phase_Trace_Array) - np.min(shaped_phase_Trace_Array))

    h5_File.close()
    print('Done. Closed hd5 file.')

    # ---------------Next step is to apply median difference to rows--------------
    # Create a function to take two adjacent rows and return the alignment required to
    # move the second row in line with the first

    print('* Applying median aligner', end='... ')


    def line_align(row1, row2):
        diff = row1 - row2
        bins = np.linspace(np.min(diff), np.max(diff), 1000)
        binned_indices = np.digitize(diff, bins, right=True)
        np.sort(binned_indices)
        median_index = np.median(binned_indices)
        return bins[int(median_index)]


    row_fit_data_Trace_Array = norm_data_Trace_Array
    row_fit_data_Trace_Array[1, :] = norm_data_Trace_Array[1, :] - np.mean(norm_data_Trace_Array[1, :])

    aligned_med_data_Trace_Array = row_fit_data_Trace_Array
    aligned_med_phase_Trace_Array = norm_phase_Trace_Array

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
        # Count rows where the mode is the same as 95% of the values in the row or ie a dead scan line
        row_mode = stats.mode(row_i)
        # counter = np.count_nonzero(row_i == (stats.mode(row_i))[0])
        counter = np.count_nonzero((0.95 * row_mode[0] < row_i) < 1.05 * row_mode[0]) # counts how many rows are just the mode value
        flat = counter > 0.95 * row_num
        #  -----------------------------------------------------------
        slope = row_i == np.sort(row_i)
        slope_rev = row_i == np.sort(row_i)[::-1]
        dud_row += (counter > (0.95 * row_num))\
                  + sum(slope) > (0.95 * row_num) + sum(slope_rev) > (0.95 * row_num)
        #  -----------------------------------------------------------
        # Still not great, maybe has the line fit to a polynomial and use a chi fit

    dud_perc = 100 * dud_row / row_num
    row_pure[files_ibw.index(k)] = "%.1f" % dud_perc

    if dud_perc >= 5:
        print('Rejected! ' + str("%.1f" % dud_perc) + '% of the rows were corrupt.')
    else:
        print('Done.')

    # ----------------Next step is to flatten the surface-------------------------

    print('* Applying planar flattening', end='... ')

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
    print('Done.')

    # The plane is removed
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(aligned_med_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Median aligned', fontsize=10)

    flattened_data_Trace_Array = aligned_med_data_Trace_Array - test_plane

    # -------Next step is to calculate an optimal threshold for image binarising------

    # Normalise the array such that all values lie between 0 and 1
    norm_data_Trace_Array = (flattened_data_Trace_Array - np.min(flattened_data_Trace_Array)) \
                            / (np.max(flattened_data_Trace_Array) - np.min(flattened_data_Trace_Array))

    plt.subplot(2, 2, 2)
    plt.imshow(norm_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Planar flattened', fontsize=10)

    # Consider all possible threshold values
    print('* Finding optimal threshold value', end='... ')
    n = 1000
    thres = np.linspace(0, 1, n)
    pix = np.zeros((n,))
    for i, t in enumerate(thres):
        pix[i] = np.sum(norm_data_Trace_Array < t)

    plt.subplot(2, 2, 3)
    threshold_plot = plt.plot(thres, pix)
    plt.grid(True)
    # plt.title('Threshold height sweep', fontsize=10)
    plt.xlabel('Threshold', fontsize=8)
    plt.ylabel('Pixels', fontsize=8)

    gauss_sigma = 10
    pix_gauss_grad = ndimage.gaussian_gradient_magnitude(pix, gauss_sigma)
    peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=1)
    troughs, properties2 = signal.find_peaks(-pix_gauss_grad, prominence=1)

    plt.subplot(2, 2, 4)
    dif_threshold_plot = plt.plot(thres, pix_gauss_grad)
    dif_threshold_scatter = plt.scatter(thres[peaks], pix_gauss_grad[peaks])
    dif_threshold_scatter2 = plt.scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
    plt.grid(True)
    # plt.title('Gaussian gradient (\u03C3=' + str(gauss_sigma) + ')', fontsize=10)
    plt.xlabel('Threshold', fontsize=8)
    plt.ylabel('\u0394Pixels', fontsize=8)
    fig_loc = dir + 'tempr/' + k.replace('.ibw', 'FIG.png')
    plt.savefig(fig_loc)
    # , bbox = 'tight'

    if len(troughs) < 1:
        print('Rejected! No clear optimisation.')
        opt_thres = 0
        opt_peak[files_ibw.index(k)] = np.nan

    else:
        opt_thres = thres[troughs[len(troughs)-1]]
        print('Threshold found to be %.2f' % opt_thres + '.')
        opt_peak[files_ibw.index(k)] = opt_thres

    if opt_thres != 0 and dud_perc < 5:
        # Print the image
        plt.figure()
        plt.imshow(norm_data_Trace_Array > opt_thres, extent=(0, row_num, 0, row_num), origin='lower', cmap='gray')
        plt.title(k.replace('.ibw', '') + ' - Threshold = %.2f' % opt_thres)
        # plt.ion() # comment out to suppress output of a bunch of images
        thres_loc = dir + 'tempr/' + k.replace('.ibw', 'THRES.png')
        print('* ' + k.replace('.ibw', '') + ' passed all checks. Saving image.')
        plt.imsave(thres_loc, norm_data_Trace_Array > opt_thres, cmap='gray')  # The image is flipped vertically, sorry
        successes = successes + 1

    else:
        print('* ' + (k.replace('.ibw', '') + ' did not pass all checks.'))

    plt.close('all') # comment out when plt.ion() not commented out

    if p_s == 0:
        sys.stdout = sys.__stdout__

print('\nDone! ' + str(successes) + ' of ' + str(files_ibw.__len__()) + ' files passed.  Saving results to table...')

# Save all values to a table with pandas

# Use this to force a plot
# plt.ion()
