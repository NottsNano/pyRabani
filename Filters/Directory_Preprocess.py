import pycroscopy as scope
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os


def preprocess(ibw_name, ibw_path):
    # Create an object capable of translating .ibw files
    TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)

    # Translate the requisite file
    Output = TranslateObj.translate(
        file_path=ibw_path, verbose=False)

    print(Output)

    # Opening this file to read in sections as a numpy array
    # Read_Path = Output
    h5_File = h5py.File(Output, mode='r')

    data_Trace = h5_File['Measurement_000/Channel_000/Raw_Data']
    # phase_Trace = h5_File['Measurement_000/Channel_002/Raw_Data']

    data_Trace_Array = np.array(data_Trace[:])
    # phase_Trace_Array = np.array(phase_Trace[:])

    # Identify the size of the data trace array
    if data_Trace_Array.shape[0] == 65536:
        row_num = 256
    elif data_Trace_Array.shape[0] == 262144:
        row_num = 512
    elif data_Trace_Array.shape[0] == 1048576:
        row_num = 1024
    else:
        row_num = 0
        norm_data_Trace_Array = 0

    h5_File.close()
    os.remove(Output)
    # print('Data trace array found to have ' + str(row_num) + 'rows')

    if row_num > 0:
        shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))

        # shaped_phase_Trace_Array = np.reshape(phase_Trace_Array, (row_num, row_num))

        # ---------------Next step is to apply median difference to rows--------------
        # Create a function to take two adjacent rows and return the alignment required to
        # move the second row in line with the first

        # Phase components commented out to speed up code

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
        # aligned_med_phase_Trace_Array = shaped_phase_Trace_Array

        for i in range(1, row_num):
            row_iless1 = aligned_med_data_Trace_Array[i - 1, :]
            row_i = aligned_med_data_Trace_Array[i, :]
            Offset = line_align(row_iless1, row_i)
            aligned_med_data_Trace_Array[i, :] = aligned_med_data_Trace_Array[i, :] + Offset

            # row_iless1 = aligned_med_phase_Trace_Array[i - 1, :]
            # row_i = aligned_med_phase_Trace_Array[i, :]
            # Offset = line_align(row_iless1, row_i)
            # aligned_med_phase_Trace_Array[i, :] = aligned_med_phase_Trace_Array[i, :] + Offset

        # -------------------APPLY A POLYNOMIAL BACKGROUND REMOVAL--------------------

        # Define a function that sets an image numpy array's values (pixel intensities) to be between 0 and 1
        def normalise(array):
            norm_array = (array - np.min(array)) \
                         / (np.max(array) - np.min(array))
            return norm_array

        plt.rc('font', size=4)  # Set the fonts in graphs to 4
        line_array = np.arange(0, row_num)  # x-direction array used for the polynomial plots
        degree = 5  # Determines the n in the n-th degree polynomial fits of method 2 and 3

        # Finds the mean of every row combined into a 1d array to summarise the image in a vertical direction, and the same
        # is done for every column for the horizontal direction.  An n-th degree polynomial is fitted over the array in each
        # direction, then a product-meshgrid is formed, by addition of the resulting polynomials spread over
        # the range of the image size, and subtracted from the original image.  The square difference is found and this
        # process is repeated for different degrees until the smallest square difference is found for degrees in x and y
        # separately, this is the optimal polynomial background for the provided range.

        horz_mean = np.mean(aligned_med_data_Trace_Array, axis=0)  # averages all the columns into a x direction array
        vert_mean = np.mean(aligned_med_data_Trace_Array, axis=1)  # averages all the rows into a y direction array

        # Provide a minimal and maximum degree of polynomial that will be attempted to fit over the image
        max_degree = 5
        min_degree = 1
        square_differences = np.zeros([max_degree - min_degree + 1, max_degree - min_degree + 1])

        # Define functions for finding the polynomial fit in the x and y direction
        def polybgfitter_i(horz_mean, line_array, degree_i):
            horz_fit = np.polyfit(line_array, horz_mean, degree_i)
            horz_polyval = -np.poly1d(horz_fit)
            return horz_polyval

        def polybgfitter_j(vert_mean, line_array, degree_j):
            horz_fit = np.polyfit(line_array, vert_mean, degree_j)
            horz_polyval = -np.poly1d(horz_fit)
            return horz_polyval

        # Fit polynomials between and including the provided minimum and maximum degrees individually in the x and y
        # directions, creating a mesh-grid for each possible combination and recording the square difference between the
        # mesh grid and image array.
        for i in range(min_degree, max_degree + 1):
            i_polyval = polybgfitter_i(horz_mean, line_array, i)
            for j in range(min_degree, max_degree + 1):
                j_polyval = polybgfitter_i(horz_mean, line_array, j)
                horz_array, vert_array = np.meshgrid(i_polyval(line_array), j_polyval(line_array))
                mesh = horz_array + vert_array
                square_differences[i - min_degree, j - min_degree] = np.sum(
                    np.square(aligned_med_data_Trace_Array + mesh))

        # Find the index of the lowest square difference value and use to determine the returned meshgrid
        bestindices = np.unravel_index(np.argmin(square_differences, axis=None), square_differences.shape)
        best_i_polyval = polybgfitter_i(horz_mean, line_array, bestindices[0] + min_degree)
        best_j_polyval = polybgfitter_j(vert_mean, line_array, bestindices[1] + min_degree)
        horz_array, vert_array = np.meshgrid(best_i_polyval(line_array), best_j_polyval(line_array))
        mesh = horz_array + vert_array

        # Apply the mesh to the image array and normalise
        norm_data_Trace_Array = normalise(aligned_med_data_Trace_Array + mesh)

        # Find the minimum and maximum values in the horizontal and vertical arrays of the mesh grid, such that they can be
        # plotted on the same colourbar later
        _min, _max = np.amin([np.amin(horz_array), np.amin(vert_array)]), np.amax([np.amax(horz_array), np.amax(
            vert_array)])

        # # -------------------------------PLOT THE RESULTING GRAPHS----------------------------------------
        # # Return the graphs provided by the method chosen, comments taken from a report.
        #
        # plt.subplot(3, 4, 1)
        # plt.imshow(shaped_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
        #            cmap='RdGy')
        # # plt.title('Raw')
        # # the appearance of the raw ibw file’s data trace, the ibw file in every image in the dataset contains a data trace and
        # # phase trace.  This data trace is extracted from the ibw file by importing the ibw file as a hd5 file and saves the
        # # data trace from the hd5 as a pixel intensity grayscale array.  The colour map is a Red to Black scheme
        #
        # plt.subplot(3, 4, 2)
        # plt.imshow(aligned_med_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
        #            cmap='RdGy')
        # # plt.title('Median Aligned')
        # # the grayscale array is often 512 x 512 pixels.  The median aligner splits the array into 512 rows, calculates the
        # # median of each row, then row by row systematically matches the medians by raising and lowering each pixel in the
        # # previous row’s pixel intensity by the difference in median by the same amount.
        #
        # plt.subplot(3, 4, 5)
        # plt.imshow(horz_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
        #            cmap='RdGy')
        # # plt.title('Horz. Fit')
        # # The first of 4 stages of the polynomial background subtraction algorithm, this was the most general x-y plane method
        # # of the three methods chosen.  The horizontal fit is calculating by summarising the entire grayscale array vertically
        # # by making a horizontal array of mean pixel intensity in the vertical direction, then applying an 5th degree polynomial
        # # fit to the array.  The resulting fit is shown as a grid.
        #
        # plt.subplot(3, 4, 6)
        # plt.imshow(vert_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
        #            cmap='RdGy')
        # # plt.title('Vert. Fit')
        # # plt.colorbar()
        # # Same as the Horz. Fit, but a vertical fit.  Both the fits are on the same colour map scale.
        #
        # plt.subplot(3, 4, 7)
        # plt.imshow(mesh, extent=(0, row_num, 0, row_num), origin='lower',
        #            cmap='RdGy')
        # # plt.title('Comb. Fit')
        # # A combined fit that is the previous two fits overlaid.
        #
        # plt.subplot(3, 4, 8)
        # plt.imshow(norm_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
        #            cmap='RdGy')
        # # plt.title('Plane Sub.')
        # # The combined fit is applied to the grayscale array and then the resulting image’s pixel intensities are normalised
        # # between 0 and 1.

        # Where to save the resulting files, all appended with the file name and method number
        sav_loc = r'res/' + ibw_name + '.png'

        # Save the raw ibw and processed images

        # plt.imsave(sav_loc + '_RAW.png', shaped_data_Trace_Array, origin='lower', cmap='gray')
        plt.imsave(sav_loc, norm_data_Trace_Array, origin='lower',
                   cmap='gray')

    return norm_data_Trace_Array




import pandas as pd
import glob

# # Below few lines used for targeting unique folders, commented out when looking at whole USBs
import tkinter
from tkinter import filedialog
# root = tkinter.Tk()
# root.withdraw()
# dirname = filedialog.askdirectory(parent=root, initialdir="/", title='Please select a directory')

dirname = 'D:/USB 3'

usb_num = 3  # Change this to reflect the USB number for the ibw location

Data = pd.read_csv('ManualImageClassificationsV2_3.csv', sep=',')

FileNames = Data['Predicted ibw file name']


for k in range(0, FileNames.size):
    print('Searching for image ', k + 1, ' of ', FileNames.size)
    locate = dirname + '/**/' + FileNames[k]
    matches = glob.glob(locate, recursive=True)
    print('~~~ ' + str(len(matches)) + ' found')
    if len(matches) > 0:
        col_name = 'Ibw matches ' + str(usb_num)
        new_col = pd.Series(Data[col_name][k]+len(matches), name=col_name, index=[k])
        Data.update(new_col)
        for j in range(0, len(matches)):
            file_name = FileNames[k].replace('.ibw', '') + '_' + str(usb_num) + '_' + str(j)
            preprocess(file_name, matches[j])  # Run this commented out to just count the number of found images


# Run this to save changes to the csv if doesn't reach end
Data.to_csv('ManualImageClassificationsV2_3.csv', index=False)
