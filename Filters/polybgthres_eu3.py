# Finds and processes ibws in eu_test, offers a median of differences row, phase-boundary,
# 1st phase and median aligner, applies a polynomial background subtraction, then runs two different threshold methods

# This is a more trimmed back version of the second iteration,

# Saves images to res4

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pycroscopy as scope
from sklearn.mixture import GaussianMixture

# Create an object capable of translating .ibw files
TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)

# Change the file here!
file_name = 'C12b_Ci_ring8_0006.ibw'

# A BRIEF OVERVIEW OF THE IMAGES
# # # C12_Ci4_ring5_0001.ibw - Original image I used for testing whole py and the N = 2 Gaussian Curve Fit
# # #  C12b_Ci_ring8_0006.ibw - Very successful showing, but was a fairly easy image
# # # Si_d10th_ring5_05mgmL_0005.ibw - Very successful showing, but was a fairly easy image
# C12b_Ci_ring8_0001.ibw - Garbage image that probably used the phase array for images, test with phase
#   maxes out the curve fitter
# s5b_th_ring_Eout_0001.ibw - N=2 fit successful, findpeaks less so, not a great image for thresholding either way
# Si_c2_1_0002HtT.tif - lone tif file, returns a 470 x 470 that needs grayscale and cropping, comment lines in and
#   out past the h5 closing to test this.  Image is too corrupted for use however, the centre ruins the polyfitter.
# SiO2_contacts_21_0000.ibw - The two binarisers choose different thresholds, N=2 fit looks better

file_path = r'thres_img/eu_tests/' + file_name


def trace_loader(filepath):
    # Translate the requisite file
    Output = TranslateObj.translate(
        file_path=filepath, verbose=False)
    print(Output)
    # Opening this file to read in sections as a numpy array
    read_path = Output
    h5_File = h5py.File(Output, mode='r')
    data_Trace = h5_File['Measurement_000/Channel_000/Raw_Data']
    # phase_Trace = h5_File['Measurement_000/Channel_002/Raw_Data']
    data_Trace_Array = np.array(data_Trace[:])
    # phase_Trace_Array = np.array(phase_Trace[:])
    h5_File.close()
    return data_Trace_Array


data_Trace_Array = trace_loader(file_path)

# Identify the size of the data trace array
if data_Trace_Array.shape[0] == 65536:
    row_num = 256
elif data_Trace_Array.shape[0] == 262144:
    row_num = 512
elif data_Trace_Array.shape[0] == 1048576:
    row_num = 1024

shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))

plt.rc('font', size=4)  # Set the fonts in graphs to 4

plt.figure()
plt.subplot(2, 6, 1)
plt.imshow(shaped_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
plt.title('Raw')

# ---------------Next step is to apply an alignment algorithm to the rows--------------
# Create a function to take two adjacent rows and return the alignment required to
# move the second row in line with the first

# Define a function that sets an image numpy array's values (pixel intensities) to be between 0 and 1


def normalise(array):
    norm_array = (array - np.min(array)) \
                 / (np.max(array) - np.min(array))
    return norm_array


# Where to save the resulting files, all appended with the file name and method number
sav_loc = r'thres_img/eu_tests/res4/' + file_name.replace('.ibw', '')



# raw_shaped_data_Trace_Array = normalise(shaped_data_Trace_Array)

# ## Median Alignment approach # ###
# # Align every median in the normalised array to be equal by finding the difference between the 1st and i-th row's
# # medians, and then offset all the data in that row by the same amount in order to equate them
#
# aligned_medi_data_Trace_Array = normalise(shaped_data_Trace_Array)
#
# i0_medi = np.median(aligned_medi_data_Trace_Array[0, :])
# print('Median for row 1 found to be ' + str(i0_medi))
#
# for i in range(1, row_num):
#     i_medi = np.median(aligned_medi_data_Trace_Array[i, :])
#     # print('Median for row ' + str(i + 1) + ' found to be ' + str(i_medi))
#     medi_offset = i0_medi - i_medi
#     aligned_medi_data_Trace_Array[i, :] += medi_offset
#

## # Median of Differences Alignment approach # ##


def line_align(row1, row2):
    diff = row1 - row2
    bins = np.linspace(np.min(diff), np.max(diff), 1000)
    binned_indices = np.digitize(diff, bins, right=True)
    np.sort(binned_indices)
    median_index = np.median(binned_indices)
    return bins[int(median_index)]


def mod_align(img, row_number):
    img = normalise(img)
    for j in range(1, row_number):
        row_jless1 = img[j - 1, :]
        row_j = img[j, :]
        offset = line_align(row_jless1, row_j)
        img[j, :] = img[j, :] + offset
    return img


aligned_med_data_Trace_Array = mod_align(shaped_data_Trace_Array, row_num)

plt.subplot(2, 6, 2)
plt.imshow(aligned_med_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
plt.title('Median of Differences Aligned')


def img_mask(img, deg):
    # ## APPLY IMAGE MASK ## #
    # Before running the GMM-based aligners, apply an image mask that truncates data outside pixel intensity mean +- 3
    # standard deviations to those limits accordingly, using the MoD aligned image as a base
    # The resulting offset array is then applied to the MoD aligned image.
    img_mean = np.mean(img)
    img_std = np.std(img)
    img_u_mask = img_mean + deg*img_std
    print(img_u_mask)
    img_l_mask = img_mean - deg*img_std
    print(img_l_mask)
    u_maskd = np.where(img < img_u_mask, img, img_u_mask)
    l_maskd = np.where(u_maskd > img_l_mask, u_maskd, img_l_mask)
    return l_maskd


masked_data_Trace_Array = img_mask(aligned_med_data_Trace_Array, 3)
print('Max of Unmask is ' + str(np.max(aligned_med_data_Trace_Array)) + ' while Min is '+str(np.min(aligned_med_data_Trace_Array)))
print('Max of Mask is ' + str(np.max(masked_data_Trace_Array)) + ' while Min is '+str(np.min(masked_data_Trace_Array)))

plt.subplot(2, 6, 3)
plt.imshow(masked_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
plt.title('Masked MoD Aligned')

alignment = 'phase'
#phase, bg, wbg

def BimodalMixtureThreshold(img_array, pdf_point):
    data = img_array.reshape(-1, 1)
    mix_model = GaussianMixture(2).fit(data)
    pdf_range = np.linspace(np.min(data), np.max(data), pdf_point).reshape(-1, 1)
    prob = mix_model.predict_proba(pdf_range)
    logprob = mix_model.score_samples(pdf_range)
    pdf = np.exp(logprob)
    pdf_individual = prob * pdf[:, np.newaxis]
    p_i = np.sort(np.array(
        [np.argmax(pdf_individual[:, 1]), np.argmax(pdf_individual[:, 0])]))
    # Peak Indices - indices of each peak in the individual pdfs
    # print(p_i)
    equiprob_ind = p_i[0] + np.argmin(np.abs(prob[:, 1][p_i[0]:p_i[1]] - prob[:, 0][p_i[0]:p_i[1]]))
    threshold = equiprob_ind / pdf_point
    return threshold


def BimodalMixtureBackground(img_array, pdf_point):
    data = img_array.reshape(-1, 1)
    mix_model = GaussianMixture(2).fit(data)
    pdf_range = np.linspace(np.min(data), np.max(data), pdf_point).reshape(-1, 1)
    prob = mix_model.predict_proba(pdf_range)
    logprob = mix_model.score_samples(pdf_range)
    pdf = np.exp(logprob)
    pdf_individual = prob * pdf[:, np.newaxis]
    p_i = np.sort(np.array(
        [np.argmax(pdf_individual[:, 1]), np.argmax(pdf_individual[:, 0])]))
    # Peak Indices - indices of each peak in the individual pdfs
    # print(p_i)
    bg_ind = p_i[0]
    threshold = bg_ind / pdf_point
    return threshold

if alignment == 'phase':
    ## # Phase-Boundary Alighnement approach # ##
    # Using a 2 population gaussian mixture model, the threshold required to most effectively binarise each row is
    # calculated.  The difference between threshold of the 1st and i-th row is used to offset the data in that row such that
    # they're equal and hence aligned.

    aligned_boundary_data_Trace_Array = masked_data_Trace_Array


    # aligned_boundary_data_Trace_Array[0, :] -= np.mean(aligned_boundary_data_Trace_Array[0, :])


    print("Boundary Alignment started")
    i0_thres = BimodalMixtureThreshold(aligned_boundary_data_Trace_Array[0, :], 2000)
    # print('Boundary for row 1 found at ' + str(i0_thres))

    i_thres_arr = np.zeros(row_num)
    i_thres_arr[0] = i0_thres

    for i in range(1, row_num):
        i_thres = BimodalMixtureThreshold(aligned_boundary_data_Trace_Array[i, :], 2000)
        # print('Boundary for row ' + str(i + 1) + ' found at ' + str(i_thres))
        i_thres_arr[i] = i_thres
        thres_offset = i0_thres - i_thres
        aligned_boundary_data_Trace_Array[i, :] += thres_offset

    print("Boundary Alignment ended")

    plt.subplot(2, 6, 4)
    plt.imshow(aligned_boundary_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Boundary Aligned')


elif alignment == 'bg':
    # ## 1st Phase (Background) Alignment Approach ## #
    # Using a 2 population gaussian mixture model, the pixel intensity representative height of the Si background layer is
    # calculated.  The difference between threshold of the 1st and i-th row is used to offset the data in that row such that
    # they're equal and hence aligned.
    aligned_bg_data_Trace_Array = masked_data_Trace_Array

    print("Background Alignment started")

    i0_bg = BimodalMixtureBackground(aligned_bg_data_Trace_Array[0, :], 20000)
    # print('Background for row 1 found at ' + str(i0_bg))

    i_bg_arr = np.zeros(row_num)
    i_bg_arr[0] = i0_bg

    for i in range(1, row_num):
        i_bg = BimodalMixtureBackground(aligned_bg_data_Trace_Array[i, :], 20000)
        # print('Background for row ' + str(i + 1) + ' found at ' + str(i_bg))
        i_bg_arr[i] = i_bg
        bg_offset = i0_bg - i_bg
        aligned_bg_data_Trace_Array[i, :] += bg_offset

    print("Background Alignment ended")

    plt.subplot(2, 6, 4)
    plt.imshow(aligned_bg_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Background Aligned')


elif alignment == 'wbg':
    # ## EXPERIMENTAL WEIGHTED ALIGNED ## #
    # Uses the 2 nearest rows (i+1 and i-1) in the mixture model for finding the background, locally weighting the GMM
    # In the case of the last of first row, use 0,1,2 and row_num-2, row_num-1 and row_num respectively

    aligned_mod_weighted_bg_data_Trace_Array = masked_data_Trace_Array

    print("Weighted Background Alignment started")

    i0_w_bg = BimodalMixtureBackground(aligned_mod_weighted_bg_data_Trace_Array[0:2, :], 20000)
    # print('Background for row 1 found at ' + str(i0_w_bg))
    i_w_bg_arr = np.zeros(row_num)
    i_w_bg_arr[0] = i0_w_bg

    for i in range(1, row_num):
        if i == row_num - 1:
            i_w_bg = BimodalMixtureBackground(aligned_mod_weighted_bg_data_Trace_Array[i - 2:row_num, :], 20000)
        else:
            i_w_bg = BimodalMixtureBackground(aligned_mod_weighted_bg_data_Trace_Array[i - 1:i + 2, :], 20000)

        # print('Background for row ' + str(i + 1) + ' found at ' + str(i_w_bg))
        i_w_bg_arr[i] = i_w_bg
        w_bg_offset = i0_w_bg - i_w_bg
        aligned_mod_weighted_bg_data_Trace_Array[i, :] += w_bg_offset

    print("Weighted Background Alignment ended")

    plt.subplot(2, 6, 4)
    plt.imshow(aligned_mod_weighted_bg_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Weighted Aligned')

# Strangely a leak causes the masked array to change
print('Max of Unmask is ' + str(np.max(aligned_med_data_Trace_Array)) + ' while Min is '+str(np.min(aligned_med_data_Trace_Array)))
print('Max of Mask is ' + str(np.max(masked_data_Trace_Array)) + ' while Min is '+str(np.min(masked_data_Trace_Array)))



# aligned_mod_weighted_bg_data_Trace_Array = aligned_med_data_Trace_Array


# Save the resulting images of each method to their own figure in svg format





# plt.subplot(2, 6, 3)
# plt.imshow(aligned_medi_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
#            cmap='RdGy')
# plt.title('Median Aligned')



# plt.subplot(2, 6, 10)
# plt.plot(i_thres_arr, range(0, row_num), linewidth=0.5)  # y, x
# plt.ylim(row_num-1, 0)
#

#
# plt.subplot(2, 6, 11)
# plt.plot(i_bg_arr, range(0, row_num), linewidth=0.5)  # y, x
# plt.ylim(row_num-1, 0)
# #plt.subplot(1,6,2)
#

#
# plt.subplot(2, 6, 12)
# plt.plot(i_w_bg_arr, range(0, row_num), linewidth=0.5)  # y, x
# plt.ylim(row_num-1, 0)
# #plt.subplot(1,6,2)

plt.savefig(sav_loc + 'ALIGNERS.svg')

# -------------------APPLY A POLYNOMIAL BACKGROUND REMOVAL--------------------

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
        square_differences[i - min_degree, j - min_degree] = np.sum(np.square(aligned_med_data_Trace_Array + mesh))

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

# -------------------------------PLOT THE RESULTING GRAPHS----------------------------------------
# Return the graphs provided by the method chosen, comments taken from a report.

plt.figure()
plt.subplot(3, 4, 1)
plt.imshow(shaped_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
# plt.title('Raw')
# the appearance of the raw ibw file’s data trace, the ibw file in every image in the dataset contains a data trace and
# phase trace.  This data trace is extracted from the ibw file by importing the ibw file as a hd5 file and saves the
# data trace from the hd5 as a pixel intensity grayscale array.  The colour map is a Red to Black scheme

plt.subplot(3, 4, 2)
plt.imshow(aligned_med_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
# plt.title('Median of Differences Aligned')
# the grayscale array is often 512 x 512 pixels.  The median aligner splits the array into 512 rows, calculates the
# median of each row, then row by row systematically matches the medians by raising and lowering each pixel in the
# previous row’s pixel intensity by the difference in median by the same amount.
# Wrong!  This is Median of Differences


plt.subplot(3, 4, 3)
plt.imshow(aligned_boundary_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

plt.subplot(3, 4, 5)
plt.imshow(horz_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
           cmap='RdGy')
# plt.title('Horz. Fit')
# The first of 4 stages of the polynomial background subtraction algorithm, this was the most general x-y plane method
# of the three methods chosen.  The horizontal fit is calculating by summarising the entire grayscale array vertically
# by making a horizontal array of mean pixel intensity in the vertical direction, then applying an 5th degree polynomial
# fit to the array.  The resulting fit is shown as a grid.

plt.subplot(3, 4, 6)
plt.imshow(vert_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
           cmap='RdGy')
# plt.title('Vert. Fit')
# plt.colorbar()
# Same as the Horz. Fit, but a vertical fit.  Both the fits are on the same colour map scale.

plt.subplot(3, 4, 7)
plt.imshow(mesh, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
# plt.title('Comb. Fit')
# A combined fit that is the previous two fits overlaid.


plt.subplot(3, 4, 8)
plt.imshow(norm_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
# plt.title('Plane Sub.')
# The combined fit is applied to the grayscale array and then the resulting image’s pixel intensities are normalised
# between 0 and 1.


# Save the raw ibw and processed images

plt.imsave(sav_loc + '_RAW.png', shaped_data_Trace_Array, origin='lower', cmap='gray')
plt.imsave(sav_loc + '_PROCESSED.png', norm_data_Trace_Array, origin='lower',
           cmap='gray')

plt.savefig(sav_loc + 'PROCESSES.svg')
# -----------------------------------NOW ATTEMPT TO BINARISE IMAGE--------------------------------------
# A statistical approach is taken to binarisation in the form of Otsu's thresholding.
# Both methods deployed rely on finding an intersection in the probability distribution created by the image histogram,
# a histogram that bins the pixel intensities in the grayscale image array.  This intersection value physically
# represents when a pixel has a 50/50 chance to be part of the nanoparticle layer or the substrate layer.  This relies
# on the image being bimodal, where each mode represents these two layers.




# **************FINDPEAKS METHOD******************

# # Consider all possible threshold values and use them to plot a cumulative distribution, a threshold sweep
# n = 1000  # The sensitivity of the thresholder
# thres = np.linspace(0, 1, n)
# pix = np.zeros((n,))
# L = 10
# cut_off = 0
# start = 0
#
# for i, t in enumerate(thres):
#     pix[i] = np.sum(norm_data_Trace_Array < t)
#     # Average over the previous L values and if the mean is 1% off the total number of pixels, write a cut off value.
#     # This removes the portion of the graph that is after the threshold saturates.
#     if i > (L - 2) and cut_off == 0 and np.mean(pix[np.arange(i - (L - 1), i + 1)]) > (0.99 * row_num * row_num):
#         cut_off = i
#     else:
#         cut_off = cut_off
#     # Similarly attempt to truncate the data used by recording a point where data becomes non-zero at the start of the
#     # sweep.
#     if i > (L - 2) and start == 0 and np.mean(pix[np.arange(i - (L - 1), i + 1)]) > (0.01 * row_num * row_num):
#         start = i
#     else:
#         start = start
#
# if cut_off == 0:  # A safety valve for if the above loop doesn't find a start or cut off value
#     cut_off = n - 1
# if start == n - 1:
#     start = 0
#
# print('Start found to be ' + str(start) + ' while Cut-off is ' + str(cut_off))
#
# plt.subplot(3, 4, 9)
# threshold_plot = plt.plot(thres, pix)
# start_line = plt.vlines(x=thres[start], ymin=0, ymax=row_num * row_num, colors='k', linestyles='dashed')
# cut_off_line = plt.vlines(x=thres[cut_off], ymin=0, ymax=row_num * row_num, colors='k', linestyles='dashed')
# plt.grid(True)
# # plt.title('Threshold height sweep', fontsize=10)
# plt.xlabel('Threshold', fontsize=3)
# plt.ylabel('Pixels', fontsize=3)
# # plt.title('Threshold Sweep')
# # 1000 threshold values spaced between 0 to 1 are applied to the grayscale array.  A logic statement sets pixels to 0 if
# # they take a pixel intensity value below the particular threshold value in a loop, or 1 if they’re above, then a sum
# # function sees how many pixels took a value of 1.  Trailing data ie when the threshold value finds every pixel to take
# # a value of 1 is removed in later graphs.  This truncates the graph to add more precision to some fits, but probably
# # not as effective as histogram normalisation.
#
#
# # Fit a Gaussian-smoothed curve to the differential of the threshold sweep graph between the start and the cut off value
# gauss_sigma = 10
# pix_gauss_grad = ndimage.gaussian_gradient_magnitude(pix[0:cut_off + 1], gauss_sigma)
#
# # Identify the peaks and troughs of the signal
# peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=1)
# troughs, properties2 = signal.find_peaks(-pix_gauss_grad, prominence=1)
#
# plt.subplot(3, 4, 11)
# dif_threshold_plot = plt.plot(thres[0:cut_off + 1], pix_gauss_grad)
# dif_threshold_scatter = plt.scatter(thres[peaks], pix_gauss_grad[peaks])
# dif_threshold_scatter2 = plt.scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
# plt.grid(True)
# # plt.title('Gaussian gradient (\u03C3=' + str(gauss_sigma) + ')', fontsize=10)
# plt.xlabel('Threshold', fontsize=3)
# plt.ylabel('\u0394Pixels', fontsize=3)
# # plt.title('Intensity Peaks')
# # The Gaussian-smoothed image histogram derived from the 1st-derivative of the cumulative distribution.  The peaks and
# # troughs are marked, found by the findpeaks function.
#
# # If at least one trough is found, set that as the optimal threshold value for binarisation, otherwise move on
# if len(troughs) < 1:
#     print('Rejected! No clear optimisation.')
#     opt_thres = 0
#
# else:
#     opt_thres = thres[troughs[len(troughs) - 1]]
#     print('Threshold from histogram fit found to be %.3f' % opt_thres + '.')
#
# # Save a binarised image (if possible) as well as all the subplots created during processing
# if opt_thres != 0:
#     plt.subplot(3, 4, 12)
#     plt.imshow(norm_data_Trace_Array > opt_thres, extent=(0, row_num, 0, row_num), origin='lower', cmap='gray')
#     # plt.title('Final')
#     plt.savefig(sav_loc + '.png')
#     plt.savefig(sav_loc + '.svg')
#     # plt.ion() # comment out to suppress output of a bunch of images
#     plt.imsave(sav_loc + '_THRES.png', norm_data_Trace_Array > opt_thres,
#                cmap='gray')  # The image is flipped vertically
# else:
#     plt.savefig(sav_loc + '.png')
#     plt.savefig(sav_loc + '.svg')
#
# # Save the image histogram fit as a separate image file too
# plt.figure()
# dif_threshold_plot = plt.plot(thres[0:cut_off + 1], pix_gauss_grad)
# dif_threshold_scatter = plt.scatter(thres[peaks], pix_gauss_grad[peaks])
# dif_threshold_scatter2 = plt.scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
# plt.grid(True)
# plt.savefig(sav_loc + '_HISTFIT.png')
#
# # **************CURVE FIT METHOD********************
# # Fit 2 Gaussians to the image histogram using a curvefit function and uses their intercept as the threshold value
#
# # Find the image histogram
# plt.figure()
# hist, bin_edges = np.histogram(norm_data_Trace_Array.ravel(), bins=256, range=(0.0, 1.0), density=True)
# bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#
# # Define a gaussian function and a mixture model consisting of two gaussians
# def gauss(x, *p):
#     A, mu, sigma = p
#     return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))
#
#
# def gauss2(x, *p):
#     A1, mu1, sigma1, A2, mu2, sigma2 = p
#     return A1 * np.exp(-(x - mu1) ** 2 / (2. * sigma1 ** 2)) + A2 * np.exp(-(x - mu2) ** 2 / (2. * sigma2 ** 2))
#
#
# # p0 is the initial guess for the fitting coefficients initialize them differently so the optimisation algorithm works
# #   better
# p0 = [1., -1., 1., 1., -1., 1.]
#
# # Optimise and in the end you will have 6 coeff (3 for each gaussian)
# coeff, var_matrix = curve_fit(gauss2, bin_centres, hist, p0=p0)
#
# # You can plot each gaussian separately using the defined single gaussian function by splitting up the coefficients
# pg1 = coeff[0:3]
# pg2 = coeff[3:]
#
# g1 = gauss(bin_centres, *pg1)
# g2 = gauss(bin_centres, *pg2)
#
# plt.plot(bin_centres, hist, label='Data')
# plt.plot(bin_centres, g1, label='Gaussian1')
# plt.plot(bin_centres, g2, label='Gaussian2')
# plt.savefig(sav_loc + 'N2.png')
# # Plot all 3 curves, data should really be a barchart
# hist_thres = np.mean([pg1[1], pg2[1]])  # Finds the midpoint of the 2 mu values, probably should use intersect instead
#
# # Save the binary image
# plt.imsave(sav_loc + 'THRES_CF.png', norm_data_Trace_Array > hist_thres, cmap='gray')
#
# # FIND A HISTOGRAM OF THE IMAGE PIXEL INTENSITY DATA BEFORE THE CUT-OFF POINT FOR A CROPPED SMOOTHER GRAPH
#
# # Use the start and cut-off value for the range of a histogram
# # in range replace 0.0 with start/1000 and 1.0 with cut_off/1000 and vice versa to test truncation
# hist_c, bin_edges_c = np.histogram(norm_data_Trace_Array, bins=256, range=(0.0, cut_off / 1000), density=True)
# bin_centres_c = (bin_edges_c[:-1] + bin_edges_c[1:]) / 2
#
# p0 = [1., -1., 1., 1., -1., 1.]
#
# coeff, var_matrix = curve_fit(gauss2, bin_centres_c, hist_c, p0=p0)
#
# pg1 = coeff[0:3]
# pg2 = coeff[3:]
#
# g1_c = gauss(bin_centres_c, *pg1)
# g2_c = gauss(bin_centres_c, *pg2)
#
# plt.figure()
# plt.plot(bin_centres_c, hist_c, label='Data')
# plt.plot(bin_centres_c, g1_c, label='Gaussian1')
# plt.plot(bin_centres_c, g2_c, label='Gaussian2')
# plt.savefig(sav_loc + 'N2CROP.png')
#
# hist_thres = np.mean([pg1[1], pg2[1]])
#
# plt.imsave(sav_loc + 'THRES_CF_CROP.png', norm_data_Trace_Array > hist_thres, cmap='gray')
#

#
# # -----------------------------------------------------------------------------
# # THIS SECTION IS VERY MESSY AND CARGOCULT SO NEEDS FIXING
#
# # ----------------------------------------------------------------------
# # This function adjusts matplotlib settings for a uniform feel in the textbook.
# # Note that with usetex=True, fonts are rendered with LaTeX.  This may
# # result in an error if LaTeX is not installed on your system.  In that case,
# # you can set usetex to False.
# if "setup_text_plots" not in globals():
#     from astroML.plotting import setup_text_plots
# setup_text_plots(fontsize=8, usetex=False)
#
# # ------------------------------------------------------------
# # Set up the dataset.
# #  We'll create our dataset by drawing samples from Gaussians.
#
# random_state = np.random.RandomState(seed=1)
#
# X = np.concatenate([random_state.normal(-1, 1.5, 350),
#                     random_state.normal(0, 1, 500),
#                     random_state.normal(3, 0.5, 150)]).reshape(-1, 1)
#
# X = norm_data_Trace_Array.reshape(-1, 1)
#
# # ------------------------------------------------------------
# # Learn the best-fit GaussianMixture models
# #  Here we'll use scikit-learn's GaussianMixture model. The fit() method
# #  uses an Expectation-Maximization approach to find the best
# #  mixture of Gaussians for the data
#
# # fit models with 1-10 components
# # N = np.arange(1, 11)
# # models = [None for i in range(len(N))]
# #
# # for i in range(len(N)):
# #     models[i] = GaussianMixture(N[i]).fit(X)
# #
# # # compute the AIC and the BIC
# # AIC = [m.aic(X) for m in models]
# # BIC = [m.bic(X) for m in models]
#
# model = GaussianMixture(2).fit(X)
#
# # ------------------------------------------------------------
# # Plot the results
# #  We'll use three panels:
# #   1) data + best-fit mixture
# #   2) AIC and BIC vs number of components
# #   3) probability that a point came from each component
#
# fig = plt.figure(figsize=(5, 1.7))
# fig.subplots_adjust(left=0.12, right=0.97,
#                     bottom=0.21, top=0.9, wspace=0.5)
#
# # plot 1: data + best-fit mixture
# ax = fig.add_subplot(111)
# # M_best = models[np.argmin(AIC)]
# M_best = model  # Only tries N=2 rn
#
# pdf_points = 5000  # The number of points for plotting the pdf smoothly
# x = np.linspace(0, 1, 5000)  # range is fudged, return to 0 to 1
# logprob = M_best.score_samples(x.reshape(-1, 1))
# responsibilities = M_best.predict_proba(x.reshape(-1, 1))
# pdf = np.exp(logprob)
# pdf_individual = responsibilities * pdf[:, np.newaxis]
#
# p_i = np.sort(np.array(
#         [np.argmax(pdf_individual[:, 1]), np.argmax(pdf_individual[:, 0])]))
# # Peak Indices - indices of each peak in the individual pdfs
# # print(p_i)
# equiprob_ind = p_i[0] + np.argmin(np.abs(responsibilities[:, 1][p_i[0]:p_i[1]] - responsibilities[:, 0][p_i[0]:p_i[1]]))
#
# # opt_ind = np.argmin(np.abs(responsibilities[:, 1] - responsibilities[:, 0]))
# # opt_thres = opt_ind / pdf_points
#
# opt_thres = BimodalMixtureThreshold(X, pdf_points)
#
# print('Threshold from mixture model found to be %.3f' % opt_thres + '.')
#
# ax.hist(X, 256, density=True, histtype='stepfilled', alpha=0.4)
# ax.plot(x, pdf, '-k')
# ax.plot(x, pdf_individual, '--r', linewidth=0.5)
# ax.scatter(x[equiprob_ind], pdf_individual[:, 0][equiprob_ind])
# # ax.text(0.04, 0.96, "Best-fit Mixture",
# #         ha='left', va='top', transform=ax.transAxes)
# ax.set_xlabel('$x$')
# ax.set_ylabel('$p(x)$')
# plt.savefig(sav_loc + 'MIXTURE.svg')
#
# plt.imsave(sav_loc + 'THRES_MIXTURE.png', norm_data_Trace_Array > opt_thres, cmap='gray')
#
# # # plot 2: AIC and BIC
# # ax = fig.add_subplot(132)
# # ax.plot(N, AIC, '-k', label='AIC')
# # ax.plot(N, BIC, '--k', label='BIC')
# # ax.set_xlabel('n. components')
# # ax.set_ylabel('information criterion')
# # ax.legend(loc=2)
