# This code was used in testing of a few unique images provided by University of Loughborough.
# It deployed 3 unique techniques to remove the polynomial background ie inherent wavy-ness of AFM images
# as well as tested a new way to find the perfect binary image of each image, using Otsu's thresholding.
# Images are aligned horizontally using a median of differences row aligner

import pycroscopy as scope
import matplotlib.pyplot as plt
import h5py
from scipy import ndimage, signal
from scipy.optimize import curve_fit
import numpy as np

# Choose one of the three methods, 1, 2 or 3, they're each explained within their respective lines
method = 1

# Create an object capable of translating .ibw files
TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)

# Change the file here!
file_name = 'Si_d10th_ring5_05mgmL_0005.ibw'

# A BRIEF OVERVIEW OF THE IMAGES
# C12_Ci4_ring5_0001.ibw - Original image I used for testing whole py and the N = 2 Gaussian Curve Fit
# C12b_Ci_ring8_0001.ibw - Garbage image that probably used the phase array for images, test with phase
#   maxes out the curve fitter
# C12b_Ci_ring8_0006.ibw - Very successful showing, but was a fairly easy image
# s5b_th_ring_Eout_0001.ibw - N=2 fit successful, findpeaks less so, not a great image for thresholding either way
# Si_c2_1_0002HtT.tif - lone tif file, returns a 470 x 470 that needs grayscale and cropping, comment lines in and
#   out past the h5 closing to test this.  Image is too corrupted for use however, the centre ruins the polyfitter.
# Si_d10th_ring5_05mgmL_0005.ibw - Very successful showing, but was a fairly easy image
# SiO2_contacts_21_0000.ibw - The two binarisers choose different thresholds, N=2 fit looks better

# Translate the requisite file
Output = TranslateObj.translate(
    file_path=r'thres_img/eu_tests/' + file_name, verbose=False)

print(Output)

# Opening this file to read in sections as a numpy array
Read_Path = Output
h5_File = h5py.File(Output, mode='r')

data_Trace = h5_File['Measurement_000/Channel_000/Raw_Data']
phase_Trace = h5_File['Measurement_000/Channel_002/Raw_Data']
data_Trace_Array = np.array(data_Trace[:])
phase_Trace_Array = np.array(phase_Trace[:])

# Identify the size of the data trace array
if data_Trace_Array.shape[0] == 65536:
    row_num = 256
elif data_Trace_Array.shape[0] == 262144:
    row_num = 512
elif data_Trace_Array.shape[0] == 1048576:
    row_num = 1024

shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))
shaped_phase_Trace_Array = np.reshape(phase_Trace_Array, (row_num, row_num))

h5_File.close()

# # Code used for testing the lone tif file, comment out when not in use
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
# file_name = 'Si_c2_1_0002HtT.tif'
# shaped_data_Trace_Array = plt.imread(r'thres_img/eu_tests/Si_c2_1_0002HtT.tif')
# shaped_data_Trace_Array = rgb2gray(shaped_data_Trace_Array)
# row_num = shaped_data_Trace_Array.shape[0]


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
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array

plt.rc('font', size=4)  # Set the fonts in graphs to 4
line_array = np.arange(0, row_num)  # x-direction array used for the polynomial plots
degree = 5  # Determines the n in the n-th degree polynomial fits of method 2 and 3

if method == 1:
    # METHOD 1

    # Finds the mean of every row combined into a 1d array to summarise the image in a vertical direction, and the same
    # is done for every column for the horizontal direction.  An n-th degree polynomial is fitted over the array in each
    # direction, then a product-meshgrid is formed, by addition of the resulting polynomials spread over
    # the range of the image size, and subtracted from the original image.  The square difference is found and this
    # process is repeated for different degrees until the smallest square difference is found for degrees in x and y
    # separately, this is the optimal polynomial background.

    horz_mean = np.mean(aligned_med_data_Trace_Array, axis=0)  # averages all the columns into a x direction array
    vert_mean = np.mean(aligned_med_data_Trace_Array, axis=1)  # averages all the rows into a y direction array

    # Provide a minimal and maximum degree of polynomial that will be attempted to fit over the image
    max_degree = 5
    min_degree = 1
    square_differences = np.zeros([max_degree-min_degree+1, max_degree-min_degree+1])

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
            square_differences[i-min_degree, j-min_degree] = np.sum(np.square(aligned_med_data_Trace_Array + mesh))

    # Find the index of the lowest square difference value and use to determine the returned meshgrid
    bestindices = np.unravel_index(np.argmin(square_differences, axis=None), square_differences.shape)
    best_i_polyval = polybgfitter_i(horz_mean, line_array, bestindices[0]+min_degree)
    best_j_polyval = polybgfitter_j(vert_mean, line_array, bestindices[1]+min_degree)
    horz_array, vert_array = np.meshgrid(best_i_polyval(line_array), best_j_polyval(line_array))
    mesh = horz_array + vert_array

    # Apply the mesh to the image array and normalise
    norm_data_Trace_Array = normalise(aligned_med_data_Trace_Array + mesh)

    # Find the minimum and maximum values in the horizontal and vertical arrays of the mesh grid, such that they can be
    # plotted on the same colourbar later
    _min, _max = np.amin([np.amin(horz_array), np.amin(vert_array)]), np.amax([np.amax(horz_array), np.amax(
        vert_array)])




elif method == 2:
    # METHOD 2
    # Find the polynomial that fits over each individual row and column, form a product-meshgrid in the horizontal and
    # vertical direction, apply a strong Gaussian filter over the meshgrid then subtract it from the original image.

    # Make 2 zeros arrays the same size as the image
    horz_array = np.zeros((row_num, row_num))
    vert_array = np.zeros((row_num, row_num))

    # Fit each row and column of the image to a polynomial of n-th degree, store the resulting plots in the above arrays

    for i in range(0, row_num - 1):
        row_i = aligned_med_data_Trace_Array[i, :]
        column_i = aligned_med_data_Trace_Array[:, i]
        horz_fit = np.polyfit(line_array, row_i, degree)
        vert_fit = np.polyfit(line_array, column_i, degree)
        horz_polyval2 = -np.poly1d(horz_fit)
        vert_polyval2 = -np.poly1d(vert_fit)
        horz_array[i, :] = horz_polyval2(line_array)
        vert_array[:, i] = vert_polyval2(line_array)

    # Add the resulting arrays together and blur them
    mesh = horz_array + vert_array
    gauss_mesh = ndimage.gaussian_filter(mesh, sigma=10)

    _min, _max = np.amin([np.amin(horz_array), np.amin(vert_array)]), np.amax(
        [np.amax(horz_array), np.amax(vert_array)])


    # Substract the resulting plane and normalise
    norm_data_Trace_Array = normalise(aligned_med_data_Trace_Array + gauss_mesh)


elif method == 3:
    # METHOD 3
    # Similar to Method 2, but the polyfitter is augmented by nearest neighbours data using a kernel and no gaussian blur
    # is applied

    conv_data_Trace_Array = ndimage.gaussian_filter(aligned_med_data_Trace_Array, sigma=5)  # Gauss kernel for now
    # Try this using kernels that ignore the rows/columns, so 2 convoluted data arrays

    # Make 2 zeros arrays the same size as the image
    horz_array = np.zeros((row_num, row_num))
    vert_array = np.zeros((row_num, row_num))
    line_array = np.arange(0, row_num)
    degree = 5

    # Fit each row and column of the image to a polynomial of n-th degree, store the resulting plots in the above arrays

    for i in range(0, row_num - 1):
        row_i = conv_data_Trace_Array[i, :]
        column_i = conv_data_Trace_Array[:, i]
        horz_fit = np.polyfit(line_array, row_i, degree)
        vert_fit = np.polyfit(line_array, column_i, degree)
        horz_polyval2 = -np.poly1d(horz_fit)
        vert_polyval2 = -np.poly1d(vert_fit)
        horz_array[i, :] = horz_polyval2(line_array)
        vert_array[:, i] = vert_polyval2(line_array)

    # Add the resulting arrays together
    mesh = horz_array + vert_array

    _min, _max = np.amin([np.amin(horz_array), np.amin(vert_array)]), np.amax(
        [np.amax(horz_array), np.amax(vert_array)])

    # Substract the resulting plane and normalise
    norm_data_Trace_Array = normalise(aligned_med_data_Trace_Array + mesh)

else:
    print('Provide a valid Method number!')

# -------------------------------PLOT THE RESULTING GRAPHS----------------------------------------
# Return the graphs provided by the method chosen, comments taken from a report.

plt.subplot(3, 5, 1)
plt.imshow(shaped_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
plt.title('Raw')
# the appearance of the raw ibw file’s data trace, the ibw file in every image in the dataset contains a data trace and
# phase trace.  This data trace is extracted from the ibw file by importing the ibw file as a hd5 file and saves the
# data trace from the hd5 as a pixel intensity grayscale array.  The colour map is a Red to Black scheme

plt.subplot(3, 5, 2)
plt.imshow(aligned_med_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
plt.title('Median Aligned')
# the grayscale array is often 512 x 512 pixels.  The median aligner splits the array into 512 rows, calculates the
# median of each row, then row by row systematically matches the medians by raising and lowering each pixel in the
# previous row’s pixel intensity by the difference in median by the same amount.

if method == 3:
    plt.subplot(3, 5, 3)
    plt.imshow(conv_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Blurred')
    # Method 3 involves blurring the data trace array before polynomial fitting, this is what the polyfitter is fitting
    # to.

plt.subplot(3, 5, 6)
plt.imshow(horz_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
           cmap='RdGy')
plt.title('Horz. Fit')
# The first of 4 stages of the polynomial background subtraction algorithm, this was the most general x-y plane method
# of the three methods chosen.  The horizontal fit is calculating by summarising the entire grayscale array vertically
# by making a horizontal array of mean pixel intensity in the vertical direction, then applying an 5th degree polynomial
# fit to the array.  The resulting fit is shown as a grid.

plt.subplot(3, 5, 7)
plt.imshow(vert_array, extent=(0, row_num, 0, row_num), origin='lower', vmin=_min, vmax=_max,
           cmap='RdGy')
plt.title('Vert. Fit')
# plt.colorbar()
# Same as the Horz. Fit, but a vertical fit.  Both the fits are on the same colour map scale.

plt.subplot(3, 5, 8)
plt.imshow(mesh, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')
plt.title('Comb. Fit')
# A combined fit that is the previous two fits overlaid.

if method == 2:
    plt.subplot(3, 5, 9)
    plt.imshow(gauss_mesh, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Blur. Fit')
    # Method 2 involves blurring the resulting meshgrid before subtraction, this is the appearance of the plane
    # subtracted from the image

    plt.subplot(3, 5, 10)
    plt.imshow(norm_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Plane Sub.')
else:
    plt.subplot(3, 5, 9)
    plt.imshow(norm_data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
               cmap='RdGy')
    plt.title('Plane Sub.')
    # The combined fit is applied to the grayscale array and then the resulting image’s pixel intensities are normalised
    # between 0 and 1.

# -----------------------------------NOW ATTEMPT TO BINARISE IMAGE--------------------------------------
# A statistical approach is taken to binarisation in the form of Otsu's thresholding.
# Both methods deployed rely on finding an intersection in the probability distribution created by the image histogram,
# a histogram that bins the pixel intensities in the grayscale image array.  This intersection value physically
# represents when a pixel has a 50/50 chance to be part of the nanoparticle layer or the substrate layer.  This relies
# on the image being bimodal, where each mode represents these two layers.



# Where to save the resulting files, all appended with the file name and method number
sav_loc = r'thres_img/eu_tests/res2/' + file_name.replace('.ibw', '') + '_M' + str(method)

# Attempt to Gaussian blur the image beforehand then renormalise to improve Gaussian Differential Fit
blur_str = 0

# Apply the blur and append blur strength to file path if blur is chosen
if blur_str > 0:
    gauss_data_Trace_Array = ndimage.gaussian_filter(norm_data_Trace_Array, sigma=blur_str)
    sav_loc += 'BLUR='+str(blur_str)
else:
    gauss_data_Trace_Array = norm_data_Trace_Array
gauss_data_Trace_Array = normalise(gauss_data_Trace_Array)

#**************FINDPEAKS METHOD******************

# Consider all possible threshold values and use them to plot a cumulative distribution, a threshold sweep
n = 1000  # The sensitivity of the thresholder
thres = np.linspace(0, 1, n)
pix = np.zeros((n,))
cut_off = 0
start = 0

for i, t in enumerate(thres):
    pix[i] = np.sum(gauss_data_Trace_Array < t)
    # Average over the previous 10 values and if the mean is 1% off the total number of pixels, write a cut off value.
    # This removes the portion of the graph that is after the threshold saturates.
    if i > 8 and cut_off == 0 and np.mean(pix[np.arange(i - 9, i + 1)]) > (0.99 * row_num * row_num):
        cut_off = i
    else:
        cut_off = cut_off
    # Similarly attempt to truncate the data used by recording a point where data becomes non-zero at the start of the
    # sweep.
    if i > 8 and start == 0 and np.mean(pix[np.arange(i - 9, i + 1)]) > (0.01 * row_num * row_num):
        start = i
    else:
        start = start

if cut_off == 0:  # A safety valve for if the above loop doesn't find a start or cut off value
    cut_off = n - 1
if start == n - 1:
    start = 0


plt.subplot(3, 5, 11)
threshold_plot = plt.plot(thres, pix)
plt.grid(True)
# plt.title('Threshold height sweep', fontsize=10)
plt.xlabel('Threshold', fontsize=3)
plt.ylabel('Pixels', fontsize=3)
plt.title('Threshold Sweep')
# 1000 threshold values spaced between 0 to 1 are applied to the grayscale array.  A logic statement sets pixels to 0 if
# they take a pixel intensity value below the particular threshold value in a loop, or 1 if they’re above, then a sum
# function sees how many pixels took a value of 1.  Trailing data ie when the threshold value finds every pixel to take
# a value of 1 is removed in later graphs.  This truncates the graph to add more precision to some fits, but probably
# not as effective as histogram normalisation.

# Fit a Gaussian-smoothed curve to the differential of the threshold sweep graph between the start and the cut off value
gauss_sigma = 10
pix_gauss_grad = ndimage.gaussian_gradient_magnitude(pix[0:cut_off+1], gauss_sigma)

# Identify the peaks and troughs of the signal
peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=1)
troughs, properties2 = signal.find_peaks(-pix_gauss_grad, prominence=1)

plt.subplot(3, 5, 12)
dif_threshold_plot = plt.plot(thres[0:cut_off+1], pix_gauss_grad)
dif_threshold_scatter = plt.scatter(thres[peaks], pix_gauss_grad[peaks])
dif_threshold_scatter2 = plt.scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
plt.grid(True)
# plt.title('Gaussian gradient (\u03C3=' + str(gauss_sigma) + ')', fontsize=10)
plt.xlabel('Threshold', fontsize=3)
plt.ylabel('\u0394Pixels', fontsize=3)
plt.title('Intensity Peaks')
# The Gaussian-smoothed image histogram derived from the 1st-derivative of the cumulative distribution.  The peaks and
# troughs are marked, found by the findpeaks function.

# If at least one trough is found, set that as the optimal threshold value for binarisation, otherwise move on
if len(troughs) < 1:
    print('Rejected! No clear optimisation.')
    opt_thres = 0

else:
    opt_thres = thres[troughs[len(troughs)-1]]
    print('Threshold found to be %.3f' % opt_thres + '.')

# Save a binarised image (if possible) as well as all the subplots created during processing
if opt_thres != 0:
    plt.subplot(3, 5, 13)
    plt.imshow(norm_data_Trace_Array > opt_thres, extent=(0, row_num, 0, row_num), origin='lower', cmap='gray')
    plt.title('Final')
    plt.savefig(sav_loc + '.png')
    # plt.ion() # comment out to suppress output of a bunch of images
    plt.imsave(sav_loc + 'THRES.png', norm_data_Trace_Array > opt_thres, cmap='gray')# The image is flipped vertically
else:
    plt.savefig(sav_loc + '.png')

# Save the image histogram fit as a separate image file too
plt.figure()
dif_threshold_plot = plt.plot(thres[0:cut_off+1], pix_gauss_grad)
dif_threshold_scatter = plt.scatter(thres[peaks], pix_gauss_grad[peaks])
dif_threshold_scatter2 = plt.scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
plt.grid(True)
plt.savefig(sav_loc +'HISTFIT.png')


#**************CURVE FIT METHOD********************
# Fit 2 Gaussians to the image histogram using a curvefit function and uses their intercept as the threshold value

# Find the image histogram
plt.figure()
hist, bin_edges = np.histogram(gauss_data_Trace_Array.ravel(), bins=256, range=(0.0, 1.0), density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# Define a gaussian function and a mixture model consisting of two gaussians
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss2(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

# p0 is the initial guess for the fitting coefficients initialize them differently so the optimisation algorithm works
#   better
p0 = [1., -1., 1.,1., -1., 1.]

# Optimise and in the end you will have 6 coeff (3 for each gaussian)
coeff, var_matrix = curve_fit(gauss2, bin_centres, hist, p0=p0)

# You can plot each gaussian separately using the defined single gaussian function by splitting up the coefficients
pg1 = coeff[0:3]
pg2 = coeff[3:]

g1 = gauss(bin_centres, *pg1)
g2 = gauss(bin_centres, *pg2)

plt.plot(bin_centres, hist, label='Data')
plt.plot(bin_centres, g1, label='Gaussian1')
plt.plot(bin_centres, g2, label='Gaussian2')
plt.savefig(sav_loc +'N2.png')
# Plot all 3 curves, data should really be a barchart
hist_thres = np.mean([pg1[1], pg2[1]]) # Finds the midpoint of the 2 mu values, probably should use intersect instead

# Save the binary image
plt.imsave(sav_loc + 'THRES_CF.png', gauss_data_Trace_Array > hist_thres, cmap='gray')



# FIND A HISTOGRAM OF THE IMAGE PIXEL INTENSITY DATA BEFORE THE CUT-OFF POINT FOR A CROPPED SMOOTHER GRAPH

# Use the start and cut-off value for the range of a histogram
# replace range 0.0 with start/1000 and 1.0 with cut_off/1000 and vice versa to test truncation
hist_c, bin_edges_c = np.histogram(gauss_data_Trace_Array, bins=256, range=(0.0, cut_off/1000), density=True)
bin_centres_c = (bin_edges_c[:-1] + bin_edges_c[1:])/2

p0 = [1., -1., 1.,1., -1., 1.]

coeff, var_matrix = curve_fit(gauss2, bin_centres_c, hist_c, p0=p0)

pg1 = coeff[0:3]
pg2 = coeff[3:]

g1_c = gauss(bin_centres_c, *pg1)
g2_c = gauss(bin_centres_c, *pg2)

plt.figure()
plt.plot(bin_centres_c, hist_c, label='Data')
plt.plot(bin_centres_c, g1_c, label='Gaussian1')
plt.plot(bin_centres_c, g2_c, label='Gaussian2')
plt.savefig(sav_loc + 'N2CROP.png')

hist_thres = np.mean([pg1[1], pg2[1]])

plt.imsave(sav_loc + 'THRES_CF_CROP.png', gauss_data_Trace_Array > hist_thres, cmap='gray')


# Save the raw ibw and processed images

plt.imsave(sav_loc + '_RAW.png', shaped_data_Trace_Array, origin='lower', cmap='gray')
plt.imsave(sav_loc + '_PROCESSED.png', norm_data_Trace_Array, origin='lower',
           cmap='gray')
