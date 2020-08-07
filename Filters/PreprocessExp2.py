# Code runs a MoD aligner and then a spline and polynomial detrend of degree of freedom = 2 and saves it to a specified
# folder.

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pycroscopy as scope
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri
import glob
splines = importr('splines')   # Imports the splines package in R



# Change the file path here
file_path = r'R/img/steff_im/**.ibw'
# Change where the files are saved here
sav_dir = r'resR/steff_imR/'

# Define a degree of freedom to be used in detrenders, Df acts as the DoF in spline while DoF in the polynomial
# detrend DoF acts as Df.  Default is 2.
Df = 2
# Assign DoF to a variable in R space
DoF = ro.r.matrix(Df)
ro.r.assign("df", DoF)

# Define a boatload of functions and R strings


def trace_loader(filepath):
    # Create an object capable of translating .ibw files
    TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)
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


# Define a function that sets an image numpy array's values (pixel intensities) to be between 0 and 1
def normalise(array):
    norm_array = (array - np.min(array)) \
                 / (np.max(array) - np.min(array))
    return norm_array

# ## Median Alignment approach # ###
# # Align every median in the normalised array to be equal by finding the difference between the 1st and i-th row's
# # medians, and then offset all the data in that row by the same amount in order to equate them
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

# Description of Quadratic Spline Detrender goes here

splinestring = '''
  row_num<-dim(alignim)[1]
  im1data<-data.frame(as.vector(alignim)) # Turning image1 into a dataframe of columns x, y and intensity
  names(im1data)<-"intensity"
  im1data$x<-rep(1:row_num,row_num)
  im1data$y<-as.vector(t(matrix(rep(1:row_num,row_num),row_num,row_num)))

  im4mod3<-lm(intensity~ns(x,df)*ns(y,df),data=im1data) # 1st step of spline detrend
  im1data$lmresid<-(im4mod3$residuals -min(im4mod3$residuals))/(max(im4mod3$residuals)-min(im4mod3$residuals))  # 2nd step of spline detrend

  spline_dt<-(matrix(im1data$lmresid,row_num,row_num))
'''

# Description of Polynomial goes here

polystring = '''
  row_num<-dim(alignim)[1]
  im1data<-data.frame(as.vector(alignim)) # Turning image1 into a dataframe of columns x, y and intensity
  names(im1data)<-"intensity"
  im1data$x<-rep(1:row_num,row_num)
  im1data$y<-as.vector(t(matrix(rep(1:row_num,row_num),row_num,row_num)))

  im4mod3<-lm(intensity~poly(x,df)*poly(y,df),data=im1data) # 1st step of poly detrend
  im1data$lmresid<-(im4mod3$residuals -min(im4mod3$residuals))/(max(im4mod3$residuals)-min(im4mod3$residuals))  # 2nd step of poly detrend

  poly_dt<-(matrix(im1data$lmresid,row_num,row_num))
'''

def img_mask(img, deg):
    # ## APPLY IMAGE MASK ## #
    # Apply an image mask that truncates data outside pixel intensity mean +- deg
    img_mean = np.mean(img)
    img_std = np.std(img)
    img_u_mask = img_mean + deg*img_std
    # print(img_u_mask)
    img_l_mask = img_mean - deg*img_std
    # print(img_l_mask)
    u_maskd = np.where(img < img_u_mask, img, img_u_mask)
    l_maskd = np.where(u_maskd > img_l_mask, u_maskd, img_l_mask)
    return l_maskd


# NOW THE LOOP


matches = glob.glob(file_path, recursive=True)
for file_path in matches:
    name = file_path.replace('R/img/steff_im\\', '')
    sav_loc = sav_dir + name.replace('.ibw', '_')
    pres_sav_loc = sav_dir + 'pres/' + name.replace('.ibw', '_')  # Create a 2nd folder for storing presentable images

    # Load the ibw file from the chosen file path and save the data trace as a numpy array
    data_Trace_Array = trace_loader(file_path)

    # Identify the size of the data trace array and reshape the array accordingly
    if data_Trace_Array.shape[0] == 65536:
        row_num = 256
    elif data_Trace_Array.shape[0] == 262144:
        row_num = 512
    elif data_Trace_Array.shape[0] == 1048576:
        row_num = 1024

    shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))

    ro.numpy2ri.activate()  # Converts all R-objects entering the python variable space into numpy-able formats

    norm_data_Trace_Array = normalise(shaped_data_Trace_Array)
    aligned_data_Trace_Array = mod_align(shaped_data_Trace_Array, row_num)
    aligned_data_Trace_Array = normalise(aligned_data_Trace_Array)

    data_matrix = ro.r.matrix(aligned_data_Trace_Array, nrow=row_num, ncol=row_num)
    ro.r.assign("r_data_matrix", data_matrix)
    ro.r("alignim <- r_data_matrix")

    spline_dt_data_Trace_Array = ro.r(splinestring)
    spline_dt_data_Trace_Array = normalise(spline_dt_data_Trace_Array)
    poly_dt_data_Trace_Array = ro.r(polystring)
    poly_dt_data_Trace_Array = normalise(poly_dt_data_Trace_Array)
    plt.imsave(sav_loc + str(Df) + '_df_spline.png', spline_dt_data_Trace_Array, origin='lower', cmap='gray')
    plt.imsave(sav_loc + str(Df) + 'df_poly.png', poly_dt_data_Trace_Array, origin='lower', cmap='gray')

    trunc_aligned_data_Trace_Array = img_mask(aligned_data_Trace_Array, 3)
    trunc_spline_dt_data_Trace_Array = img_mask(spline_dt_data_Trace_Array, 3)
    trunc_poly_dt_data_Trace_Array = img_mask(poly_dt_data_Trace_Array, 3)

    plt.imsave(pres_sav_loc + str(Df) + '_df_mod.png', trunc_aligned_data_Trace_Array, origin='lower', cmap='RdGy')
    plt.imsave(pres_sav_loc + str(Df) + '_df_spline.png', trunc_spline_dt_data_Trace_Array, origin='lower', cmap='RdGy')
    plt.imsave(pres_sav_loc + str(Df) + 'df_poly.png', trunc_poly_dt_data_Trace_Array, origin='lower', cmap='RdGy')
