import pycroscopy as scope
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import pandas as pd
import glob
from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri


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

    data_Trace_Array = np.array(data_Trace[:])

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

    # Define a function that sets an image numpy array's values (pixel intensities) to be between 0 and 1
    def normalise(array):
        norm_array = (array - np.min(array)) \
                     / (np.max(array) - np.min(array))
        return norm_array

    if row_num > 0:
        shaped_data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))
        aligned_med_data_Trace_Array  = normalise(shaped_data_Trace_Array)


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

        # row_fit_data_Trace_Array = shaped_data_Trace_Array
        # row_fit_data_Trace_Array[1, :] = shaped_data_Trace_Array[1, :] - np.mean(shaped_data_Trace_Array[1, :])



        for i in range(1, row_num):
            row_iless1 = aligned_med_data_Trace_Array[i - 1, :]
            row_i = aligned_med_data_Trace_Array[i, :]
            Offset = line_align(row_iless1, row_i)
            aligned_med_data_Trace_Array[i, :] = aligned_med_data_Trace_Array[i, :] + Offset

        aligned_med_data_Trace_Array = normalise(aligned_med_data_Trace_Array)

        # -------------Next step is polynomial detrending -------------------------
        # In R Code apply a polynomial detrending algorithm to the image array

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

        data_matrix = ro.r.matrix(aligned_med_data_Trace_Array, nrow=row_num, ncol=row_num)
        ro.r.assign("r_data_matrix", data_matrix)
        ro.r("alignim <- r_data_matrix")

        poly_dt_data_Trace_Array = ro.r(polystring)
        norm_data_Trace_Array = normalise(poly_dt_data_Trace_Array)



        # Where to save the resulting files, all appended with the file name and method number
        sav_loc = r'res2/' + ibw_name + '.png'

        # Save the raw ibw and processed images


        plt.imsave(sav_loc, norm_data_Trace_Array, origin='lower',
                   cmap='gray')

    return norm_data_Trace_Array


# # Below few lines used for targeting unique folders, commented out when looking at whole USBs
import tkinter
from tkinter import filedialog
# root = tkinter.Tk()
# root.withdraw()
# dirname = filedialog.askdirectory(parent=root, initialdir="/", title='Please select a directory')

dirname = 'D:/USB 3'
usb_num = 3  # Change this to reflect the USB number for the ibw location

Data = pd.read_csv('ManualImageClassificationsV2_4.csv', sep=',')

FileNames = Data['Predicted ibw file name']

ro.numpy2ri.activate()  # Converts all R-objects entering the python variable space into numpy-able formats
# Define a degree of freedom to be used in polynomial detrender
Df = 2
# Assign DoF to a variable in R space
DoF = ro.r.matrix(Df)
ro.r.assign("df", DoF)

for k in range(0, FileNames.size):
    if k not in [363, 364, 374, 391, 395, 531, 561, 765]:
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
Data.to_csv('ManualImageClassificationsV2_4.csv', index=False)
