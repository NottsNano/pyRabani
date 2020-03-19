# -*- coding: utf-8 -*-

import pycroscopy as scope
import matplotlib.pyplot as plt
import h5py
import pyUSID
from scipy import ndimage
import numpy as np

# Create an object capable of translating .ibw files
TranslateObj = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)

# Translate the requisite file
Output = TranslateObj.translate(
    file_path=r'ibw_test.ibw', verbose=False)

print(Output)

# Opening this file to read in sections as a numpy array
Read_Path = Output
h5_File = h5py.File(Output, mode='r')

# Various commands for accessing information of the file
pyUSID.hdf_utils.print_tree(h5_File)
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

data_Trace_Array = np.reshape(data_Trace_Array, (row_num, row_num))
phase_Trace_Array = np.reshape(phase_Trace_Array, (row_num, row_num))

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


data_Trace_Array[1, :] = data_Trace_Array[1, :] - np.mean(data_Trace_Array[1, :])
for i in range(1, row_num):
    row_iless1 = data_Trace_Array[i - 1, :]
    row_i = data_Trace_Array[i, :]
    Offset = line_align(row_iless1, row_i)
    data_Trace_Array[i, :] = data_Trace_Array[i, :] + Offset

    row_iless1 = phase_Trace_Array[i - 1, :]
    row_i = phase_Trace_Array[i, :]
    Offset = line_align(row_iless1, row_i)
    phase_Trace_Array[i, :] = phase_Trace_Array[i, :] + Offset

# ----------------Next step is to flatten the surface-------------------------

horizon_left = data_Trace_Array[:, 0]
horizon_right = data_Trace_Array[:, row_num - 1]
vertical_top = data_Trace_Array[row_num - 1, :]
vertical_bottom = data_Trace_Array[0, :]

# Finding the approximate direction of the gradient of the plane
hor_gradient = np.mean(horizon_left - horizon_right) / row_num
ver_gradient = np.mean(vertical_top - vertical_bottom) / row_num

# The options for gradients of the plane
hor_grad_array = np.linspace(0 * hor_gradient, 1.5 * hor_gradient, 10)
ver_grad_array = np.linspace(-1.5 * ver_gradient, 1.5 * ver_gradient, 10)

square_differences = np.zeros([10, 10])
centroid = ndimage.measurements.center_of_mass(data_Trace_Array)
centroid_mass = data_Trace_Array[int(np.round(centroid[0])), int(np.round(centroid[1]))]
test_line_x = np.ones([row_num, row_num]) * range(-int(np.round(centroid[0])), row_num - int(np.round(centroid[0])))
test_line_y = np.ones([row_num, row_num]) * range(-int(np.round(centroid[1])), row_num - int(np.round(centroid[1])))

for i in range(0, 10):
    for j in range(0, 10):
        hor_gradient = hor_grad_array[i]
        ver_gradient = ver_grad_array[j]

        hor_array = test_line_x * - hor_gradient
        ver_array = np.transpose(test_line_y * - ver_gradient)

        test_plane = hor_array + ver_array + centroid_mass

        square_differences[i, j] = np.sum(np.square(data_Trace_Array - test_plane))

        print(i, j, square_differences[i, j])

best_indices = np.unravel_index(np.argmin(square_differences, axis=None), square_differences.shape)

hor_gradient = hor_grad_array[best_indices[0]]
ver_gradient = ver_grad_array[best_indices[1]]

# Test gradients are 1.2e-10 and 0.3e-10
print(hor_gradient)
print(ver_gradient)

hor_array = test_line_x * - hor_gradient
ver_array = np.transpose(test_line_y * - ver_gradient)
centroid = ndimage.measurements.center_of_mass(data_Trace_Array)
test_plane = hor_array + ver_array + centroid_mass
print(centroid_mass)

# The plane is removed
plt.subplot(1, 2, 1)
plt.imshow(data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

data_Trace_Array = data_Trace_Array - test_plane

plt.subplot(1, 2, 2)
plt.imshow(data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
           cmap='RdGy')

#plt.ion()

# # --------------Next step is to identify the edges present on the surface-----
#
# surf_gradients = np.gradient(data_Trace_Array)
# surf_edges = np.zeros([row_num, row_num])
# surf_edges = [np.absolute(surf_gradients) > 0.5e-10]
#
#
# def boolstr_to_floatstr(v):
#     if v == 'True':
#         return '1'
#     elif v == 'False':
#         return '0'
#     else:
#         return v
#
#
# surf_edges = np.vectorize(boolstr_to_floatstr)(surf_edges).astype(float)
#
# surf_edges = surf_edges[0, 1, :, :] + surf_edges[0, 0, :, :]
#
# # --------------Edges should be cleaned, by addition or removal----------------
#
# checks_edges = surf_edges
#
# for i in range(1, row_num - 1):
#     for j in range(1, row_num - 1):
#         if surf_edges[i, j] > 0:
#             if (surf_edges[i - 1, j] < 1 and surf_edges[i + 1, j] < 1 and surf_edges[i, j - 1] < 1 and surf_edges[
#                 i, j + 1] < 1):
#                 checks_edges[i, j] = 0
#         elif surf_edges[i, j] < 1:
#             if (surf_edges[i - 1, j] > 0 and surf_edges[i + 1, j] > 0) or (
#                     surf_edges[i, j - 1] > 0 and surf_edges[i, j + 1] > 0):
#                 checks_edges[i, j] = 1
#
# surf_edges = checks_edges
# del checks_edges
#
#
# #
# ##---------------Next is to identify terraces on the surface-------------------
# #
# def terrace_detect(check_point, checked_region, surface_to_check):
#     # Check_point should consist of two indices, referring to positions in the
#     # checked_region and surface_to_check surfaces
#     not_in_region = 1
#     i = check_point[0]
#     j = check_point[1]
#
#     if checked_region[i, j] > 0:
#         not_in_region = 1
#         points_to_check = []
#         return not_in_region, points_to_check
#
#     if surface_to_check[i, j] > 0:
#         not_in_region = 1
#         points_to_check = []
#         return not_in_region, points_to_check
#     else:
#         not_in_region = 0
#         points_to_check = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
#         return not_in_region, points_to_check
#
#
# def remove(duplicate, row_num):
#     final_list = []
#     for num in duplicate:
#         a, b = num
#         if (a > -1 and b > -1 and a < row_num and b < row_num):
#             if num not in final_list:
#                 final_list.append(num)
#     return final_list
#
#
# def terrace_create(surface_edges, origin, terrace_number):
#     row_num = surface_edges.shape[0]
#     terrace = np.zeros([row_num, row_num])
#     to_check = origin
#
#     while len(to_check) > 0:
#
#         i, j = to_check[0]
#
#         add_to_terrace, add_to_check = terrace_detect(to_check[0], terrace, surface_edges)
#
#         if len(add_to_check) > 0:
#             a, b, c, d = add_to_check
#             to_check.append(a)
#             to_check.append(b)
#             to_check.append(c)
#             to_check.append(d)
#
#         to_check = remove(to_check, row_num)
#
#         if add_to_terrace < 1:
#             terrace[i, j] = terrace_number + 1
#
#         to_check = to_check[1:]
#
#     return terrace
#
#
# ##-------Next the program generates a unique identifier for each terrace-------
# #
# terrace_number = 0
#
# while np.any(surf_edges < 1):
#     origin_i, origin_j = np.argwhere(surf_edges < 1)[0]
#     terrace_number = terrace_number + 1
#     origin = [(origin_i, origin_j)]
#     print(origin)
#     surf_edges = surf_edges + terrace_create(surf_edges, origin, terrace_number)
#
# # Normalise the phase trace
# phase_max = np.amax(phase_Trace_Array)
# phase_Trace_Array = phase_Trace_Array / phase_max
#
# # -------------------User inputs to remove graphite terraces------------------
# # User should input pairs of indices, and the program will then remove from
# # consideration the terrace containing that point.
#
#
# plt.ion()
#
# figures, (axis_1, axis_2) = plt.subplots(1, 2)
#
#
# def draw_figure():
#     plt.subplot(1, 2, 2)
#     plt.imshow(data_Trace_Array, extent=(0, row_num, 0, row_num), origin='lower',
#                cmap='RdGy')
#     plt.subplot(1, 2, 1)
#     plt.imshow(surf_edges, extent=(0, row_num, 0, row_num), origin='lower',
#                cmap='RdGy')
#
#     figures.set_figheight(10)
#     figures.set_figwidth(10)
#     # show()
#
#
# for i in range(1, 100):
#
#     drawnow(draw_figure)
#
#     x = input("Enter x-index to remove: ")
#     try:
#         int(x)
#     except:
#         break
#     y = input("Enter y-index to remove: ")
#     terrace_number = surf_edges[int(x), int(y)]
#     print(terrace_number)
#     for j in range(0, row_num):
#         for k in range(0, row_num):
#             if terrace_number - 1 < surf_edges[j, k] < terrace_number + 1:
#                 surf_edges[j, k] = 0
# #
# ##-----------------Summing the total remaining area---------------------------
# required_area = np.count_nonzero(surf_edges)
# coverage = required_area / (row_num ** 2)
# print('Surface coverage is {}%'.format(coverage * 100))
#
# #
#
#
#
#
#
#
#
#

#Write it such that data_Trace_Array isn't the primary array the whole way through

#THRESHOLDER


#normalise image so (all values-min value)/(max value-min value) so all pixels are 0 to 1

#norm_img = (img-min(img))/(max(img)-min(img))

#n = 1000 #precision of threshold
#thres = np.linspace(0,1,n)

#n.pix = np.sum(norm_img > thres)