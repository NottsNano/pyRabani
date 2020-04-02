import os
import shutil

import numpy as np


# def move_files_in_folder(root_dir, folder_nums, dataset_name):
#     for folder_num in folder_nums:
#         for file in os.listdir(f"{root_dir}/{folder_num}/Data"):
#             shutil.copyfile(f"{root_dir}/{folder_num}/Data/{file}", f"{root_dir}/{dataset_name}/{file}")
#
#
# ROOT_DIR = "/media/mltest1/Dat Storage/Alex Data"
# training_folder_nums = np.arange(1, 10)
# move_files_in_folder(ROOT_DIR, training_folder_nums, "Training")
# testing_folder_nums = [10]
# move_files_in_folder(ROOT_DIR, testing_folder_nums, "Testing")

root_dir = "/media/mltest1/Dat Storage/Alex Data"
x_data_dir = "/media/mltest1/Dat Storage/Alex Data/Training"
x1_data_dir = "/media/mltest1/Dat Storage/Alex Data/Testing"
y_data_dir = "/media/mltest1/Dat Storage/Alex Data/Categories"
missing_files = []
all_x_files = os.listdir(x_data_dir)
all_x1_files = os.listdir(x1_data_dir)
all_x_files += all_x1_files

all_y_files = os.listdir(y_data_dir)

for y_file in all_y_files:
    y_file_str = y_file[11:]
    if y_file_str not in all_x_files:
        missing_files.append(y_file_str)
        # shutil.copyfile(f"{y_data_dir}/{y_file.name}", f"{root_dir}/Training_Cat/{y_file_str}")