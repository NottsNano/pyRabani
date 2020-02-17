import os
import subprocess

import h5py
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from matplotlib import pyplot as plt


class h5RabaniDataGenerator(Sequence):
    def __init__(self, root_dir, batch_size, output_parameters_list, is_train,
                 horizontal_flip=True, vertical_flip=True, y_noise=None):
        self.root_dir = root_dir
        self.__reset_file_iterator__()
        self._get_image_res()

        self.batch_size = batch_size
        self.original_parameters_list = output_parameters_list

        self.is_training_set = is_train
        self.hflip = horizontal_flip
        self.vflip = vertical_flip
        self.ynoise = y_noise

        self._batches_counter = 0

    def _get_image_res(self):
        """Open one file to check the image resolution"""
        self.image_res = len(h5py.File(self._file_iterator.__next__().path, "r")["image"])
        self.__reset_file_iterator__()

    def on_epoch_end(self):
        self.__reset_file_iterator__()

    def __reset_file_iterator__(self):
        self._file_iterator = os.scandir(self.root_dir)
        self._batches_counter = 0

    def __len__(self):
        n_files = int(
            subprocess.Popen([f"find {self.root_dir} -maxdepth 1 | wc -l"],
                             stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True).communicate()[0]) - 1
        return int(np.floor(n_files // self.batch_size))

    def __getitem__(self, idx):
        """Get batch_size number of items, shape and augment"""

        # Preallocate output
        batch_x = np.empty((self.batch_size, self.image_res, self.image_res, 1))
        batch_y = np.empty((self.batch_size, len(self.original_parameters_list)))

        # For each file in the batch
        for i in range(self.batch_size):
            # Parse parameters from the h5 file
            file_entry = self._file_iterator.__next__().path

            h5_file = h5py.File(file_entry, "r")

            batch_x[i, :, :, 0] = h5_file["image"]

            for j, param in enumerate(self.original_parameters_list):
                batch_y[i, j] = h5_file.attrs[param]

        # Augment if we are training
        if self.is_training_set:
            if self.vflip:
                augment_inds_vflip = np.random.choice(self.batch_size, size=(self.batch_size,), replace=False)
                batch_x[augment_inds_vflip, :, :] = np.flip(batch_x[augment_inds_vflip, :, :], axis=1)
            if self.hflip:
                augment_inds_hflip = np.random.choice(self.batch_size, size=(self.batch_size,), replace=False)
                batch_x[augment_inds_hflip, :, :] = np.flip(batch_x[augment_inds_hflip, :, :], axis=2)
            if self.ynoise:
                batch_y *= np.random.normal(loc=1, scale=self.ynoise)

        self._batches_counter += 1
        if self._batches_counter >= self.__len__() and self.is_training_set is False:
            self.__reset_file_iterator__()

        return batch_x, batch_y


def train_model(train_datadir, test_datadir, y_params, batch_size, epochs):
    # Set up generators
    train_generator = h5RabaniDataGenerator(train_datadir, batch_size=batch_size,
                                            output_parameters_list=original_parameters, is_train=True)
    test_generator = h5RabaniDataGenerator(test_datadir, batch_size=batch_size,
                                           output_parameters_list=original_parameters, is_train=False)

    # Set up model
    input_shape = (train_generator.image_res, train_generator.image_res, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(y_params), activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    # Train
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        steps_per_epoch=train_generator.__len__(),
                        validation_steps=test_generator.__len__(),
                        epochs=epochs,
                        max_queue_size=100)

    return model


if __name__ == '__main__':
    # Train
    training_data_dir = "/home/mltest1/tmp/pycharm_project_883/Images/2020-02-17/14-04"
    testing_data_dir = "/home/mltest1/tmp/pycharm_project_883/Images/2020-02-17/09-44"
    # validation_data_dir = "/home/mltest1/tmp/pycharm_project_883/Images/2020-02-17/09-44"

    original_parameters = ["kT", "mu"]
    trained_model = train_model(train_datadir=training_data_dir, test_datadir=testing_data_dir,
                                y_params=original_parameters, batch_size=256, epochs=10)

    # Visualise
    for plot_metric in ['loss', 'mean_squared_error', 'mean_absolute_error']:
        plt.figure()
        plt.plot(trained_model.history.history[plot_metric])
        plt.plot(trained_model.history.history[f'val_{plot_metric}'])
        plt.legend([plot_metric, f'val_{plot_metric}'])
        plt.xlabel("Epoch")
        plt.ylabel(plot_metric)
