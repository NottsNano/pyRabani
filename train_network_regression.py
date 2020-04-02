import os
import subprocess

import h5py
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model

from utils import plot_history


class txtRegressionAlexDataGenerator(Sequence):
    def __init__(self, root_dir, batch_size, is_train,
                 horizontal_flip=True, vertical_flip=True):
        self.root_dir = root_dir
        self.batch_size = batch_size

        self.is_training_set = is_train
        self.is_validation_set = False

        self.hflip = horizontal_flip
        self.vflip = vertical_flip

        self.class_weights_dict = None
        self.__reset_file_iterator__()

        self._get_image_res()

        # self._get_class_weights()

        self._batches_counter = 0

        self.filenames = np.empty((self.__len__() * self.batch_size,), dtype=object)
        self.y_true = np.zeros((self.__len__() * self.batch_size, self.num_cats))

    def _get_class_weights(self):
        """Open all the files once to compute the class weights"""
        self.__reset_file_iterator__()
        if self.is_training_set:
            # os.scandir random iterates, so can take a subset of max 50k files to make good approximation
            length = min(int(self.__len__()) * self.batch_size, 50000)
        else:
            length = int(self.__len__()) * self.batch_size
        class_inds = np.zeros((length,))

        for i in range(length):
            file_entry = self._file_iterator.__next__().path
            h5_file = h5py.File(file_entry, "r")
            idx_find = self.original_categories_list.index(h5_file.attrs["category"])
            class_inds[i] = idx_find

        self.class_weights_dict = class_weight.compute_class_weight('balanced',
                                                                    np.arange(len(self.original_categories_list)),
                                                                    class_inds)

        self.__reset_file_iterator__()

    def _get_image_res(self):
        """Open one file to check the image resolution"""
        self.image_res = len(np.loadtxt(self._file_iterator.__next__().path, delimiter=","))
        self.num_cats = len(self._file_iterator.__next__().path.split("-")) - 2
        self.__reset_file_iterator__()

    def on_epoch_end(self):
        """At end of epoch"""
        self.__reset_file_iterator__()

    def __reset_file_iterator__(self):
        """Resets the file iterator (do this instead of os.listdir for speed on massive datasets!)"""
        self._file_iterator = os.scandir(self.root_dir)
        self._batches_counter = 0

    def __len__(self):
        n_files = int(
            subprocess.Popen([f"find '{self.root_dir}' -maxdepth 1 | wc -l"],
                             stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True).communicate()[0]) - 1
        return int(np.floor(n_files // self.batch_size))

    def parse_name(self, fname):
        chunks = fname.split("-")

        out = np.zeros((self.num_cats,))
        for i, chunk in enumerate(chunks[:self.num_cats]):
            out[i] = float(chunk.split("=")[1])

        return out

    def __getitem__(self, idx):
        """Get self.batch_size number of items, shaped and augmented"""

        # Preallocate output
        batch_x = np.empty((self.batch_size, self.image_res, self.image_res, 1))
        batch_y = np.zeros((self.batch_size, self.num_cats))
        filenames = np.empty((self.batch_size,), dtype=object)

        # For each file in the batch
        for i in range(self.batch_size):
            # Parse parameters from the h5 file
            file = self._file_iterator.__next__()
            file_entry = file.path
            file_name = file.name

            batch_x[i, :, :, 0] = np.loadtxt(file_entry, delimiter=",") // 9
            batch_y[i, :] = self.parse_name(file_name)

            if self.is_validation_set:
                filenames[i] = file.name

        if self.is_validation_set:
            self.y_true[self._batches_counter * self.batch_size:(self._batches_counter + 1) * self.batch_size,
            :] = batch_y
            self.filenames[
            self._batches_counter * self.batch_size:(self._batches_counter + 1) * self.batch_size] = filenames

        if self.is_training_set:
            batch_x = self.augment(batch_x)

        self._batches_counter += 1
        if self._batches_counter >= self.__len__() and self.is_training_set is False:
            self.__reset_file_iterator__()

        return batch_x, batch_y

    def augment(self, batch_x):
        if self.vflip:
            augment_inds_vflip = np.random.choice(self.batch_size, size=(self.batch_size,), replace=False)
            batch_x[augment_inds_vflip, :, :, 0] = np.flip(batch_x[augment_inds_vflip, :, :, 0], axis=1)
        if self.hflip:
            augment_inds_hflip = np.random.choice(self.batch_size, size=(self.batch_size,), replace=False)
            batch_x[augment_inds_hflip, :, :, 0] = np.flip(batch_x[augment_inds_hflip, :, :, 0], axis=2)

        return batch_x


def train_regression_model(train_datadir, test_datadir, batch_size, epochs):
    # Set up generators
    train_generator = txtRegressionAlexDataGenerator(train_datadir, batch_size=batch_size, is_train=True)
    test_generator = txtRegressionAlexDataGenerator(test_datadir, batch_size=batch_size, is_train=False)

    # Set up model
    input_shape = (train_generator.image_res, train_generator.image_res, 1)
    model = Sequential()
    model.add(Convolution2D(16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(train_generator.num_cats))

    csv_logger = CSVLogger('Data/Logs/regression_log.csv', append=True, separator=';')
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    model_checkpoint = ModelCheckpoint("Data/Models/regression_model.h5", monitor="val_loss", save_best_only=True)
    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Train
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        steps_per_epoch=train_generator.__len__(),
                        validation_steps=test_generator.__len__(),
                        epochs=epochs,
                        callbacks=[csv_logger, early_stopping, model_checkpoint],
                        max_queue_size=100)

    return model


def validation_pred(model, validation_datadir, batch_size):
    validation_generator = txtRegressionAlexDataGenerator(validation_datadir, batch_size=batch_size,
                                                          is_train=False)
    validation_generator.is_validation_set = True

    validation_preds = model.predict_generator(generator=validation_generator, steps=validation_generator.__len__())
    validation_truth = validation_generator.y_true
    filenames = validation_generator.filenames

    return validation_preds, validation_truth, filenames


if __name__ == '__main__':
    # Train
    training_data_dir = "/media/mltest1/Dat Storage/Alex Data/Training"
    testing_data_dir = "/media/mltest1/Dat Storage/Alex Data/Testing"

    trained_model = train_regression_model(train_datadir=training_data_dir, test_datadir=testing_data_dir,
                                           batch_size=2048,
                                           epochs=1000)
    plot_history(trained_model)

    # trained_model = load_model("Data/Models/regression_model.h5")
    plot_model(trained_model, to_file="Data/Models/regression_model.png", show_shapes=True, show_layer_names=True)
    preds, truth, files = validation_pred(trained_model, validation_datadir=testing_data_dir, batch_size=2048)

    pred_dframe = pd.DataFrame({"Filename": files,
                                "True C": truth[:, 0], "Predicted C": preds[:, 0],
                                "True eps": truth[:, 1], "Predicted eps": preds[:, 1],
                                "True k": truth[:, 2], "Predicted k": preds[:, 2]})
    pred_dframe.to_csv("Data/Logs/regression_preds.csv")
