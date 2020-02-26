import os
import subprocess

import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn import metrics
import itertools

class h5RabaniDataGenerator(Sequence):
    def __init__(self, root_dir, batch_size, output_parameters_list, output_categories_list, is_train,
                 horizontal_flip=True, vertical_flip=True, y_noise=None):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.original_parameters_list = output_parameters_list
        self.original_categories_list = output_categories_list

        self.is_training_set = is_train
        self.hflip = horizontal_flip
        self.vflip = vertical_flip
        self.ynoise = y_noise

        self.class_weights_dict = None
        self.__reset_file_iterator__()
        self._get_image_res()
        self._get_class_weights()

        self._batches_counter = 0

    def _get_class_weights(self):
        """Open all the files once to compute the class weights"""
        self.__reset_file_iterator__()
        length = int(self.__len__() * self.batch_size)

        self.class_inds = np.zeros((length,))

        for i in range(length):
            file_entry = self._file_iterator.__next__().path
            h5_file = h5py.File(file_entry, "r")
            idx_find = self.original_categories_list.index(h5_file.attrs["category"])
            self.class_inds[i] = idx_find

        self.class_weights_dict = class_weight.compute_class_weight('balanced',
                                                                    np.arange(len(self.original_categories_list)),
                                                                    self.class_inds)

        self.__reset_file_iterator__()

    def _get_image_res(self):
        """Open one file to check the image resolution"""
        self.image_res = len(h5py.File(self._file_iterator.__next__().path, "r")["sim_results"]["image"])
        self.__reset_file_iterator__()

    def on_epoch_end(self):
        self.__reset_file_iterator__()

    def __reset_file_iterator__(self):
        self._file_iterator = os.scandir(self.root_dir)
        self._batches_counter = 0

    def __len__(self):
        n_files = int(
            subprocess.Popen([f"find '{self.root_dir}' -maxdepth 1 | wc -l"],
                             stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True).communicate()[0]) - 1
        return int(np.floor(n_files // self.batch_size))

    def __getitem__(self, idx):
        """Get batch_size number of items, shape and augment"""

        # Preallocate output
        batch_x = np.empty((self.batch_size, self.image_res, self.image_res, 1))
        # batch_y = [np.empty((self.batch_size, len(self.original_parameters_list))),
        #            np.empty((self.batch_size, len(self.original_categories_list)))]
        batch_y = np.zeros((self.batch_size, len(self.original_categories_list)))

        # For each file in the batch
        for i in range(self.batch_size):
            # Parse parameters from the h5 file
            file_entry = self._file_iterator.__next__().path
            h5_file = h5py.File(file_entry, "r")

            batch_x[i, :, :, 0] = h5_file["sim_results"]["image"]

            # for j, (param, cat) in enumerate(zip(self.original_parameters_list, self.original_categories_list)):
            # batch_y[0][i, j] = h5_file.attrs[param]
            #
            # TESTRAND = np.round(np.random.normal(1))
            # batch_y[1][i, :] = [TESTRAND, np.abs(1 - TESTRAND)]
            idx_find = self.original_categories_list.index(h5_file.attrs["category"])
            batch_y[i, idx_find] = 1

        # Augment if we are training
        if self.is_training_set:
            if self.vflip:
                augment_inds_vflip = np.random.choice(self.batch_size, size=(self.batch_size,), replace=False)
                batch_x[augment_inds_vflip, :, :] = np.flip(batch_x[augment_inds_vflip, :, :], axis=1)
            if self.hflip:
                augment_inds_hflip = np.random.choice(self.batch_size, size=(self.batch_size,), replace=False)
                batch_x[augment_inds_hflip, :, :] = np.flip(batch_x[augment_inds_hflip, :, :], axis=2)
            if self.ynoise:
                batch_y[0] *= np.random.normal(loc=1, scale=self.ynoise)

        self._batches_counter += 1
        if self._batches_counter >= self.__len__() and self.is_training_set is False:
            self.__reset_file_iterator__()

        return batch_x, batch_y


def train_model(train_datadir, test_datadir, y_params, y_cats, batch_size, epochs):
    # Set up generators
    train_generator = h5RabaniDataGenerator(train_datadir, batch_size=batch_size, is_train=True,
                                            output_parameters_list=y_params, output_categories_list=y_cats)
    test_generator = h5RabaniDataGenerator(test_datadir, batch_size=batch_size, is_train=False,
                                           output_parameters_list=y_params, output_categories_list=y_cats)

    # Set up model
    input_shape = (train_generator.image_res, train_generator.image_res, 1)
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(len(y_params), activation='linear'))
    # model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    # input_layer = Input(shape=input_shape, name="Image_Input")
    # conv1 = Conv2D(14, kernel_size=4, activation='relu')(input_layer)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(7, kernel_size=4, activation='relu')(pool1)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # flatten = Flatten()(pool2)
    #
    # output1 = Dense(len(y_params), activation='linear', name="Parameter_Classification")(flatten)
    # output2 = Dense(len(y_cats), activation='sigmoid', name="Category_Classification")(flatten)
    #
    # model = Model(inputs=input_layer, outputs=[output1, output2], name="RabaniNet")
    # model.compile(loss={"Parameter_Classification": 'mse', "Category_Classification": 'categorical_crossentropy'},
    #               metrics={"Parameter_Classification": ['mse', 'mae'], "Category_Classification": 'accuracy'},
    #               optimizer=Adam())

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
    model.add(Dense(len(y_cats), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Train
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        steps_per_epoch=train_generator.__len__()//10,
                        validation_steps=test_generator.__len__(),
                        epochs=epochs,
                        max_queue_size=100)

    return model


def validation_pred(model, validation_datadir, y_params, y_cats, batch_size):
    validation_generator = h5RabaniDataGenerator(validation_datadir, batch_size=batch_size,
                                                 is_train=False, output_parameters_list=y_params,
                                                 output_categories_list=y_cats)

    validation_preds = model.predict_generator(validation_generator, steps=validation_generator.__len__())

    encoder = OneHotEncoder(sparse=False)
    validation_truth = encoder.fit_transform(validation_generator.class_inds.reshape(-1, 1))

    return validation_preds, validation_truth


def plot_history(model):
    for plot_metric in model.metrics_names:
        plt.figure()
        plt.plot(model.history.history[plot_metric])
        plt.plot(model.history.history[f'val_{plot_metric}'])
        plt.legend([plot_metric, f'val_{plot_metric}'])
        plt.xlabel("Epoch")
        plt.ylabel(plot_metric)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

if __name__ == '__main__':
    # Train
    training_data_dir = "/media/mltest1/Dat Storage/pyRabani_Images"
    testing_data_dir = "/home/mltest1/tmp/pycharm_project_883/Images/2020-02-25/13-27"
    validation_data_dir = "/home/mltest1/tmp/pycharm_project_883/Images/2020-02-25/11-26"
    original_categories = ["hole", "liquid", "cellular", "labyrinth", "island"]
    original_parameters = ["kT", "mu"]

    trained_model = train_model(train_datadir=training_data_dir, test_datadir=testing_data_dir,
                                y_params=original_parameters, y_cats=original_categories, batch_size=512, epochs=10)
    plot_history(trained_model)

    preds, truth = validation_pred(trained_model, validation_datadir=validation_data_dir,
                    y_params=original_parameters, y_cats=original_categories, batch_size=512)

    y_pred = np.argmax(preds, axis=1)
    y_truth = np.argmax(truth, axis=1)
    conf_mat = metrics.confusion_matrix(y_truth, y_pred, labels=original_categories)
    plot_confusion_matrix(conf_mat, original_categories)
    print(metrics.classification_report(y_truth, y_pred, target_names=original_categories))