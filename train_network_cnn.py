import itertools
import os
import subprocess

import h5py
import numpy as np
from matplotlib import pyplot as plt, colors
from sklearn.utils import class_weight
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.layers import Convolution2D


class h5RabaniDataGenerator(Sequence):
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

        self.filenames = np.zeros((self.__len__() * self.batch_size,))
        self.y_true = np.zeros((self.__len__() * self.batch_size, len(self.original_categories_list)))

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
        filenames = np.zeros((self.batch_size,))

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
            self.filenames = filenames[self._batches_counter * self.batch_size:(
                                                                                       self._batches_counter + 1) * self.batch_size] = filenames

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


def train_model(train_datadir, test_datadir, batch_size, epochs):
    # Set up generators
    train_generator = h5RabaniDataGenerator(train_datadir, batch_size=batch_size, is_train=True)
    test_generator = h5RabaniDataGenerator(test_datadir, batch_size=batch_size, is_train=False)

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

    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Train
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        steps_per_epoch=train_generator.__len__(),
                        validation_steps=test_generator.__len__() // 10,
                        epochs=epochs,
                        max_queue_size=100)

    return model


def show_random_selection_of_images(datadir, num_imgs, y_params, y_cats, imsize=128):
    img_generator = h5RabaniDataGenerator(datadir, batch_size=num_imgs, is_train=True, imsize=imsize)

    x, y = img_generator.__getitem__(None)
    axis_res = int(np.sqrt(num_imgs))

    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.figure()
    for i in range(axis_res ** 2):
        plt.subplot(axis_res, axis_res, i + 1)
        plt.imshow(x[i, :, :, 0], cmap=cmap)
        plt.axis("off")
        plt.title(y_cats[np.argmax(y[i, :])])


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

    if target_names:
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
    training_data_dir = "/media/mltest1/Dat Storage/Alex Data/Training"
    testing_data_dir = "/media/mltest1/Dat Storage/Alex Data/Testing"

    trained_model = train_model(train_datadir=training_data_dir, test_datadir=testing_data_dir, batch_size=1024,
                                epochs=20)
    plot_history(trained_model)

    preds, truth, files = validation_pred(trained_model, validation_datadir=testing_data_dir, batch_size=512)

    y_pred = np.argmax(preds, axis=1)
    y_truth = np.argmax(truth, axis=1)
    # conf_mat = metrics.confusion_matrix(y_truth, y_pred)
    # plot_confusion_matrix(conf_mat, original_categories)
    # print(metrics.classification_report(y_truth, y_pred, target_names=original_categories))

    # TODO: Save regression task as CSV
    # TODO: Save CNN task as CSV
    # TODO: Save training for both tasks as CSV

    # model_preds = pd.DataFrame()
