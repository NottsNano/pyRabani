import itertools
import os
import subprocess

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dense, Conv2D, Dropout
from tensorflow.python.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

from utils import plot_history


class txtCNNAlexDataGenerator(Sequence):
    def __init__(self, root_dir, category_dir, batch_size, is_train,
                 horizontal_flip=True, vertical_flip=True):
        self.root_dir = root_dir
        self.cat_dir = category_dir
        self.batch_size = batch_size

        self.is_training_set = is_train
        self.is_validation_set = False

        self.hflip = horizontal_flip
        self.vflip = vertical_flip

        self.class_weights_dict = None
        self.__reset_file_iterator__()

        self.num_cats = 4

        self._get_image_res()

        self._get_class_weights()

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
            file_entry = f"{self.cat_dir}/category---{self._file_iterator.__next__().name}"
            idx = int(np.loadtxt(file_entry, delimiter=","))
            class_inds[i] = idx

        self.class_weights_dict = class_weight.compute_class_weight('balanced',
                                                                    np.arange(self.num_cats),
                                                                    class_inds)

        self.__reset_file_iterator__()

    def _get_image_res(self):
        """Open one file to check the image resolution"""
        self.image_res = len(np.loadtxt(self._file_iterator.__next__().path, delimiter=","))
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
        file_entry = f"{self.cat_dir}/category---{fname}"
        idx = int(np.loadtxt(file_entry, delimiter=","))

        out = np.zeros((self.num_cats,))
        out[idx] = 1

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


def train_classifier_model(train_datadir, test_datadir, cat_datadir, batch_size, epochs):
    # Set up generators
    train_generator = txtCNNAlexDataGenerator(train_datadir, cat_datadir, batch_size=batch_size, is_train=True)
    test_generator = txtCNNAlexDataGenerator(test_datadir, cat_datadir, batch_size=batch_size, is_train=False)

    # Set up model
    input_shape = (train_generator.image_res, train_generator.image_res, 1)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    model.add(Dense(train_generator.num_cats))

    csv_logger = CSVLogger('Data/Logs/classification_log.csv', append=True, separator=';')
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    model_checkpoint = ModelCheckpoint("Data/Models/classification_model.h5", monitor="val_acc", save_best_only=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Train
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        steps_per_epoch=train_generator.__len__() // 10,
                        validation_steps=test_generator.__len__() // 10,
                        epochs=epochs,
                        class_weight=train_generator.class_weights_dict,
                        callbacks=[csv_logger, early_stopping, model_checkpoint],
                        max_queue_size=100)

    return model


def validation_pred(model, validation_datadir, cat_datadir, batch_size):
    validation_generator = txtCNNAlexDataGenerator(validation_datadir, cat_datadir, batch_size=batch_size,
                                                   is_train=False)
    validation_generator.is_validation_set = True

    validation_preds = model.predict_generator(generator=validation_generator, steps=validation_generator.__len__())
    validation_truth = validation_generator.y_true
    filenames = validation_generator.filenames

    return validation_preds, validation_truth, filenames


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
    cat_dir = "/media/mltest1/Dat Storage/Alex Data/Categories"
    testing_data_dir = "/media/mltest1/Dat Storage/Alex Data/Testing"

    trained_model = train_classifier_model(train_datadir=training_data_dir,
                                           test_datadir=testing_data_dir, cat_datadir=cat_dir,
                                           batch_size=1024,
                                           epochs=10)
    plot_history(trained_model)

    # trained_model = load_model("Data/Models/classification_model.h5")
    plot_model(trained_model, to_file="Data/Models/classification_model.png")
    preds, truth, files = validation_pred(trained_model, validation_datadir=testing_data_dir, cat_datadir=cat_dir,
                                          batch_size=1024)

    cm = confusion_matrix(np.argmax(truth, 1), np.argmax(preds, 1))
    plot_confusion_matrix(cm, np.arange(4))

    pred_dframe = pd.DataFrame({"Filename": files,
                                "Real Category": (np.argmax(truth, axis=1) + 1),
                                "CNN Overall Prediction": (np.argmax(preds, axis=1) + 1),
                                "CNN Confidence Cat 0": preds[:, 0],
                                "CNN Confidence Cat 1": preds[:, 1],
                                "CNN Confidence Cat 2": preds[:, 2],
                                "CNN Confidence Cat 3": preds[:, 3]})
    pred_dframe.to_csv("Data/Logs/classification_preds.csv")
