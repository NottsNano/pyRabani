import itertools

import numpy as np
from tensorflow.python.keras.models import load_model

from CNN.CNN_training import h5RabaniDataGenerator
from CNN.stats_plotting import all_preds_histogram
from Filters.alignerwthreshold import tmp_img_loader
from Rabani_Generator.plot_rabani import show_image


def validation_pred_generator(model, validation_datadir, y_params, y_cats, batch_size):
    validation_generator = h5RabaniDataGenerator(validation_datadir, batch_size=batch_size,
                                                 is_train=False, imsize=128, output_parameters_list=y_params,
                                                 output_categories_list=y_cats)
    validation_generator.is_validation_set = True

    validation_preds = model.predict_generator(validation_generator, steps=validation_generator.__len__())
    validation_truth = validation_generator.y_true

    return validation_preds, validation_truth


class ImageClassifier:
    def __init__(self, img_arr, model, window_jump=4):
        self.img_arr = img_arr

        self.model = model
        self.network_img_size = self.model.input_shape[1]

        self.jump = window_jump
        self.cnn_arr = None

        self.preds = None
        self.majority_preds = None

    def wrap_image(self):
        # Figure out how many "windows" to make
        num_jumps = int((len(self.img_arr) - self.network_img_size) / self.jump)
        jump_idx = itertools.product(np.arange(num_jumps), np.arange(num_jumps))

        # Copy each window out
        self.cnn_arr = np.zeros((num_jumps ** 2, self.network_img_size, self.network_img_size, 1))
        for i, (jump_i, jump_j) in enumerate(jump_idx):
            self.cnn_arr[i, :, :, 0] = img[(jump_i * self.jump): (jump_i * self.jump) + self.network_img_size,
                                       (jump_j * self.jump): (jump_j * self.jump) + self.network_img_size]

    def validation_pred_image(self):
        self.preds = self.model.predict(self.cnn_arr)
        self.majority_preds = np.mean(self.preds, axis=0)


if __name__ == '__main__':
    img = tmp_img_loader("Images/Parsed Dewetting 2020 for ML/thres_img/tp/000TEST.ibw")
    trained_model = load_model("Data/Trained_Networks/2020-03-24--16-58/model.h5")
    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']

    # Classify an image
    img_classifier = ImageClassifier(img, trained_model)
    img_classifier.wrap_image()
    img_classifier.validation_pred_image()

    # View some stuff
    all_preds_histogram(img_classifier.preds, cats)
    show_image(img)
