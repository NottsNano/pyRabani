import itertools

import numpy as np
from tensorflow.python.keras.models import load_model

from CNN.CNN_training import h5RabaniDataGenerator
from Filters.alignerwthreshold import tmp_img_loader


def validation_pred_generator(model, validation_datadir, y_params, y_cats, batch_size):
    validation_generator = h5RabaniDataGenerator(validation_datadir, batch_size=batch_size,
                                                 is_train=False, imsize=128, output_parameters_list=y_params,
                                                 output_categories_list=y_cats)
    validation_generator.is_validation_set = True

    validation_preds = model.predict_generator(validation_generator, steps=validation_generator.__len__())
    validation_truth = validation_generator.y_true

    return validation_preds, validation_truth


class ImageClassifier:
    def __init__(self, img_arr, model_path, window_jump=4):
        self.img_arr = img_arr

        self.model = load_model(model_path)
        self.network_img_size = self.model.input_shape[1]

        self.jump = window_jump
        self.cnn_arr = None

    def wrap_image(self):
        # Figure out how many "windows" to make
        num_jumps = int((len(img) - self.network_img_size) / self.network_img_size)
        jump_idx = itertools.product(np.arange(num_jumps), np.arange(num_jumps))

        # Copy each window out
        self.cnn_arr = np.zeros((num_jumps ** 2, self.network_img_size, self.network_img_size, 1))
        for i, (jump_i, jump_j) in enumerate(jump_idx):
            self.cnn_arr[i, :, :, 0] = img[(jump_i * self.jump): (jump_i * self.jump) + self.network_img_size,
                                       (jump_j * self.jump): (jump_j * self.jump) + self.network_img_size]


def validation_pred_image(model, image):
    pass


if __name__ == '__main__':
    # Load an image
    img = tmp_img_loader("Images/Parsed Dewetting 2020 for ML/thres_img/tp/000TEST.ibw")
