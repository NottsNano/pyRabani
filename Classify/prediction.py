import itertools
import warnings

import numpy as np
from tensorflow.python.keras.models import load_model

from Analysis.get_stats import calculate_stats, calculate_normalised_stats
from Classify.CNN_training import h5RabaniDataGenerator
from Classify.utils import adding_noise_test, zigzag_product


class ImageClassifier:
    """
    Classifies a single image after subsampling it

    Parameters
    ----------
    img: ndarray
        Either a single 2D image of size larger than the category_model, or a single image of identical size
         to the category_model and windowed and wrapped to 4D
    window_stride: int or None
        The stride length to jump by when sub-sampling img to form an image. Default 4
    cnn_model: object of type tensorflow.category_model, optional
        A trained tensorflow model. Required for CNN classifications
    sklearn_model: object of type sklearn.classifier
        A trained sklearn classifier. Required for stats classifications
    """

    def __init__(self, img, cnn_model=None, sklearn_model=None, window_stride=8):
        self.img_arr = img

        self.cnn_model = cnn_model
        self.sklearn_model = sklearn_model

        if self.cnn_model:
            self.network_img_size = self.cnn_model.input_shape[1]
        else:
            warnings.warn("No CNN input. Setting window size to 200")
            self.network_img_size = 200

        self.jump = window_stride

        if img.ndim != 4:
            self.cnn_arr = self._wrap_image_to_tensorflow(img, self.network_img_size, self.jump)
        else:
            self.cnn_arr = img

        self.cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']
        self.cnn_preds = self.cnn_majority_preds = None
        self.euler_preds = self.euler_majority_preds = None
        self.minkowski_preds = self.minkowski_majority_preds = None

    @staticmethod
    def _wrap_image_to_tensorflow(img, network_img_size, stride, zigzag=False):
        """Subsamples an image to turn it into a tensorflow-compatible shape"""
        # Figure out how many "windows" to make
        num_jumps = int((len(img) - network_img_size) / stride)

        if zigzag:
            jump_idx = zigzag_product(np.arange(num_jumps), np.arange(num_jumps))
        else:
            jump_idx = itertools.product(np.arange(num_jumps), np.arange(num_jumps))

        # Copy each window out
        cnn_arr = np.zeros((num_jumps ** 2, network_img_size, network_img_size, 1))
        for i, (jump_i, jump_j) in enumerate(jump_idx):
            cnn_arr[i, :, :, 0] = img[(jump_i * stride): (jump_i * stride) + network_img_size,
                                  (jump_j * stride): (jump_j * stride) + network_img_size]

        return cnn_arr

    @staticmethod
    def _unwrap_image_from_tensorflow(imgs, output_img_size, zigzag=False):
        # Deduce how image was windowed to tensorflow shape
        imgs = imgs.astype(int)
        num_jumps = int(np.sqrt(len(imgs)))
        network_img_size = imgs.shape[1]
        stride = (output_img_size - network_img_size) // num_jumps

        if zigzag:
            jump_idx = zigzag_product(np.arange(num_jumps), np.arange(num_jumps))
        else:
            jump_idx = itertools.product(np.arange(num_jumps), np.arange(num_jumps))

        # Spread all wrapped images over new size
        big_arr = np.zeros((len(imgs), output_img_size, output_img_size))
        big_arr[:] = np.nan
        for i, (jump_i, jump_j) in enumerate(jump_idx):
            big_arr[i,
            (jump_i * stride): (jump_i * stride) + network_img_size,
            (jump_j * stride): (jump_j * stride) + network_img_size] = imgs[i, :, :, 0]

        voted_arr = np.nanmean(big_arr, axis=0).round()

        return voted_arr

    def cnn_classify(self, perc_noise=0.05, perc_std=0.001):
        noisy_array = h5RabaniDataGenerator.speckle_noise(self.cnn_arr, perc_noise, perc_std,
                                                          randomness="batchwise",
                                                          num_uniques=len(np.unique(self.cnn_arr[0, :, :, 0])) + 1)

        self.cnn_preds = self.cnn_model.predict(noisy_array)
        self.cnn_majority_preds = self._majority_preds(self.cnn_preds)

    def euler_classify(self):
        cats = self.cats + ["none"]
        self.euler_preds = np.zeros((len(self.cnn_arr), len(cats)))

        for i, img in enumerate(self.cnn_arr):
            _, pred = calculate_stats(img=img[:, :, 0], image_res=self.network_img_size, liquid_num=1,
                                      substrate_num=0, nano_num=1)
            self.euler_preds[i, cats.index(pred)] = 1
        self.euler_majority_preds = self._majority_preds(self.euler_preds)

    def minkowski_classify(self):
        SIA = np.zeros(len(self.cnn_arr))
        SIP = np.zeros(len(self.cnn_arr))
        SIH0 = np.zeros(len(self.cnn_arr))
        SIH1 = np.zeros(len(self.cnn_arr))

        for i, img in enumerate(self.cnn_arr):
            SIA[i], SIP[i], SIH0[i], SIH1[i] = calculate_normalised_stats(img[:, :, 0])

        x = np.hstack((SIA, SIP, SIH0, SIH1))
        self.minkowski_preds = self.sklearn_model.predict_proba(x)
        self.minkowski_majority_preds = self._majority_preds(self.minkowski_preds)

    @staticmethod
    def _majority_preds(arr):
        return np.mean(arr, axis=0)


def validation_pred_generator(model, validation_datadir, network_type, y_params, y_cats, batch_size, imsize=128,
                              steps=None):
    """Prediction generator for simulated validation data"""
    validation_generator = h5RabaniDataGenerator(validation_datadir, network_type=network_type, batch_size=batch_size,
                                                 is_train=False, imsize=imsize, output_parameters_list=y_params,
                                                 output_categories_list=y_cats)
    validation_generator.is_validation_set = True

    if not steps:
        steps = validation_generator.__len__()

    if network_type == "classifier":
        validation_preds = model.predict_generator(validation_generator, steps=steps)[:steps * batch_size, :]
        validation_truth = validation_generator.y_true[:steps * batch_size, :]
    elif network_type == "autoencoder":
        validation_preds = model.predict_generator(validation_generator, steps=steps)[:steps * batch_size, :, :, :]
        validation_truth = validation_generator.x_true[:steps * batch_size, :, :, :]

    return validation_preds, validation_truth


if __name__ == '__main__':
    from Filters.alignerwthreshold import tmp_img_loader

    trained_model = load_model(
        "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5")
    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']

    # Classify a real image
    imgold = tmp_img_loader(
        "/home/mltest1/tmp/pycharm_project_883/Images/Parsed Dewetting 2020 for ML/RAW/DATA 3/Si_d8th_ring5_05mgmL_0002.ibw").astype(
        int)
    img = imgold.copy()
    img[imgold == 1] = 0
    img[imgold == 0] = 2

    # See effect of adding noise to image
    adding_noise_test(img=img, cats=cats, model=trained_model, perc_noise=0.05, perc_std=0.001, noise_steps=1)
    # img_classifier = ImageClassifier(img, trained_model)  # Do this because of immutability!
    # img_classifier.wrap_image()
    # img_classifier.validation_pred_image()
    #
    # show_image(img)
    # all_preds_histogram(img_classifier.cnn_preds, cats)
