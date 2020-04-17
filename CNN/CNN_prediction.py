import itertools

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model


class ImageClassifier:
    """Majority classify an image with window rolling"""

    def __init__(self, img_arr, model, window_jump=4):
        self.img_arr = img_arr

        self.model = model
        if self.model:
            self.network_img_size = self.model.input_shape[1]

        self.jump = window_jump
        self.cnn_arr = None

        self.cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']
        self.cnn_preds = None
        self.cnn_majority_preds = None
        self.euler_preds = None
        self.euler_majority_preds = None

    def wrap_image(self):
        # Figure out how many "windows" to make
        num_jumps = int((len(self.img_arr) - self.network_img_size) / self.jump)
        jump_idx = itertools.product(np.arange(num_jumps), np.arange(num_jumps))

        # Copy each window out
        self.cnn_arr = np.zeros((num_jumps ** 2, self.network_img_size, self.network_img_size, 1))
        for i, (jump_i, jump_j) in enumerate(jump_idx):
            self.cnn_arr[i, :, :, 0] = self.img_arr[(jump_i * self.jump): (jump_i * self.jump) + self.network_img_size,
                                       (jump_j * self.jump): (jump_j * self.jump) + self.network_img_size]

    def cnn_classify(self):
        self.cnn_preds = self.model.predict(self.cnn_arr)
        self.cnn_majority_preds = np.mean(self.cnn_preds, axis=0)

    def euler_classify(self):
        cats = self.cats + ["none"]
        self.euler_preds = np.zeros((len(self.cnn_arr), len(cats)))

        for i, img in enumerate(self.cnn_arr):
            _, pred = RabaniSweeper.calculate_stats(img=img[:, :, 0], image_res=self.network_img_size, liquid_num=2,
                                                    substrate_num=0, nano_num=1)
            self.euler_preds[i, cats.index(pred)] = 1
        self.euler_majority_preds = np.mean(self.euler_preds, axis=0)


def plot_noisy_predictions(img, model, cats, noise_steps, perc_noise, perc_std, savedir=None):
    """Progressively add noise to an image and classifying it"""

    fig, axes = plt.subplots(1, 2)
    fig.tight_layout(pad=3)
    img = img.copy()
    for i in range(noise_steps):
        axes[0].clear()
        axes[1].clear()

        img_classifier = predict_with_noise(img, model, perc_noise, perc_std)

        show_image(img, axis=axes[0])
        all_preds_histogram(img_classifier.cnn_preds, cats, axis=axes[1])

        if savedir:
            plt.savefig(f"{savedir}/img_{i}.png")


def predict_with_noise(img, model, perc_noise, perc_std):
    img = img.copy()  # Do this because of immutability!
    img = h5RabaniDataGenerator.speckle_noise(img, perc_noise, perc_std)[0, :, :, 0]
    img_classifier = ImageClassifier(img, model)
    img_classifier.wrap_image()
    img_classifier.cnn_classify()

    return img_classifier


def validation_pred_generator(model, validation_datadir, mode, y_params, y_cats, batch_size, imsize=128, steps=None):
    """Prediction generator for simulated validation data"""
    validation_generator = h5RabaniDataGenerator(validation_datadir, network_structure=mode, batch_size=batch_size,
                                                 is_train=False, imsize=imsize, output_parameters_list=y_params,
                                                 output_categories_list=y_cats)
    validation_generator.is_validation_set = True

    if not steps:
        steps = validation_generator.__len__()

    validation_preds = model.predict_generator(validation_generator, steps=steps)
    if mode is "supervised":
        validation_truth = validation_generator.y_true

        validation_preds = validation_preds[:steps*batch_size, :]
        validation_truth = validation_truth[:steps*batch_size, :]
    else:
        validation_truth = validation_generator.x_true

        validation_preds = validation_preds[:steps*batch_size, :, :, 0]
        validation_truth = validation_truth[:steps*batch_size, :, :, 0]

    return validation_preds, validation_truth


if __name__ == '__main__':
    from CNN.get_stats import all_preds_histogram
    from Rabani_Generator.plot_rabani import show_image
    from CNN.CNN_training import h5RabaniDataGenerator
    from Rabani_Generator.gen_rabanis import RabaniSweeper

    trained_model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5")
    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']

    # Classify a real image
    # imgold = tmp_img_loader(
    #     "/home/mltest1/tmp/pycharm_project_883/Images/Parsed Dewetting 2020 for ML/RAW/DATA 3/Si_d8th_ring5_05mgmL_0002.ibw").astype(
    #     int)
    # img = imgold.copy()
    # img[imgold == 1] = 0
    # img[imgold == 0] = 2

    # See effect of adding noise to image
    # plot_noisy_predictions(img=img, cats=cats, model=trained_model, perc_noise=0.05, perc_std=0.001, noise_steps=1)
    # img_classifier = ImageClassifier(img, trained_model)  # Do this because of immutability!
    # img_classifier.wrap_image()
    # img_classifier.validation_pred_image()
    #
    # show_image(img)
    # all_preds_histogram(img_classifier.cnn_preds, cats)
