import itertools

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model


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


def predict_with_noise(img, model, cats, noise_nums, noise_steps, num_noise_pixels, savedir=None):
    """Progressively add noise to an image and classifying it"""

    fig, axes = plt.subplots(1, 2)
    fig.tight_layout(pad=3)

    for i in range(noise_steps):
        axes[0].clear()
        axes[1].clear()

        img_classifier = ImageClassifier(img, model)
        del img_classifier

        img_classifier = ImageClassifier(img, model)  # Do this because of immutability!
        img_classifier.wrap_image()
        img_classifier.validation_pred_image()

        show_image(img, axis=axes[0])
        all_preds_histogram(img_classifier.preds, cats, axis=axes[1])

        for noise_num in noise_nums:
            img[np.random.randint(low=0, high=len(img), size=num_noise_pixels),
                np.random.randint(low=0, high=len(img), size=num_noise_pixels)] = noise_num

        if savedir:
            plt.savefig(f"{savedir}_{i}.png")


def validation_pred_generator(model, validation_datadir, y_params, y_cats, batch_size, imsize=128):
    validation_generator = h5RabaniDataGenerator(validation_datadir, batch_size=batch_size,
                                                 is_train=False, imsize=imsize, output_parameters_list=y_params,
                                                 output_categories_list=y_cats)
    validation_generator.is_validation_set = True

    validation_preds = model.predict_generator(validation_generator, steps=validation_generator.__len__())
    validation_truth = validation_generator.y_true

    return validation_preds, validation_truth


if __name__ == '__main__':
    from CNN.CNN_training import h5RabaniDataGenerator
    from CNN.get_stats import all_preds_histogram
    from Filters.alignerwthreshold import tmp_img_loader
    from Rabani_Generator.plot_rabani import show_image

    trained_model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-25--15-54/model.h5")
    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']

    # Classify a real image
    imgold = tmp_img_loader("Images/Parsed Dewetting 2020 for ML/thres_img/tp/Si_benzene_0000.ibw").astype(int)
    img = imgold.copy()
    img[imgold == 1] = 2
    img[imgold == 0] = 0

    # See effect of adding noise to image
    predict_with_noise(img=img, cats=cats, model=trained_model, noise_nums=[0, 1, 2],
                       num_noise_pixels=(512 ** 2) // 200,
                       noise_steps=1)
