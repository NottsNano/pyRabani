import itertools

import numpy as np
from sklearn import metrics
from tensorflow.python.keras.models import load_model

from CNN.CNN_training import h5RabaniDataGenerator
from CNN.stats_plotting import all_preds_histogram, plot_confusion_matrix
from Filters.alignerwthreshold import tmp_img_loader
from Rabani_Generator.plot_rabani import show_image, show_random_selection_of_images


def validation_pred_generator(model, validation_datadir, y_params, y_cats, batch_size, imsize=128):
    validation_generator = h5RabaniDataGenerator(validation_datadir, batch_size=batch_size,
                                                 is_train=False, imsize=imsize, output_parameters_list=y_params,
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
    from Rabani_Generator.plot_rabani import power_resize

    trained_model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-25--15-54/model.h5")

    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']
    params = ["kT", "mu"]

    # Calculate stats for simulated validation set
    validation_data_dir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/2020-03-12/14-33"
    preds, truth = validation_pred_generator(trained_model, validation_datadir=validation_data_dir,
                                   y_params=params, y_cats=cats, batch_size=100, imsize=256)

    y_pred = np.argmax(preds, axis=1)
    y_truth = np.argmax(truth, axis=1)

    show_random_selection_of_images(validation_data_dir, num_imgs=25, y_params=params,
                                    y_cats=cats, imsize=256, model=trained_model)

    conf_mat = metrics.confusion_matrix(y_truth, y_pred)
    plot_confusion_matrix(conf_mat, cats)
    print(metrics.classification_report(y_truth, y_pred, target_names=cats))

    # Classify a real image
    imgold = tmp_img_loader("Images/Parsed Dewetting 2020 for ML/thres_img/tp/000TEST.ibw").astype(int)
    img = imgold.copy()
    img[imgold==1] = 0
    img[imgold==0] = 2

    img_classifier = ImageClassifier(img, trained_model)
    img_classifier.wrap_image()
    img_classifier.validation_pred_image()

    # View an example real image and its classification
    all_preds_histogram(img_classifier.preds, cats)
    show_image(img)
