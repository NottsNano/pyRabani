import warnings

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize


def make_pd_nans_identical(df, replacement_value=None):
    return df.replace({np.nan: replacement_value})


def adding_noise_test(img, model, cats, noise_steps, perc_noise, perc_std, savedir=None):
    """Progressively add noise to an image and classifying it"""

    from Rabani_Generator.plot_rabani import show_image
    from CNN.get_stats import all_preds_histogram
    fig, axes = plt.subplots(1, 2)
    fig.tight_layout(pad=3)
    img = img.copy()
    for i in range(noise_steps):
        axes[0].clear()
        axes[1].clear()

        img_classifier = single_prediction_with_noise(img, model, perc_noise, perc_std)

        show_image(img, axis=axes[0])
        all_preds_histogram(img_classifier.cnn_preds, cats, axis=axes[1])

        if savedir:
            plt.savefig(f"{savedir}/img_{i}.png")


def single_prediction_with_noise(img, cnn_model, perc_noise, perc_std):
    from CNN.CNN_training import h5RabaniDataGenerator
    from CNN.CNN_prediction import ImageClassifier

    img = img.copy()  # Do this because of immutability!
    img = h5RabaniDataGenerator.speckle_noise(img, perc_noise, perc_std)[0, :, :, 0]
    img_classifier = ImageClassifier(img, cnn_model)
    img_classifier.cnn_classify()

    return img_classifier


def power_resize(image, newsize):
    """Enlarge image by a factor of ^2"""
    if image.shape[0] != newsize:
        num_tiles = newsize / image.shape[0]
        if num_tiles % 1 == 0:
            new_image = np.repeat(np.repeat(image, int(num_tiles), axis=0), int(num_tiles), axis=1)
        else:
            warnings.warn(
                f"Scaling is not ^2 (Requested {image.shape[0]} -> {newsize}). Using sklearn nearest-neighbour")
            new_image = nn_resize(image, newsize)
    else:
        new_image = image

    return new_image


def nn_resize(image, newsize):
    """Resize image by any amount, with nearest-neighbour interpolation"""
    return resize(image, (newsize, newsize), order=0, anti_aliasing=True)
