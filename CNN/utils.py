import itertools

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize


def make_pd_nans_identical(df, replacement_value=None):
    return df.replace({np.nan: replacement_value})


def adding_noise_test(img, model, cats, noise_steps, perc_noise, perc_std, savedir=None):
    """Progressively add noise to an image and classifying it"""

    from Analysis.plot_rabani import show_image
    from Analysis.get_stats import all_preds_histogram
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


def minkowski_stability_test(filepath, window_size):
    from Filters.screening import FileFilter
    from skimage import measure
    from Analysis.plot_rabani import show_image

    filterer = FileFilter()
    _, _, _, _, _, data, _, _ = filterer._load_and_preprocess(filepath=filepath)
    wrapped_arr = filterer._wrap_image_to_tensorflow(data, window_size, zigzag=True)

    fig, axs = plt.subplots(1, 4)
    lims = [-0.00025, -0.001, -0.01, -0.03, -0.04]
    for lim in lims:
        axs[1].axhline(lim, color='k', linestyle='--')

    axs[2].set_ylim(0, 1)
    axs[3].set_ylim(50, 250)

    axs[1].set_xlabel("Subimage Number")
    axs[2].set_xlabel("Subimage Number")
    axs[3].set_xlabel("Subimage Number")
    axs[1].set_ylabel("Normalised Euler Number")
    axs[2].set_ylabel("Eccentricity")
    axs[3].set_ylabel("Equivalent Diameter")

    show_image(data, axis=axs[0])
    for i, img in enumerate(wrapped_arr):
        region = measure.regionprops((img[:, :, 0] != 0) + 1)[1]
        euler_num = region["euler_number"] / np.sum(img == 1)
        eccentricity = region["eccentricity"]
        equivalent_diameter = region["equivalent_diameter"]
        axs[1].plot(i, euler_num, 'rx')
        axs[2].plot(i, eccentricity, 'rx')
        axs[3].plot(i, equivalent_diameter, 'rx')


def adding_noise_euler_test(num_steps, perc_noise, save=True):
    from Rabani_Simulation.rabani import rabani_single
    from CNN.CNN_training import h5RabaniDataGenerator
    from Analysis.plot_rabani import show_image
    from skimage import measure
    from matplotlib.ticker import PercentFormatter

    # Gen rabani
    img, _ = rabani_single(kT=0.12, mu=2.9, MR=1, C=0.3, e_nl=1.5, e_nn=2, L=200, MCS_max=5000, early_stop=True)

    # Set up figure
    fig, axs = plt.subplots(1, 2)
    lims = [-0.00025, -0.001, -0.01, -0.03, -0.04]
    axs[1].set_xlim(0, num_steps * perc_noise)
    axs[1].set_ylim(lims[-1], 0)
    for lim in lims:
        axs[1].axhline(lim, color='k', linestyle='--')
    axs[1].set_xlabel("Percentage Speckle Noise")
    axs[1].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[1].set_ylabel("Euler Number")

    # Continuously calculate Euler number while adding speckle noise
    for i in range(num_steps):
        region = (measure.regionprops((img != 0) + 1)[0])
        euler_num = region["euler_number"] / np.sum(img == 2)
        axs[1].plot(i * perc_noise, euler_num, 'rx')

        show_image(img, axis=axs[0])

        img = h5RabaniDataGenerator.speckle_noise(img, perc_noise, perc_std=None, randomness="batchwise",
                                                  num_uniques=4, scaling=False)
        del region

        if save:
            plt.savefig(f"/home/mltest1/tmp/pycharm_project_883/Data/Plots/euler_noise_comp/{i}.png")


def single_prediction_with_noise(img, cnn_model, perc_noise, perc_std):
    from CNN.CNN_training import h5RabaniDataGenerator
    from CNN.CNN_prediction import ImageClassifier

    img = img.copy()  # Do this because of immutability!
    img = h5RabaniDataGenerator.speckle_noise(img, perc_noise, perc_std)[0, :, :, 0]
    img_classifier = ImageClassifier(img, cnn_model)
    img_classifier.cnn_classify()

    return img_classifier


def zigzag_product(iterable_1, iterable_2):
    """
    Zigzag along two iterables to avoid discontinuities when indexing windows

    Parameters
    ----------
    iterable_1 : iterable
    iterable_2 : iterable

    Examples
    --------
    Without zigzagging
    >>> list(itertools.product([1,2,3],[1,2,3]))
    [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]

    With zigzagging
    >>> zigzag_product([1,2,3],[1,2,3])
    [(1,1), (1,2), (1,3), (2,3), (2,2), (2,1), (3,1), (3,2), (3,3)]
    """

    assert len(iterable_1) == len(iterable_2)
    window_length = len(iterable_1)

    iterated = list(itertools.product(iterable_1, iterable_2))
    new_iterated = []
    for cnt, i in enumerate(range(0, len(iterated), window_length)):
        if cnt % 2 == 1:
            new_iterated += list(reversed(iterated[i:i + window_length]))
        else:
            new_iterated += iterated[i:i + window_length]

    return new_iterated


def resize_image(image, newsize):
    """Enlarge image"""
    assert image.shape[0] <= newsize, f"New size ({newsize}) must be larger than original size ({image.shape[0]})"
    if image.shape[0] != newsize:
        num_tiles = newsize / image.shape[0]
        if num_tiles % 1 == 0:
            # Resize 2^n
            new_image = np.repeat(np.repeat(image, int(num_tiles), axis=0), int(num_tiles), axis=1)
        else:
            # Resize by nn-interpolation
            new_image = (resize(image, (newsize, newsize), order=0) * 255 // 2).astype(int)

            # Fix aliasing making NP -> L
            if np.all(np.unique(image) == [0, 2]):
                new_image[new_image == 1] = 2
    else:
        new_image = image

    return new_image


def remove_least_common_level(image):
    level_vals, counts = np.unique(image, return_counts=True)

    if len(counts) > 2:
        if (np.sum(image == 1) / image.shape[0] ** 2 >= 0.4) and (
                np.sum(image == 0) / image.shape[0] ** 2 >= 0.02):  # Hole, so don't remove substrate
            least_common_ind = 2
        else:
            least_common_ind = np.argmin(counts)
            least_common_val = level_vals[least_common_ind]

        common_vals = np.delete(level_vals, least_common_ind)

        replacement_inds = np.nonzero(image == least_common_ind)
        replacement_vals = np.random.choice(common_vals, size=(len(replacement_inds[0]),), p=[0.5, 0.5])

        image[replacement_inds[0], replacement_inds[1]] = replacement_vals

    return image


def normalise(image):
    image -= image.min()
    image /= image.max()

    return image
