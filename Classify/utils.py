import glob
import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm


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


def everything_test(filepath, window_size, num_steps, perc_noise):
    from Filters.screening import FileFilter
    from skimage import measure
    from Analysis.plot_rabani import show_image
    from Classify.CNN_training import h5RabaniDataGenerator
    from matplotlib.ticker import PercentFormatter
    from tqdm import tqdm

    # Load in file
    filterer = FileFilter()
    _, _, _, _, _, data, _, _ = filterer._load_and_preprocess(filepath=filepath, threshold_method="multiotsu",
                                                              nbins=1000)
    wrapped_arr = filterer._wrap_image_to_tensorflow(data, window_size, zigzag=True)
    wrapped_arr_for_noise = wrapped_arr.copy()

    # Calculate stats as function of window num
    euler_nums = np.zeros(len(wrapped_arr))
    perimeters = np.zeros(len(wrapped_arr))
    for i, img in enumerate(tqdm(wrapped_arr)):
        region = measure.regionprops((img[:, :, 0] != 0) + 1)[1]
        euler_nums[i] = region["euler_number"] / np.sum(img == 1)
        perimeters[i] = region["perimeter"] / np.sum(img == 1)

    # Calculate stats as function of noise
    euler_nums_noise = np.zeros(num_steps)
    perimeters_noise = np.zeros(num_steps)
    euler_nums_noise_std = np.zeros(num_steps)
    perimeters_noise_std = np.zeros(num_steps)
    for i in tqdm(range(num_steps)):
        euler_nums_noise_step = np.zeros(len(wrapped_arr))
        perimeters_noise_step = np.zeros(len(wrapped_arr))
        for j, img in enumerate(wrapped_arr_for_noise):
            region = measure.regionprops((img[:, :, 0] != 0) + 1)[1]
            euler_nums_noise_step[j] = region["euler_number"] / np.sum(img == 1)
            perimeters_noise_step[j] = region["perimeter"] / np.sum(img == 1)

        euler_nums_noise[i] = np.mean(euler_nums_noise_step)
        euler_nums_noise_std[i] = np.std(euler_nums_noise_step)
        perimeters_noise[i] = np.mean(perimeters_noise_step)
        perimeters_noise_std[i] = np.std(perimeters_noise_step)

        wrapped_arr_for_noise = h5RabaniDataGenerator.speckle_noise(wrapped_arr_for_noise, perc_noise, perc_std=None,
                                                                    num_uniques=2, randomness="batchwise_flip",
                                                                    scaling=False)

    # Plot
    fig, axs = plt.subplots(1, 5, figsize=(1400 / 96, 500 / 96))

    [ax.set_xlabel("Subimage Number") for ax in axs[1:3]]
    [ax.set_xlabel("% Speckle Noise") for ax in axs[3:]]
    [ax.xaxis.set_major_formatter(PercentFormatter(xmax=1)) for ax in axs[3:]]

    [ax.set_ylabel("Normalised Euler Number") for ax in axs[1::2]]
    [ax.set_ylabel("Normalised Perimeter") for ax in axs[2::2]]

    lims = [-0.00025, -0.001, -0.01, -0.03, -0.04]
    for ax in axs[1::2]:
        for lim in lims:
            ax.axhline(lim, color='k', linestyle='--')

    show_image(data, axis=axs[0])
    axs[1].plot(euler_nums)
    axs[2].plot(perimeters)
    axs[3].errorbar(np.arange(num_steps) * perc_noise, euler_nums_noise, euler_nums_noise_std)
    axs[4].errorbar(np.arange(num_steps) * perc_noise, perimeters_noise, perimeters_noise_std)

    plt.tight_layout()


def minkowski_stability_test(filepath, window_size, save):
    from Filters.screening import FileFilter
    from skimage import measure
    from Analysis.plot_rabani import show_image

    filterer = FileFilter()
    _, _, _, _, _, data, _, _ = filterer._load_and_preprocess(filepath=filepath, threshold_method="multiotsu",
                                                              nbins=1000)
    wrapped_arr = filterer._wrap_image_to_tensorflow(data, window_size, zigzag=True)

    fig, axs = plt.subplots(1, 3, figsize=(1000 / 96, 480 / 96))
    lims = [-0.00025, -0.001, -0.01, -0.03, -0.04]
    for lim in lims:
        axs[1].axhline(lim, color='k', linestyle='--')

    axs[2].set_ylim(0, 1)
    # axs[3].set_ylim(50, 250)

    axs[1].set_xlabel("Subimage Number")
    axs[2].set_xlabel("Subimage Number")

    axs[1].set_ylabel("Normalised Euler Number")
    axs[2].set_ylabel("Normalised Perimeter")

    show_image(data, axis=axs[0])
    for i, img in enumerate(wrapped_arr):
        region = measure.regionprops((img[:, :, 0] != 0) + 1)[0]
        euler_num = region["euler_number"] / np.sum(img == 1)
        eccentricity = region["perimeter"] / np.sum(img == 1)
        axs[1].plot(i, euler_num, 'rx')
        axs[2].plot(i, eccentricity, 'rx')

    plt.tight_layout()

    if save:
        savedir = '/'.join(filepath.split('/')[-2:])
        plt.savefig(f"/home/mltest1/tmp/pycharm_project_883/Data/Plots/minkowski_stability/{savedir}")
        plt.close()


def adding_noise_euler_test(num_steps, perc_noise, save=True):
    from Rabani_Simulation.rabani import rabani_single
    from Classify.CNN_training import h5RabaniDataGenerator
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


def pick_random_images(dir, n_ims):
    from Filters.screening import FileFilter

    df_summary = pd.DataFrame(columns=["File Name", "Classification"])
    all_files = [f for f in glob.glob(f"{dir}/**/*.ibw", recursive=True)]
    rand_files = np.random.choice(all_files, n_ims)

    for i, filepath in enumerate(tqdm(rand_files)):
        filterer = FileFilter()
        _, _, _, _, flattened_data, _, _, _ = filterer._load_and_preprocess(filepath=filepath, threshold_method="otsu")

        if flattened_data is not None:
            plt.imsave(f"Data/Random_Images/{os.path.basename(filepath)}.png", flattened_data, cmap="RdGy")
            df_summary.loc[i, ["File Name"]] = [os.path.basename(filepath)]

    return df_summary


def single_prediction_with_noise(img, cnn_model, perc_noise, perc_std):
    from Classify.CNN_training import h5RabaniDataGenerator
    from Classify.prediction import ImageClassifier

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
