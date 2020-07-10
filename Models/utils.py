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


def ind_to_onehot(y_preds):
    if np.array(y_preds).ndim != 2:
        return np.eye(np.max(y_preds) + 1)[np.array(y_preds)]
    else:
        return y_preds


def onehot_to_ind(y_preds):
    if np.array(y_preds).ndim != 1:
        return np.argmax(y_preds, axis=1)
    else:
        return y_preds


def ensure_dframe_is_pandas(dframe):
    if type(dframe) is str:
        dframe = pd.read_csv(dframe)

    return dframe
