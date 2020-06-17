import glob
import os
import warnings
from ast import literal_eval

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors, pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator
from skimage import measure
from skimage.filters import gaussian
from tensorflow.python.keras.models import load_model

from CNN.utils import resize_image, remove_least_common_level

cmap_rabani = colors.ListedColormap(["black", "white", "orange"])
boundaries = [0, 0.5, 1]
norm = colors.BoundaryNorm(boundaries, cmap_rabani.N, clip=True)


def dualscale_plot(xaxis, yaxis, root_dir, num_axis_ticks=15, trained_model=None, categories=None, img_res=None):
    """Plot two variables against another, and optionally the CNN predictions"""

    from CNN.CNN_training import h5RabaniDataGenerator
    files = os.listdir(root_dir)

    # Find axis details to allow for preallocation
    x_range_all = np.zeros((len(files),))
    y_range_all = np.zeros((len(files),))
    m_all = np.zeros((len(files),))
    img_res_all = np.zeros((len(files),))

    for i, file in enumerate(files):
        img_file = h5py.File(f"{root_dir}/{file}", "r")
        x_range_all[i] = img_file.attrs[xaxis]
        y_range_all[i] = img_file.attrs[yaxis]
        m_all[i] = img_file["sim_results"]["num_mc_steps"][()]
        img_res_all[i] = len(img_file["sim_results"]["image"])

    assert len(np.unique(x_range_all)) == len(
        np.unique(y_range_all)), f"{xaxis} must have same simulation resolution as {yaxis}"
    axis_res = len(np.unique(x_range_all))
    if not img_res and len(np.unique(img_res_all)) == 1:
        img_res = int(np.unique(img_res_all))
    else:
        assert img_res, "If data folder has multiple values of L, img_res must be defined"

    x_range = [np.min(x_range_all), np.max(x_range_all)]
    y_range = [np.min(y_range_all), np.max(y_range_all)]

    # Preallocate parsers
    big_img_arr = np.zeros((img_res * axis_res, img_res * axis_res))
    preds_arr = np.zeros((img_res * axis_res, img_res * axis_res))
    x_vals = np.linspace(x_range[0], x_range[1], axis_res)
    y_vals = np.linspace(y_range[0], y_range[1], axis_res)
    eulers = np.zeros((axis_res, axis_res))
    eulers_cmp = np.zeros((axis_res, axis_res))

    for i, file in enumerate(files):
        img_file = h5py.File(f"{root_dir}/{file}", "r")

        # Find most appropriate location to place image in image grid
        x_ind = np.searchsorted(x_vals, img_file.attrs[xaxis])
        y_ind = np.searchsorted(y_vals, img_file.attrs[yaxis])

        bin_img = remove_least_common_level(img_file["sim_results"]["image"][()])
        img = resize_image(bin_img, img_res)

        big_img_arr[(y_ind * img_res):((y_ind + 1) * img_res), (x_ind * img_res):((x_ind + 1) * img_res)] = np.flipud(
            img)

        # If there's a trained model input, make an array of predictions
        if trained_model:
            assert categories, "Need categories if also inputting a model"
        if categories:
            if trained_model:
                img = np.float64(img)
                shaped = h5RabaniDataGenerator._patch_binarisation(np.expand_dims(np.expand_dims(img, 0), -1))
                pred = np.argmax(trained_model.predict(shaped))
            else:
                pred = cats.index(img_file.attrs["category"])

            preds_arr[(y_ind * img_res):((y_ind + 1) * img_res), (x_ind * img_res):((x_ind + 1) * img_res)] = pred

        # Parse the normalised euler number
        eulers[y_ind, x_ind] = img_file['sim_results']["region_props"]["normalised_euler_number"][()]
        reg = measure.regionprops((img_file['sim_results']["image"][()] != 0) + 1)[0]
        eulers_cmp[y_ind, x_ind] = reg["euler_number"] / np.sum(img_file['sim_results']["image"][()] == 2)

    # Plot
    num_tick_skip = len(y_vals) // np.min((num_axis_ticks, len(y_vals)))

    x_labels = [f"{x_val:.2f}" for x_val in x_vals]
    y_labels = [f"{y_val:.2f}" for y_val in y_vals]

    blank_labels_mu = [None] * len(x_labels)
    blank_labels_y = [None] * len(y_labels)
    blank_labels_mu[::num_tick_skip] = x_labels[::num_tick_skip]
    blank_labels_y[::num_tick_skip] = y_labels[::num_tick_skip]

    # Sample grid
    fig1, ax1 = plt.subplots()
    plt.imshow(big_img_arr, cmap=cmap_rabani, origin="lower")
    if categories:
        cmap_pred = get_cmap("viridis", len(categories))
        cax1 = plt.imshow(preds_arr, cmap=cmap_pred, origin="lower", alpha=0.6)
        cbar1 = fig1.colorbar(cax1, ticks=np.arange(np.max(preds_arr) + 1))
        cbar1.ax.set_yticklabels(categories)

    plt.xticks(np.arange(len(y_vals)) * img_res + img_res / 2, blank_labels_mu, rotation=90)
    plt.yticks(np.arange(len(x_vals)) * img_res + img_res / 2, blank_labels_y)
    ax1.set_xticks([x * img_res for x in range(axis_res)], minor=True)
    ax1.set_yticks([y * img_res for y in range(axis_res)], minor=True)

    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)

    plt.grid(which="minor", ls="-", lw=2, color="r")

    # Euler
    fig2, ax2 = plt.subplots()
    cax2 = ax2.matshow(eulers, origin="lower", cmap="jet")

    X2, Y2 = np.meshgrid(np.arange(len(eulers)), np.arange(len(eulers)))
    cnts2 = plt.contour(X2, Y2, gaussian(eulers, 1), levels=15, colors="w", linestyles="solid")

    # Remove small contour circles
    for level in cnts2.collections:
        for kp, path in reversed(list(enumerate(level.get_paths()))):
            verts = path.vertices
            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
            if diameter < 10:
                del (level.get_paths()[kp])

    # Pretty up axis
    ax2.xaxis.tick_bottom()
    ax2.set_xticklabels([''] + x_labels[::num_tick_skip], rotation=90)
    ax2.set_yticklabels([''] + y_labels[::num_tick_skip])
    ax2.xaxis.set_major_locator(MultipleLocator(num_tick_skip))
    ax2.yaxis.set_major_locator(MultipleLocator(num_tick_skip))

    cbar2 = fig2.colorbar(cax2)
    cbar2.add_lines(cnts2)
    cbar2.set_label('Normalised Euler Characteristic', rotation=270)
    cbar2.ax.get_yaxis().labelpad = 15

    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)

    return big_img_arr, eulers


def plot_threshold_selection(root_dir, categories, img_res, plot_config=(5, 5), trained_model=None):
    """Plot a selection of images between a range of normalised euler numbers,
    to eventually determine training labels"""
    # Setup and parse input
    files = os.listdir(root_dir)

    fig, axs = plt.subplots(1, len(categories), sharex=True, sharey=True)

    # For each threshold
    for plot_num, category in enumerate(categories):

        plot_i = -1
        plot_j = 0

        big_img = np.zeros((img_res * plot_config[0], img_res * plot_config[1]))

        # For each file
        for file in files:
            # Determine the euler number
            img_file = h5py.File(f"{root_dir}/{file}", "r")

            if trained_model:
                bin_img = remove_least_common_level(img_file["sim_results"]["image"][()])
                img = resize_image(bin_img, img_res)
                img_category = categories[np.argmax(trained_model.predict(np.expand_dims(np.expand_dims(img, 0), -1)))]
            else:
                img_category = img_file.attrs["category"]

            # If we are going to plot
            if category == img_category:
                # Pick the subplot to plot on
                if plot_i >= plot_config[1] - 1:
                    plot_i = 0
                    plot_j += 1
                else:
                    plot_i += 1
                if plot_j >= plot_config[0]:
                    break

                # Plot
                big_img[plot_j * img_res:(plot_j + 1) * img_res,
                plot_i * img_res:(plot_i + 1) * img_res] = resize_image(
                    img_file["sim_results"]["image"][()], img_res)

        axs[plot_num].imshow(big_img, cmap=cmap_rabani)

        axs[plot_num].set_xticks(np.arange(0, img_res * plot_config[1], img_res))
        axs[plot_num].set_yticks(np.arange(0, img_res * plot_config[0], img_res))
        axs[plot_num].grid(ls="-", lw=2, color="r", )
        axs[plot_num].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        axs[plot_num].title.set_text(f"{category}")


def plot_random_simulated_images(datadir, num_imgs, y_params, y_cats, imsize=128, model=None):
    """Show a random selection of simulated images, with categories chosen by simulation/CNN prediction"""
    from CNN.CNN_training import h5RabaniDataGenerator

    img_generator = h5RabaniDataGenerator(datadir, network_type="classifier", batch_size=num_imgs, is_train=False,
                                          imsize=imsize, force_binarisation=False,
                                          output_parameters_list=y_params, output_categories_list=y_cats)
    img_generator.is_validation_set = True

    x, y = img_generator.__getitem__(None)
    axis_res = int(np.sqrt(num_imgs))

    plt.figure()
    for i in range(axis_res ** 2):
        plt.subplot(axis_res, axis_res, i + 1)
        plt.imshow(x[i, :, :, 0], cmap=cmap_rabani)
        plt.axis("off")

        if model:
            pred = model.predict(np.expand_dims(np.expand_dims(x[i, :, :, 0], 0), 3))
            cat = y_cats[np.argmax(pred[0, :])]
        else:
            cat = y_cats[np.argmax(y[i, :])]

        plt.title(cat)

    return x, y


def show_image(img, axis=None, title=None):
    """Show a binarized image

    Parameters
    ----------
    img : ndarray
        The image to plot
    axis : object of type matplotlib.pyplot.axis.axes, Optional
        If specified, the axis to plot the image to. If None (default), make a new figure
    title : str, Optional
        The title of the figure. Default None.
    """

    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.imshow(img, cmap=cmap_rabani)
    axis.axis("off")

    if title:
        axis.set_title(title)


def visualise_autoencoder_preds(model, *datadirs):
    """Show effect of denoising images

    Parameters
    ----------
    model : object of type tf.model
        Tensorflow model used for denoising
    datadirs : list of str
        List containing every directory containing files (e.g. "good" images, "bad" images, simulated images) to test
    """
    from CNN.CNN_prediction import validation_pred_generator
    from Filters.screening import FileFilter

    # For each provided directory
    imsize = model.input_shape[1]
    for datadir in datadirs:
        is_ibw = len(glob.glob(f"{datadir}/*.ibw")) == 0

        if is_ibw:  # If real files, preprocess
            truth = np.zeros((len(glob.glob(f"{datadir}/*.ibw")), imsize, imsize, 1))
            for i, file in enumerate(glob.glob(f"{datadir}/*.ibw")):
                filterer = FileFilter()
                _, _, _, _, _, binarized_data, _, _ = filterer._load_and_preprocess(file)

                if binarized_data is not None:
                    truth[i, :, :, 0] = binarized_data[:imsize, :imsize]
                else:
                    warnings.warn(f"Failed to preprocess {file}")
        else:  # If simulated files, use datagen
            preds, truth = validation_pred_generator(model=model,
                                                     validation_datadir=datadir,
                                                     network_type="autoencoder", y_params=["kT", "mu"],
                                                     y_cats=["liquid", "hole", "cellular", "labyrinth", "island"],
                                                     batch_size=10, imsize=imsize, steps=1)

        # Predict
        preds = model.predict(truth)

        # Ensure binarisation
        truth = np.round(truth)
        preds = np.round(preds)

        # Calculate mse
        mse = np.squeeze(((preds - truth) ** 2).mean(axis=(1, 2)))

        # Plot stacks of images
        pred = np.reshape(preds, (-1, preds.shape[1]))
        true = np.reshape(truth, (-1, truth.shape[1]))
        img = np.concatenate((truth, preds), axis=1)

        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"Mean error: {np.mean(mse) :.2f}")
        ax.imshow(img, cmap=cmap_rabani)
        ax.axis("off")

        # Show mse for each image
        ax.text(0, -10, "Orig")
        ax.text(imsize, -10, "Recons")
        for i, j in enumerate(mse):
            ax.text((imsize * 2.5), (i * imsize) + (imsize // 2), f"mse = {j:.2f}")


def plot_fail_reason_distribution(summary_csv):
    """Plot a bar chart showing the different reasons for failing. Some images can have multiple fail reasons!

    Parameters
    ----------
    summary_csv : str
        Path to a csv file containing classifications
    """

    # Read in and extract failure reasons
    df_summary = pd.read_csv(summary_csv)
    df_summary["Fail Reasons"] = df_summary["Fail Reasons"].fillna("['None']")
    df_summary["Fail Reasons"] = df_summary["Fail Reasons"].apply(literal_eval)

    fail_reasons = [item for sublist in df_summary["Fail Reasons"] for item in sublist]

    plt.figure()
    chart = sns.countplot(fail_reasons)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')


def plot_random_classified_images(summary_csv, category=None, max_ims=25):
    """Plot a random selection of CNN classified images

    Parameters
    ----------
    summary_csv : str
        Path to a csv file containing classifications
    category : str or None, Optional
        If specified, only plot classifications of this category. Default None
    max_ims : int, Optional
        The maximum number of images to plot. Default 25

    See Also
    --------
    Filters.directory_screening
    """

    from Filters.screening import FileFilter

    # Read in file
    df_summary = pd.read_csv(summary_csv)
    df_classified = df_summary[pd.isnull(df_summary["Fail Reasons"])]

    # Only consider category if specified
    if category:
        df_classified = df_classified[df_classified["CNN Classification"] == category]

    # Randomly plot up to max_ims binarized files
    ax_res = int(np.sqrt(max_ims))
    fig, ax = plt.subplots(ax_res, ax_res, sharex=True, sharey=True)
    ax = np.reshape(ax, -1)

    max_ims = np.min((max_ims, len(df_classified)))
    for i, file in enumerate(df_classified["File Path"].sample(max_ims)):
        filterer = FileFilter()
        _, _, _, _, _, binarized_data, _, _ = filterer._load_and_preprocess(file)

        show_image(binarized_data, axis=ax[i], title=file.split("/")[-1])


if __name__ == '__main__':
    dir = "Data/Simulated_Images/2020-06-12/11-06"
    model = load_model("Data/Trained_Networks/2020-06-12--13-07/model.h5")
    cats = ["liquid", "hole", "cellular", "labyrinth", "island"]
    big_img, eul = dualscale_plot(xaxis="mu", yaxis="kT", root_dir=dir, img_res=200, trained_model=model,
                                  categories=cats)
    plot_threshold_selection(root_dir=dir, categories=cats, img_res=200, trained_model=model)

    x, y = plot_random_simulated_images(dir, 25,
                                        ["kT", "mu"], ["liquid", "hole", "cellular", "labyrinth", "island"],
                                        200, model=model)
