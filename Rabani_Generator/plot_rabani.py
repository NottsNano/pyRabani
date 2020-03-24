import os

import h5py
import numpy as np
from matplotlib import colors, pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator
from skimage import measure
from skimage.filters import gaussian
from tensorflow.python.keras.models import load_model


def power_resize(image, newsize):
    """Enlarge image by a factor of ^2"""
    num_tiles = (newsize / image.shape[0]) ** 0.5 % 2
    assert num_tiles == 0, "New image must be ^2 larger than original image"
    new_image = np.tile(image, int(num_tiles))

    return new_image


def dualscale_plot(xaxis, yaxis, root_dir, num_axis_ticks=15, trained_model=None, categories=None, img_res=None):
    """Plot two variables against another"""
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
        img = power_resize(img_file["sim_results"]["image"][()], img_res) * 255 // 2
        big_img_arr[(y_ind * img_res):((y_ind + 1) * img_res), (x_ind * img_res):((x_ind + 1) * img_res)] = np.flipud(
            img)

        # If there's a trained model input, make an array of predictions
        if trained_model:
            pred = np.argmax(
                trained_model.predict(np.expand_dims(np.expand_dims(img, 0), -1)))
            preds_arr[(y_ind * img_res):((y_ind + 1) * img_res), (x_ind * img_res):((x_ind + 1) * img_res)] = pred

        eulers[y_ind, x_ind] = img_file['sim_results']["region_props"]["normalised_euler_number"][()]
        reg = measure.regionprops((img_file['sim_results']["image"][()] != 0) + 1)[0]
        eulers_cmp[y_ind, x_ind] = reg["euler_number"] / np.sum(img_file['sim_results']["image"][()] == 2)

    # Plot
    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    num_tick_skip = len(y_vals) // np.min((num_axis_ticks, len(y_vals)))

    x_labels = [f"{x_val:.2f}" for x_val in x_vals]
    y_labels = [f"{y_val:.2f}" for y_val in y_vals]

    blank_labels_mu = [None] * len(x_labels)
    blank_labels_y = [None] * len(y_labels)
    blank_labels_mu[::num_tick_skip] = x_labels[::num_tick_skip]
    blank_labels_y[::num_tick_skip] = y_labels[::num_tick_skip]

    # Sample grid
    fig1, ax1 = plt.subplots()
    plt.imshow(big_img_arr, cmap=cmap, origin="lower")
    if trained_model:
        cmap_pred = get_cmap("viridis", np.max(preds_arr) + 1)
        cax1 = plt.imshow(preds_arr, cmap=cmap_pred, origin="lower", alpha=0.6)
        cbar1 = fig1.colorbar(cax1, ticks=np.arange(np.max(preds_arr) + 1))
        if categories:
            cbar1.ax.set_yticklabels(["hole", "liquid", "cellular", "labyrinth", "island"])

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


def plot_threshold_selection(root_dir, categories, img_res, plot_config=(5, 5)):
    """Plot a selection of images between a range of normalised euler numbers,
    to eventually determine training labels"""
    # Setup and parse input
    files = os.listdir(root_dir)
    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

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
                plot_i * img_res:(plot_i + 1) * img_res] = power_resize(
                    img_file["sim_results"]["image"][()], img_res) * 255 // 2

        axs[plot_num].imshow(big_img, cmap=cmap)

        axs[plot_num].set_xticks(np.arange(0, img_res * plot_config[1], img_res))
        axs[plot_num].set_yticks(np.arange(0, img_res * plot_config[0], img_res))
        axs[plot_num].grid(ls="-", lw=2, color="r", )
        axs[plot_num].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        axs[plot_num].title.set_text(f"{category}")


if __name__ == '__main__':
    dir = "Images/2020-03-10/15-35"
    model = load_model("new_model.h5")
    cats = ["hole", "liquid", "cellular", "labyrinth", "island"]
    big_img, eul = dualscale_plot(xaxis="mu", yaxis="kT", root_dir=dir, img_res=128)
    plot_threshold_selection(root_dir=dir, categories=cats, img_res=128)


def show_random_selection_of_images(datadir, num_imgs, y_params, y_cats, imsize=128):
    from CNN.CNN_training import h5RabaniDataGenerator

    img_generator = h5RabaniDataGenerator(datadir, batch_size=num_imgs, is_train=True, imsize=imsize)

    x, y = img_generator.__getitem__(None)
    axis_res = int(np.sqrt(num_imgs))

    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.figure()
    for i in range(axis_res ** 2):
        plt.subplot(axis_res, axis_res, i + 1)
        plt.imshow(x[i, :, :, 0], cmap=cmap)
        plt.axis("off")
        plt.title(y_cats[np.argmax(y[i, :])])
