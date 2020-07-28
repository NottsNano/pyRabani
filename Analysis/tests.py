import glob
import os
import shutil

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from tqdm import tqdm

from Analysis.compare_assessments import _restore_stored_list
from Analysis.image_stats import calculate_normalised_stats
from Analysis.model_stats import confusion_matrix
from Analysis.plot_rabani import show_image
from Models.predict import ImageClassifier
from Models.train_regression import load_sklearn_model, convert_dframe_to_sklearn
from Models.utils import ensure_dframe_is_pandas, make_folder_if_not_exists


def adding_noise_test(img, model, cats, noise_steps, perc_noise, perc_std, savedir=None):
    """Progressively add noise to an image and classifying it"""

    from Analysis.plot_rabani import show_image
    from Analysis.model_stats import preds_histogram
    fig, axes = plt.subplots(1, 2)
    fig.tight_layout(pad=3)
    img = img.copy()
    for i in range(noise_steps):
        axes[0].clear()
        axes[1].clear()

        img_classifier = single_prediction_with_noise(img, model, perc_noise, perc_std)

        show_image(img, axis=axes[0])
        preds_histogram(img_classifier.cnn_preds, cats, axis=axes[1])

        if savedir:
            plt.savefig(f"{savedir}/img_{i}.png")


def everything_test(filepath, window_size, num_steps, perc_noise):
    from Filters.screening import FileFilter
    from skimage import measure
    from Analysis.plot_rabani import show_image
    from Models.h5_iterator import h5RabaniDataGenerator
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
    from Models.h5_iterator import h5RabaniDataGenerator
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
    from Models.h5_iterator import h5RabaniDataGenerator
    from Models.predict import ImageClassifier

    img = img.copy()  # Do this because of immutability!
    img = h5RabaniDataGenerator.speckle_noise(img, perc_noise, perc_std)[0, :, :, 0]
    img_classifier = ImageClassifier(img, cnn_model)
    img_classifier.cnn_classify()

    return img_classifier


def test_minkowski_scale_invariance(img, stride=8, max_subimgs=20):
    xaxis = np.arange(1, len(img))
    num_jumps = int((len(img)) / stride)
    SIA = np.zeros((len(img) - 1, num_jumps ** 2))
    SIP = np.zeros((len(img) - 1, num_jumps ** 2))
    SIE = np.zeros((len(img) - 1, num_jumps ** 2))

    SIA[:] = np.nan
    SIP[:] = np.nan
    SIE[:] = np.nan

    # For each window size, make sub-images
    for i in tqdm(xaxis[::2]):
        sub_imgs = ImageClassifier._wrap_image_to_tensorflow(img=img, network_img_size=i, stride=stride)[:, :, :, 0]

        # For each sub-image, calculate normalised stats
        rand_inds = np.random.choice(len(sub_imgs), replace=False, size=np.min((max_subimgs, len(sub_imgs))))
        rand_sub_imgs = sub_imgs[rand_inds, :, :]

        for j, sub_img in enumerate(rand_sub_imgs):
            SIA[i, j], SIP[i, j], SIE[i, j] = calculate_normalised_stats(sub_img)

    # Take means
    SIA_mean = np.nanmean(SIA, axis=1)
    SIA_std = np.nanstd(SIA, axis=1)
    SIP_mean = np.nanmean(SIP, axis=1)
    SIP_std = np.nanstd(SIP, axis=1)
    SIE_mean = np.nanmean(SIE, axis=1)
    SIE_std = np.nanstd(SIE, axis=1)

    # Plot
    fig, axs = plt.subplots(1, 4, sharex=True, figsize=(1280 / 96, 480 / 96))

    show_image(img, axis=axs[0])
    axs[1].errorbar(xaxis, SIA_mean, SIA_std, fmt='rx')
    axs[2].errorbar(xaxis, SIP_mean, SIP_std, fmt='rx')
    axs[3].errorbar(xaxis, SIE_mean, SIE_std, fmt='rx')

    axs[1].set_ylabel("SIA")
    axs[2].set_ylabel("SIP")
    axs[3].set_ylabel("SIE")
    axs[2].set_xlabel("Window Size")

    fig.tight_layout()


def test_filtering_fail_reasons(csv_path, classifier_type="CNN"):
    dframe = ensure_dframe_is_pandas(csv_path)

    cats = ["cellular", "labyrinth", "island", "fail"]

    truth = [cats.index(i) for i in (list(dframe["Manual Classification"]))]
    preds = np.zeros(len(truth))

    for i, row in dframe.iterrows():
        if (row[f"{classifier_type} Classification"] == "hole") or (type(row["Fail Reasons"]) is str):
            preds[i] = 3
        else:
            preds[i] = cats.index(row[f"{classifier_type} Classification"])

    confusion_matrix(y_pred=preds, y_truth=truth, cats=cats)


def tune_filtering_with_ROC(csv_path_good, csv_path_bad, labelevery=5, classifier_type="CNN"):
    dframe = pd.concat((ensure_dframe_is_pandas(csv_path_good), ensure_dframe_is_pandas(csv_path_bad)))
    dframe[[f"{classifier_type} Mean", f"{classifier_type} std"]] = _restore_stored_list(
        dframe[[f"{classifier_type} Mean", f"{classifier_type} std"]])

    means = np.max(np.array(dframe[f"{classifier_type} Mean"].tolist()), axis=1)
    stds = np.max(np.array(dframe[f"{classifier_type} std"].tolist()), axis=1)
    true = np.array(dframe["Manual Classification"].str.contains("Good")).astype(int)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].set_xlabel("Wrongly Filtered In")
    axs[1].set_xlabel("Wrongly Filtered In")
    axs[0].set_title("Mean Filter")
    axs[1].set_title("Std Dev Filter")
    axs[0].set_ylabel("Rightly Filtered In")

    for i, stat in enumerate([means, stds]):
        if i == 1:
            true = np.abs(1 - true)

        fpr, tpr, thold = metrics.roc_curve(y_true=true, y_score=stat)
        auroc = metrics.auc(fpr, tpr)
        axs[i].plot(fpr, tpr, label=f"AUROC = {auroc:.2f}")
        axs[i].legend()

        fpr_labels = fpr[::labelevery]
        tpr_labels = tpr[::labelevery]
        thold_labels = thold[::labelevery]
        for j in range(len(fpr_labels)):
            axs[i].annotate(s=f"{thold_labels[j]:.2f}", xy=(fpr_labels[j], tpr_labels[j]))

        axs[i].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        axs[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))


def select_random_images_from_filtered_dataset(filtered_dir, cats, num_imgs, output_dir):
    dframe = pd.DataFrame(columns=["Image Name", "Category", "Reasonable?"])
    row = 0
    for cat in cats:
        make_folder_if_not_exists(f"{output_dir}/{cat}")

        valid_ims = [img_name for img_name in os.listdir(f"{filtered_dir}/{cat}") if img_name[:5] == "image"]
        selected_ims = np.random.choice(valid_ims, num_imgs, replace=False)

        for im in selected_ims:
            shutil.copy(src=f"{filtered_dir}/{cat}/{im}", dst=f"{output_dir}/{cat}/{im}")

            row += 1
            dframe.loc[row, ["Image Name"]] = [im]
            dframe.loc[row, ["Category"]] = [cat]

    dframe.to_csv(f"{output_dir}/random_selection.csv")


def select_random_images_from_unfiltered_dataset(filtered_dir, cats, num_imgs, output_dir):
    all_files = [f for f in glob.glob(f"{filtered_dir}/**/*.png", recursive=True) if "stats_" in f]
    selected_ims = np.random.choice(all_files, num_imgs * len(cats), replace=False)

    dframe = pd.DataFrame(columns=["Image Name", "Category", "Reasonable?"])
    row = 0

    for i, cat in enumerate(cats):
        make_folder_if_not_exists(f"{output_dir}/{cat}")
        for im in selected_ims[i * num_imgs: (i + 1) * num_imgs]:
            img_name = im.split("/")[-1]
            shutil.copy(src=f"{im}", dst=f"{output_dir}/{cat}/{img_name}")
            row += 1
            dframe.loc[row, ["Image Name"]] = [im]
            dframe.loc[row, ["Category"]] = [cat]

    dframe.to_csv(f"{output_dir}/random_selection.csv")


def plot_regression_3d(sklearn_model, cats, dframe_path):
    x_test, y_test = convert_dframe_to_sklearn(
            dframe=dframe_path,
            data_column_headers=["SIA", "SIP", "SIE"],
            cats=cats)
    y_pred = sklearn_model.predict(x_test)

    SIA = x_test[:, 0]
    SIP = x_test[:, 1]
    SIE = x_test[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(np.log(SIA), np.log(SIP), np.log(SIE), c=y_pred)
    ax.set_xlabel("Log SIA")
    ax.set_ylabel("Log SIP")
    ax.set_zlabel("Log SIE")

    ax.legend(sc.legend_elements()[0], cats)