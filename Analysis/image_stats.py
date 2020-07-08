import numpy as np
from scipy.stats import mode
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from sklearn import metrics
from tensorflow.python.keras.models import load_model

from Analysis.model_stats import confusion_matrix, ROC_one_vs_all, PR_one_vs_all


def calculate_stats(img, image_res, substrate_num=0, liquid_num=1, nano_num=2):
    # Region Properties
    region = (measure.regionprops((img != 0) + 1)[0])

    # Broadly estimate category
    if int(mode(img, axis=None).mode) == liquid_num:
        if np.sum(img == substrate_num) / image_res ** 2 >= 0.02:
            # Hole if dominant category is water and also has an amount of substrate
            cat = "hole"
        else:
            # Liquid if dominant category is water (==1)
            cat = "liquid"
    elif -0.00025 <= region["euler_number"] / np.sum(img == nano_num):
        # Cell/Worm if starting to form
        cat = "cellular"
    elif -0.01 <= region["euler_number"] / np.sum(img == nano_num) < -0.001:
        # Labyrinth
        cat = "labyrinth"
    elif region["euler_number"] / np.sum(img == nano_num) <= -0.03:
        # Island
        cat = "island"
    else:
        cat = "none"

    return region, cat


def calculate_normalised_stats(img):
    # Find unique sections
    img_inv = np.abs(1 - img)
    label_img = label(closing(img, square(3)))
    label_img_inv = label(closing(img_inv, square(3)))

    # image_label_overlay = label2rgb(label_img, image=img, bg_label=0)
    # image_label_overlay_inv = label2rgb(label_img_inv, image=img, bg_label=0)

    # Get stats
    H0 = label_img.max()
    H1 = label_img_inv.max()
    if H0 > H1:
        average_particle_size = np.sum(label_img > 0) / H0
    else:
        average_particle_size = np.sum(label_img_inv > 0) / H1

    tot_perimeter = 0
    for region, region_inv in zip(regionprops(label_img), regionprops(label_img_inv)):
        tot_perimeter += region["perimeter"] + region_inv["perimeter"]

    # Make stats size invariant
    SIA = average_particle_size / np.size(label_img)
    SIP = tot_perimeter / (H0 * np.sqrt(average_particle_size))
    SIH0 = H0 * average_particle_size
    SIH1 = H1 * average_particle_size

    return SIA, SIP, SIH0, SIH1


if __name__ == '__main__':
    from Models.train_CNN import validate_CNN
    from Analysis.plot_rabani import plot_random_simulated_images

    trained_model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5")

    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']
    params = ["kT", "mu"]

    # Predict simulated validation set
    validation_data_dir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/2020-03-30/16-44"
    y_preds, y_truth = validate_CNN(trained_model, validation_datadir=validation_data_dir,
                                    y_params=params, y_cats=cats, batch_size=100, imsize=128)

    # Calculate CNN_classification stats
    ROC_one_vs_all(y_preds, y_truth, cats)
    test = PR_one_vs_all(y_preds, y_truth, cats)

    y_preds_arg = np.argmax(y_preds, axis=1)
    y_truth_arg = np.argmax(y_truth, axis=1)

    plot_random_simulated_images(validation_data_dir, num_imgs=25, y_params=params,
                                 y_cats=cats, imsize=256)

    confusion_matrix(y_truth_arg, y_preds_arg, cats)
    print(metrics.classification_report(y_truth_arg, y_preds_arg, target_names=cats))
