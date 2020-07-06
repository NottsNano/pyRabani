import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mode
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from sklearn import metrics
from tensorflow.python.keras.models import load_model


def plot_model_history(model):
    for plot_metric in model.metrics_names:
        plt.figure()
        plt.plot(model.history.history[plot_metric])
        plt.plot(model.history.history[f'val_{plot_metric}'])
        plt.legend([plot_metric, f'val_{plot_metric}'])
        plt.xlabel("Epoch")
        plt.ylabel(plot_metric)


def plot_confusion_matrix(y_truth, y_pred, cats, cmap=None, normalize=True, title=None):
    cm = metrics.confusion_matrix(y_truth, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()

    if cats:
        tick_marks = np.arange(len(cats))
        plt.xticks(tick_marks, cats, rotation=45)
        plt.yticks(tick_marks, cats)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if title:
        plt.title(title)


def all_preds_percentage(preds, cats, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)

    axis.pie(np.sum(preds, axis=0), labels=cats)
    axis.axis("equal")


def all_preds_histogram(preds, cats, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        axis.hist(preds[:, i], bins=100, alpha=0.8, range=[0, 1], density=True)

    if not axis.get_legend():
        axis.legend(cats)
        axis.set_xlabel("Network Confidence")
        axis.set_ylabel("Normalized Frequency")
        axis.set_ylim([0, 100])


def ROC_one_vs_all(majority_pred, truth, cats, title=None, axis=None):
    fpr = {}
    tpr = {}
    tholds = {}
    roc_auc = {}

    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        fpr[cat], tpr[cat], tholds[cat] = (metrics.roc_curve(
            truth[:, i].astype(int), majority_pred[:, i]))
        roc_auc[cat] = metrics.auc(fpr[cat], tpr[cat])
        axis.plot(fpr[cat], tpr[cat], label=f'{cat}, AUC = {roc_auc[cat]:.2f}')
    axis.plot([0, 1], [0, 1], 'k--')
    
    if title:
        axis.set_title(title)
    else:
        axis.set_title('Receiver Operating Characteristic')

    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1])
    axis.legend()

    return tpr, fpr, tholds, roc_auc


def PR_one_vs_all(majority_pred, truth, cats, title=None, axis=None):
    prec = {}
    recall = {}
    tholds = {}
    pr_auc = {}

    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        prec[cat], recall[cat], tholds[cat] = (metrics.precision_recall_curve(
            truth[:, i].astype(int), majority_pred[:, i]))
        pr_auc[cat] = metrics.auc(recall[cat], prec[cat])
        axis.plot(prec[cat], recall[cat], label=f'{cat}, AUC = {pr_auc[cat]:.2f}')

    if title:
        axis.set_title(title)
    else:
        axis.set_title('Precision-Recall')
    axis.set_xlabel('Precision')
    axis.set_ylabel('Recall')
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1])
    axis.legend()

    return recall, prec, tholds, pr_auc


if __name__ == '__main__':
    from Classify.prediction import validation_pred_generator
    from Analysis.plot_rabani import plot_random_simulated_images

    trained_model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5")

    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']
    params = ["kT", "mu"]

    # Predict simulated validation set
    validation_data_dir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/2020-03-30/16-44"
    y_preds, y_truth = validation_pred_generator(trained_model, validation_datadir=validation_data_dir,
                                                 y_params=params, y_cats=cats, batch_size=100, imsize=128)

    # Calculate CNN_classification stats
    ROC_one_vs_all(y_preds, y_truth, cats)
    test = PR_one_vs_all(y_preds, y_truth, cats)

    y_preds_arg = np.argmax(y_preds, axis=1)
    y_truth_arg = np.argmax(y_truth, axis=1)

    plot_random_simulated_images(validation_data_dir, num_imgs=25, y_params=params,
                                 y_cats=cats, imsize=256)

    plot_confusion_matrix(y_truth_arg, y_preds_arg, cats)
    print(metrics.classification_report(y_truth_arg, y_preds_arg, target_names=cats))


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

    # Get stats
    tot_area = np.sum(label_img > 0)

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