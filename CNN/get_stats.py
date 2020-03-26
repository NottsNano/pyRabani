import itertools

import numpy as np
from matplotlib import pyplot as plt
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


def plot_confusion_matrix(y_truth, y_pred, cats, cmap=None, normalize=True):
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


def ROC_one_vs_all(majority_pred, truth, cats, axis=None):
    fpr = tpr = tholds = roc_auc = {}

    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        fpr[cat], tpr[cat], tholds[cat] = (metrics.roc_curve(
            truth[:, i].astype(int), majority_pred[:, i]))
        roc_auc[cat] = metrics.auc(fpr[cat], tpr[cat])
        axis.plot(fpr[cat], tpr[cat], label=f'{cat}, AUC = {roc_auc[cat]:.2f}')

    axis.set_title('Receiver Operating Characteristic')
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1])
    axis.legend()

    return tpr, fpr, tholds, roc_auc


def PR_one_vs_all(majority_pred, truth, cats, axis=None):
    prec = recall = tholds = pr_auc = {}

    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        prec[cat], recall[cat], tholds[cat] = (metrics.precision_recall_curve(
            truth[:, i].astype(int), majority_pred[:, i]))
        pr_auc[cat] = metrics.auc(prec[cat], recall[cat])
        axis.plot(prec[cat], recall[cat], label=f'{cat}, AUC = {pr_auc[cat]:.2f}')

    axis.set_title('Precision-Recall')
    axis.set_xlabel('Precision')
    axis.set_ylabel('Recall')
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1])
    axis.legend()

    return recall, prec, tholds, pr_auc


if __name__ == '__main__':
    from CNN.CNN_prediction import validation_pred_generator
    from Rabani_Generator.plot_rabani import show_random_selection_of_images

    trained_model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-25--15-54/model.h5")

    cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']
    params = ["kT", "mu"]

    # Predict simulated validation set
    validation_data_dir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/2020-03-12/14-33"
    preds, truth = validation_pred_generator(trained_model, validation_datadir=validation_data_dir,
                                             y_params=params, y_cats=cats, batch_size=100, imsize=256)

    # Calculate classification stats
    ROC_one_vs_all(preds, truth, cats)
    PR_one_vs_all(preds, truth, cats)

    y_pred = np.argmax(preds, axis=1)
    y_truth = np.argmax(truth, axis=1)

    show_random_selection_of_images(validation_data_dir, num_imgs=25, y_params=params,
                                    y_cats=cats, imsize=256, model=trained_model)

    plot_confusion_matrix(y_truth, y_pred, cats)
    print(metrics.classification_report(y_truth, y_pred, target_names=cats))
