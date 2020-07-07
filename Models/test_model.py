import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from Models.utils import ind_to_onehot, onehot_to_ind


def plot_model_history(model):
    for plot_metric in model.metrics_names:
        plt.figure()
        plt.plot(model.history.history[plot_metric])
        plt.plot(model.history.history[f'val_{plot_metric}'])
        plt.legend([plot_metric, f'val_{plot_metric}'])
        plt.xlabel("Epoch")
        plt.ylabel(plot_metric)


def confusion_matrix(y_truth, y_pred, cats, cmap=None, normalize=True, title=None):
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

    return cm


def preds_pie(y_preds, cats, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)

    axis.pie(np.sum(y_preds, axis=0), labels=cats)
    axis.axis("equal")


def preds_histogram(y_preds, cats, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        axis.hist(y_preds[:, i], bins=100, alpha=0.8, range=[0, 1], density=True)

    if not axis.get_legend():
        axis.legend(cats)
        axis.set_xlabel("Network Confidence")
        axis.set_ylabel("Normalized Frequency")
        axis.set_ylim([0, 100])


def ROC_one_vs_all(y_preds, y_truth, cats, title=None, axis=None):
    fpr = {}
    tpr = {}
    tholds = {}
    roc_auc = {}

    if np.array(y_preds).ndim != 2:
        y_preds = ind_to_onehot(y_preds)
    if np.array(y_truth).ndim != 2:
        y_truth = ind_to_onehot(y_truth)

    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        fpr[cat], tpr[cat], tholds[cat] = (metrics.roc_curve(
            y_truth[:, i].astype(int), y_preds[:, i]))
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


def PR_one_vs_all(y_preds, y_truth, cats, title=None, axis=None):
    prec = {}
    recall = {}
    tholds = {}
    pr_auc = {}

    if np.array(y_preds).ndim != 2:
        y_preds = ind_to_onehot(y_preds)
    if np.array(y_truth).ndim != 2:
        y_truth = ind_to_onehot(y_truth)

    if not axis:
        fig, axis = plt.subplots(1, 1)

    for i, cat in enumerate(cats):
        prec[cat], recall[cat], tholds[cat] = (metrics.precision_recall_curve(
            y_truth[:, i].astype(int), y_preds[:, i]))
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


def test_classifier(model, x_true, y_true, cats, y_pred=None):
    """Tests a classifier"""

    if "sklearn" in str(type(model)):
        y_pred = model.predict_proba(x_true)
    elif "tensorflow" in str(type(model)):
        if y_pred is None:
            y_pred = model.predict(x_true)
    else:
        raise ValueError("Model must be from sklearn or tensorflow")

    y_pred_arg = onehot_to_ind(y_pred)
    y_true_arg = onehot_to_ind(y_true)

    performance = {}
    performance_funcs = [metrics.accuracy_score, metrics.balanced_accuracy_score,
                 metrics.confusion_matrix, metrics.hamming_loss]

    for func in performance_funcs:
        performance[func.__name__] = func(y_pred=y_pred_arg, y_true=y_true_arg)

    ROC_one_vs_all(y_preds=y_pred, y_truth=y_true, cats=cats)
    PR_one_vs_all(y_preds=y_pred, y_truth=y_true, cats=cats)
    confusion_matrix(y_pred=y_pred_arg, y_truth=y_true, cats=cats)

    return performance