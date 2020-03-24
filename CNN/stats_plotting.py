import itertools

import numpy as np
from matplotlib import pyplot as plt


def plot_model_history(model):
    for plot_metric in model.metrics_names:
        plt.figure()
        plt.plot(model.history.history[plot_metric])
        plt.plot(model.history.history[f'val_{plot_metric}'])
        plt.legend([plot_metric, f'val_{plot_metric}'])
        plt.xlabel("Epoch")
        plt.ylabel(plot_metric)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

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


def all_preds_histogram(preds, cats):
    plt.figure()

    for i, cat in enumerate(cats):
        plt.hist(preds[:, i], bins=100, alpha=0.8, range=[0, 1], density=True)
    plt.legend(cats)
    plt.xlabel("Network Confidence")
    plt.ylabel("Normalized Frequency")
