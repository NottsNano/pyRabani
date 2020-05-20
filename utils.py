import numpy as np
from matplotlib import pyplot as plt


def make_pd_nans_identical(df, replacement_value=None):
    return df.replace({np.nan: replacement_value})


def adding_noise_test(img, model, cats, noise_steps, perc_noise, perc_std, savedir=None):
    """Progressively add noise to an image and classifying it"""

    from Rabani_Generator.plot_rabani import show_image
    from CNN.get_stats import all_preds_histogram
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


def single_prediction_with_noise(img, cnn_model, perc_noise, perc_std):
    from CNN.CNN_training import h5RabaniDataGenerator
    from CNN.CNN_prediction import ImageClassifier

    img = img.copy()  # Do this because of immutability!
    img = h5RabaniDataGenerator.speckle_noise(img, perc_noise, perc_std)[0, :, :, 0]
    img_classifier = ImageClassifier(img, cnn_model)
    img_classifier.cnn_classify()

    return img_classifier
