import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from Analysis.image_stats import calculate_normalised_stats
from Analysis.model_stats import test_classifier
from Analysis.plot_rabani import cmap_rabani
from Models.h5_iterator import h5RabaniDataGenerator
from Models.train_CNN import get_model_storage_path
from Models.utils import ensure_dframe_is_pandas


def make_dataset(rabani_dir, output_dir=None, batch_size=128, save_ims=False):
    """
    Make a dataset that can be used in sklearn

    Parameters
    ----------
    rabani_dir: str
        Directory of a folder containing h5 files made by Rabani_Simulation.rabani_generator
    output_dir: str, optional
        Output directory to write csv/images to
    batch_size: int, optional
        Number of h5 files to hold in memory at once. Default 128.
    save_ims: bool, optional
        If the images should be saved to the `output_dir' directory. Default False
    """

    y_params = ["kT", "mu"]
    y_cats = ["liquid", "hole", "cellular", "labyrinth", "island"]

    dframe = pd.DataFrame(columns=["label", "SIA", "SIP", "SIH0", "SIH1"])

    img_generator = h5RabaniDataGenerator(rabani_dir, network_type="classifier", batch_size=batch_size, is_train=False,
                                          imsize=200, force_binarisation=True,
                                          output_parameters_list=y_params, output_categories_list=y_cats)
    img_generator.is_validation_set = True

    # Get each batch of images
    for i in tqdm(range(img_generator.__len__())):
        x, y = img_generator.__getitem__(None)

        # For each image/inverse image
        for j in range(batch_size):
            SIA, SIP, SIH0, SIH1 = calculate_normalised_stats(x[j, :, :, 0])

            # Place in dframe
            num_img = (i * batch_size) + j
            dframe.loc[num_img, "label"] = y_cats[y[j].argmax()]
            dframe.loc[num_img, "SIA"] = SIA
            dframe.loc[num_img, "SIP"] = SIP
            dframe.loc[num_img, "SIH0"] = SIH0
            dframe.loc[num_img, "SIH1"] = SIH1

            if save_ims:
                img = closing(x[j, :, :, 0], square(3))
                plt.imsave(f"{output_dir}/{num_img}.png", img, cmap=cmap_rabani)

    if output_dir:
        dframe.to_csv(f"{output_dir}/{rabani_dir.split('/')[-1]}.csv")

    return dframe


def convert_dframe_to_sklearn(dframe, data_column_headers, cats, num_items=None):
    """Converts a dframe/path to a dframe to a sklearn compatible format

    Parameters
    ----------
    dframe: str or object of type pd.DataFrame
        Either a dataframe, or a directory of a csv file containing a dataframe
    data_column_headers: list
        List of column names denoting which columns contain input data
    cats: list of str
        List of names of each structure type
    num_items: int, optional
        Don't return the entire dataframe, but a random selection of number num_items. Default None (return all)
    """

    dframe = ensure_dframe_is_pandas(dframe)

    all_cats = ["liquid", "hole", "cellular", "labyrinth", "island"]
    cats_to_drop = list(set(all_cats) - set(cats))
    dframe = dframe[~dframe['label'].isin(cats_to_drop)]

    if num_items:
        dframe.sample(num_items)

    x = np.array(dframe[data_column_headers])
    y = [cats.index(struct) for struct in dframe["label"]]

    # scaler = MinMaxScaler()
    # x = scaler.fit_transform(x)

    return x, y


def train_classifier(x_train, y_train, **kwargs):
    """Trains an sklearn classifier"""
    model = LogisticRegression(**kwargs)
    model.fit(x_train, y_train)

    return model


def load_sklearn_model(model_dir):
    """Loads an sklearn classifier"""
    return joblib.load(model_dir)


def save_classifier(root_dir, model):
    """Saves classifier"""
    path = get_model_storage_path(root_dir)[:-3] + ".p"
    joblib.dump(model, open(path, "wb"))


if __name__ == '__main__':
    # df_test = make_dataset("/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/NewTest",
    #              output_dir="/home/mltest1/tmp/pycharm_project_883/Data/Classical_Stats/Dataset", save_ims=True)
    # df_train = make_dataset("/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/TrainFinal",
    #              output_dir="/home/mltest1/tmp/pycharm_project_883/Data/Classical_Stats/simulated_train.csv")
    cats = ["hole", "cellular", "labyrinth", "island"]

    x_train, y_train = convert_dframe_to_sklearn(
        dframe="/home/mltest1/tmp/pycharm_project_883/Data/Classical_Stats/simulated_train.csv",
        data_column_headers=["SIA", "SIP", "SIH0", "SIH0"],
        cats=cats, num_items=2000)
    x_test, y_test = convert_dframe_to_sklearn(
        dframe="/home/mltest1/tmp/pycharm_project_883/Data/Classical_Stats/simulated_newtest.csv",
        data_column_headers=["SIA", "SIP", "SIH0", "SIH0"],
        cats=cats)

    model = train_classifier(x_train, y_train, max_iter=1000)
    performance = test_classifier(model, x_test, y_test, cats=cats)

    save_classifier("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks", model)
