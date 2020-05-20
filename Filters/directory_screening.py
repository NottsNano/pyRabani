import glob
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from Filters.screening import FileFilter

# image_dir = "/home/mltest1/tmp/pycharm_project_883/Images/Parsed Dewetting 2020 for ML/thres_img"
IMAGE_DIR = "/home/mltest1/tmp/pycharm_project_883/Images/Parsed Dewetting 2020 for ML/thres_img/tp"
CNN_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5"
DENOISER_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-05-19--19-36/model.h5"
SEARCH_RECURSIVE = True

# Setup
cnn_model = load_model(CNN_DIR)
denoiser_model = load_model(DENOISER_DIR)

df_summary = pd.DataFrame(
    columns=["File Path", "Resolution", "Fail Reasons",
             "CNN Classification", "CNN Mean", "CNN std",
             "Euler Classification", "Euler Mean", "Euler std"])

# Filter every ibw file in the directory, and build up a dataframe
all_files = [f for f in glob.glob(f"{IMAGE_DIR}/**/*.ibw", recursive=SEARCH_RECURSIVE)]
t = tqdm(total=len(all_files), smoothing=True)
for i, file in enumerate(all_files):
    t.update(1)
    t.set_description(file[len(IMAGE_DIR):])

    filterer = FileFilter()
    filterer.assess_file(filepath=file, cnn_model=cnn_model, denoising_model=denoiser_model, plot=False,
                       savedir="/home/mltest1/tmp/pycharm_project_883/Images/testfilter")

    df_summary.loc[i, ["File Path"]] = [file]
    df_summary.loc[i, ["Resolution"]] = [filterer.image_res]
    df_summary.loc[i, ["Fail Reasons"]] = [filterer.fail_reasons]
    df_summary.loc[i, ["CNN Classification"]] = [filterer.CNN_classification]
    df_summary.loc[i, ["Euler Classification"]] = [filterer.euler_classification]

    if filterer.image_classifier:
        df_summary.loc[i, ["CNN Mean"]] = [filterer.image_classifier.cnn_preds.mean(axis=0)]
        df_summary.loc[i, ["CNN std"]] = [filterer.image_classifier.cnn_preds.std(axis=0)]

        df_summary.loc[i, ["Euler Mean"]] = [filterer.image_classifier.euler_preds.mean(axis=0)]
        df_summary.loc[i, ["Euler std"]] = [filterer.image_classifier.euler_preds.std(axis=0)]

    del filterer

df_summary = df_summary.replace({np.nan: None})

# To load the list columns properly, do np.array(dframe[column].tolist())
df_summary.to_csv("/home/mltest1/tmp/pycharm_project_883/Images/BigParse2.csv", index=False)
