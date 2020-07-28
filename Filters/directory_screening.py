import glob

import pandas as pd
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from Filters.screening import FileFilter
from Models.train_regression import load_sklearn_model
from Models.utils import make_pd_nans_identical

IMAGE_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Classification_Performance_Images/Good_Images"#"/home/mltest1/tmp/pycharm_project_883/Data/Steff_Images/Raw"#"/media/mltest1/Dat Storage/Manu AFM CD Box" #
CNN_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-06-15--12-18/model.h5"
DENOISER_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-05-29--14-07/model.h5"
MINKOWSKI_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-07-28--11-15/model.p"
OUTPUT_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Classification_Performance_Images/Final"
ASSESS_EULER = False
SEARCH_RECURSIVE = True

# Load models
cnn_model = None#load_model(CNN_DIR)
denoiser_model = load_model(DENOISER_DIR)
sklearn_model = load_sklearn_model(MINKOWSKI_DIR)

df_summary = pd.DataFrame(
    columns=["File Path", "Resolution", "Size (m)", "Fail Reasons",
             "CNN Classification", "CNN Mean", "CNN std",
             "Euler Classification", "Euler Mean", "Euler std",
             "Stats Regression Classification", "Stats Regression Mean", "Stats Regression std",
             "Manual Classification", "SIA", "SIP", "SIE"])

# Filter every ibw file in the directory, and build up a dataframe
all_files = [f for f in glob.glob(f"{IMAGE_DIR}/**/*.ibw", recursive=SEARCH_RECURSIVE)]

t = tqdm(total=len(all_files), smoothing=True)
for i, file in enumerate(all_files):
    t.set_description(f"...{file[-25:]}")

    filterer = FileFilter()
    filterer.assess_file(filepath=file, threshold_method="multiotsu",
                         category_model=None, denoising_model=denoiser_model, minkowski_model=sklearn_model,
                         assess_euler=ASSESS_EULER, nbins=1000)#, savedir=f"{OUTPUT_DIR}/Filtered")

    df_summary.loc[i, ["File Path"]] = [file]
    df_summary.loc[i, ["Resolution"]] = [filterer.image_res]
    df_summary.loc[i, ["Size (m)"]] = [filterer.image_size]
    df_summary.loc[i, ["Fail Reasons"]] = [filterer.fail_reasons]
    df_summary.loc[i, ["Manual Classification"]] = [file.split("/")[-2]]

    if filterer.CNN_classification:
        df_summary.loc[i, ["CNN Classification"]] = [filterer.CNN_classification]
        df_summary.loc[i, ["CNN Mean"]] = [filterer.image_classifier.cnn_preds.mean(axis=0)]
        df_summary.loc[i, ["CNN std"]] = [filterer.image_classifier.cnn_preds.std(axis=0)]

    if filterer.euler_classification:
        df_summary.loc[i, ["Euler Classification"]] = [filterer.euler_classification]
        df_summary.loc[i, ["Euler Mean"]] = [filterer.image_classifier.euler_preds.mean(axis=0)]
        df_summary.loc[i, ["Euler std"]] = [filterer.image_classifier.euler_preds.std(axis=0)]

    if filterer.minkowski_classification:
        df_summary.loc[i, ["Stats Regression Classification"]] = [filterer.minkowski_classification]
        df_summary.loc[i, ["Stats Regression Mean"]] = [filterer.image_classifier.minkowski_preds.mean(axis=0)]
        df_summary.loc[i, ["Stats Regression std"]] = [filterer.image_classifier.minkowski_preds.std(axis=0)]

    del filterer  # Just to be safe!
    t.update(1)

df_summary = make_pd_nans_identical(df_summary)
df_summary.to_csv(f"{OUTPUT_DIR}/good_classifications_minkowski_newstats.csv", index=False)
