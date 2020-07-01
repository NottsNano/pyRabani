import glob

import pandas as pd
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from Filters.screening import FileFilter
from CNN.utils import make_pd_nans_identical

IMAGE_DIR = "../Data/Classification_Performance_Images/Bad_Images"#"/media/mltest1/Dat Storage/Manu AFM CD Box"#"/home/mltest1/tmp/pycharm_project_883/Data/Classification_Performance_Images/Good_Images"#
CNN_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-06-15--12-18/model.h5"
DENOISER_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-05-29--14-07/model.h5"
OUTPUT_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Steff_Images/Unet"#"/home/mltest1/tmp/pycharm_project_883/Data/Classification_Performance_Images/Filtered_All_TweakedModelDenoising"
ASSESS_EULER = False
SEARCH_RECURSIVE = True

# Load models
cnn_model = None#load_model(CNN_DIR)
denoiser_model = None#load_model(DENOISER_DIR)

df_summary = pd.DataFrame(
    columns=["File Path", "Resolution", "Size (m)", "Fail Reasons",
             "CNN Classification", "CNN Mean", "CNN std",
             "Euler Classification", "Euler Mean", "Euler std", "Manual Classification"])

# Filter every ibw file in the directory, and build up a dataframe
all_files = [f for f in glob.glob(f"{IMAGE_DIR}/**/*.ibw", recursive=SEARCH_RECURSIVE)]

t = tqdm(total=len(all_files), smoothing=True)
for i, file in enumerate(all_files):
    t.set_description(f"...{file[-25:]}")

    filterer = FileFilter()
    filterer.assess_file(filepath=file, threshold_method="otsu", category_model=cnn_model, denoising_model=denoiser_model,
                         assess_euler=ASSESS_EULER, nbins=1000)#, savedir=f"{OUTPUT_DIR}/Filtered")

    df_summary.loc[i, ["File Path"]] = [file]
    df_summary.loc[i, ["Resolution"]] = [filterer.image_res]
    df_summary.loc[i, ["Size (m)"]] = [filterer.image_size]
    df_summary.loc[i, ["Fail Reasons"]] = [filterer.fail_reasons]
    # df_summary.loc[i, ["Manual Classification"]] = [file.split("\\")[-2]]

    if filterer.CNN_classification:
        df_summary.loc[i, ["CNN Classification"]] = [filterer.CNN_classification]
        df_summary.loc[i, ["CNN Mean"]] = [filterer.image_classifier.cnn_preds.mean(axis=0)]
        df_summary.loc[i, ["CNN std"]] = [filterer.image_classifier.cnn_preds.std(axis=0)]

    if filterer.euler_classification:
        df_summary.loc[i, ["Euler Classification"]] = [filterer.euler_classification]
        df_summary.loc[i, ["Euler Mean"]] = [filterer.image_classifier.euler_preds.mean(axis=0)]
        df_summary.loc[i, ["Euler std"]] = [filterer.image_classifier.euler_preds.std(axis=0)]

    del filterer    # Just to be safe!
    t.update(1)

df_summary = make_pd_nans_identical(df_summary)
# df_summary.to_csv(f"{OUTPUT_DIR}/classifications_euler.csv", index=False)
