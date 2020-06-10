import glob

import pandas as pd
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from Filters.screening import FileFilter
from CNN.utils import make_pd_nans_identical

IMAGE_DIR = "/media/mltest1/Dat Storage/Manu AFM CD Box"# "/home/mltest1/tmp/pycharm_project_883/Data/Classification_Performance_Images/Good_Images"
CNN_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-06-10--12-22/model.h5"
DENOISER_DIR = "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-05-29--14-07/model.h5"
ASSESS_EULER = True
SEARCH_RECURSIVE = True

# Load models
cnn_model = load_model(CNN_DIR)
denoiser_model = load_model(DENOISER_DIR)

df_summary = pd.DataFrame(
    columns=["File Path", "Resolution", "Size (m)", "Fail Reasons",
             "CNN Classification", "CNN Mean", "CNN std",
             "Euler Classification", "Euler Mean", "Euler std"])

# Filter every ibw file in the directory, and build up a dataframe
all_files = [f for f in glob.glob(f"{IMAGE_DIR}/**/*.ibw", recursive=SEARCH_RECURSIVE)]
t = tqdm(total=len(all_files), smoothing=True)
for i, file in enumerate(all_files):
    t.set_description(f"...{file[-25:]}")

    filterer = FileFilter()
    filterer.assess_file(filepath=file, category_model=cnn_model, assess_euler=ASSESS_EULER, savedir="/home/mltest1/tmp/pycharm_project_883/Data/Classification_Performance_Images/Filtered_All_NewModel/Filtered")

    df_summary.loc[i, ["File Path"]] = [file]
    df_summary.loc[i, ["Resolution"]] = [filterer.image_res]
    df_summary.loc[i, ["Size (m)"]] = [filterer.image_size]
    df_summary.loc[i, ["Fail Reasons"]] = [filterer.fail_reasons]

    if filterer.image_classifier:
        if cnn_model:
            df_summary.loc[i, ["CNN Classification"]] = [filterer.CNN_classification]
            df_summary.loc[i, ["CNN Mean"]] = [filterer.image_classifier.cnn_preds.mean(axis=0)]
            df_summary.loc[i, ["CNN std"]] = [filterer.image_classifier.cnn_preds.std(axis=0)]

        if ASSESS_EULER:
            df_summary.loc[i, ["Euler Classification"]] = [filterer.euler_classification]
            df_summary.loc[i, ["Euler Mean"]] = [filterer.image_classifier.euler_preds.mean(axis=0)]
            df_summary.loc[i, ["Euler std"]] = [filterer.image_classifier.euler_preds.std(axis=0)]

    del filterer    # Just to be safe!
    t.update(1)

df_summary = make_pd_nans_identical(df_summary)
df_summary.to_csv("/home/mltest1/tmp/pycharm_project_883/Data/Classification_Performance_Images/Filtered_All_NewModel/classification.csv", index=False)
