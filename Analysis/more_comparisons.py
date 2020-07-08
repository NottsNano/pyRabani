import pandas as pd
import numpy as np
from Analysis.model_stats import confusion_matrix

csv_path = "/home/mltest1/tmp/pycharm_project_883/Data/Steff_Images/Raw/cnn_classifications.csv"
dframe = pd.read_csv(csv_path)

cats = ["cellular", "labyrinth", "island", "fail"]

truth = [cats.index(i) for i in (list(dframe["Manual Classification"]))]
preds = np.zeros(len(truth))

for i, row in dframe.iterrows():
    if type(row["Fail Reasons"]) is str:
        preds[i] = 3
    else:
        preds[i] = cats.index(row["CNN Classification"])

confusion_matrix(truth, preds, cats)