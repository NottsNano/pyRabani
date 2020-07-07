import pandas as pd

from Models.test_model import confusion_matrix

csv_path = "/home/mltest1/tmp/pycharm_project_883/Data/Steff_Images/Unet/classifications_euler.csv"
dframe = pd.read_csv(csv_path)

cats = ["cellular", "labyrinth", "island", "fail"]

truth = [cats.index(i) for i in (list(dframe["Manual Classification"]))]
preds = [cats.index(i) if i in cats else len(cats)-1 for i in list(dframe["Euler Classification"])]

confusion_matrix(truth, preds, cats)