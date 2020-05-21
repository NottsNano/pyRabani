import pandas as pd
import numpy as np

AUTOMATED_ASSESSMENT_FILE = "/home/mltest1/tmp/pycharm_project_883/Data/ClassificationWithFullDenoising.csv"
MANUAL_ASSESSMENT_FILE = "/home/mltest1/tmp/pycharm_project_883/Data/steff_assessment.csv"

# Load the two dataframes
df_manual = pd.read_csv(MANUAL_ASSESSMENT_FILE)
df_automated = pd.read_csv(AUTOMATED_ASSESSMENT_FILE)

# Remove the file extension and the "HtTM0" marker from the shortened manual files
df_manual["File"] = df_manual["File"].str.replace("HtTM0.tif", "")
df_automated["File"] = df_automated["File Path"].str.extract('.*\/(.*)\..*')

# Drop duplicates, ambiguous files, and files not present in both assessments
df_automated = df_automated[df_automated["File"].str.match('|'.join(df_manual["File"]))].drop_duplicates("File")
df_manual = df_manual[df_manual["File"].str.match('|'.join(df_automated["File"]))].drop_duplicates("File")
df_automated = df_automated[~df_automated["File"].str.contains("image|file|test", case=False)]
df_manual = df_manual[~df_manual["File"].str.contains("image|file|test", case=False)]

# Reorder rows to be identical
df_manual = df_manual.set_index("File").sort_index()
df_automated = df_automated.set_index("File").sort_index()

assert df_automated.index.equals(df_manual.index), "Manual and automated files must be identical"

# Attempt to fill in category based on regime text (N.B. OVERWRITES manual 1/0s)
df_manual["finger"] = df_manual["Regime"].str.contains("finger", case=False).astype(int)
df_manual["cell"] = df_manual["Regime"].str.contains("cell", case=False).astype(int)
df_manual["particle"] = df_manual["Regime"].str.contains("particle", case=False).astype(int)
df_manual["Labrynthine"] = df_manual["Regime"].str.contains("laby", case=False).astype(int)
df_manual["worm"] = df_manual["Regime"].str.contains("worm", case=False).astype(int)

# Combine labyrinthine and worm columns
df_manual["labyrinthine"] = df_manual[["Labrynthine", "worm"]].max(axis=1)
df_manual = df_manual.drop(["Labrynthine", "worm"], axis=1)


