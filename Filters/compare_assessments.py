from ast import literal_eval

import numpy as np
import pandas as pd


def load_and_parse(manual_file, automated_file):
    # Load the two dataframes
    df_manual = pd.read_csv(manual_file)
    df_automated = pd.read_csv(automated_file)

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

    # Attempt to fill in category based on regime text (N.B. OVERWRITES manual 1/0s in classification cols.)
    df_manual["finger"] = df_manual["Regime"].str.contains("finger", case=False).astype(int)
    df_manual["cell"] = df_manual["Regime"].str.contains("cell", case=False).astype(int)
    df_manual["particle"] = df_manual["Regime"].str.contains("particle", case=False).astype(int)
    df_manual["Labrynthine"] = df_manual["Regime"].str.contains("laby", case=False).astype(int)
    df_manual["worm"] = df_manual["Regime"].str.contains("worm", case=False).astype(int)

    # Combine 'labyrinthine' and 'worm' columns
    df_manual["Labrynthine"] = df_manual[["Labrynthine", "worm"]].max(axis=1)
    df_manual = df_manual.drop(["worm"], axis=1)

    df_automated[["CNN Mean", "CNN std", "Euler Mean", "Euler std"]] = _restore_stored_list(df_automated[["CNN Mean", "CNN std", "Euler Mean", "Euler std"]])

    return df_manual, df_automated


def _restore_stored_list(df):
    """Allows pandas cells saved as lists to be acted on as lists instead of strings"""
    df = df.copy()      # Avoid mutability issues

    for col in list(df.columns):
        df[col] = df[col].str.replace("\s{1,10}", ",")
        df[col] = df[col].str.replace(",]", "]")

        list_len = int(df[col].str.count(",").max())+1
        strng = str([0.] * list_len)
        df[col] = df[col].replace(np.nan, strng)

        df[col] = df[col].apply(literal_eval)      # pd.eval only works with 100 vals - recommended workaround

    return df


if __name__ == '__main__':
    AUTOMATED_ASSESSMENT_FILE = "/home/mltest1/tmp/pycharm_project_883/Data/ClassificationWithFullDenoising.csv"
    MANUAL_ASSESSMENT_FILE = "/home/mltest1/tmp/pycharm_project_883/Data/steff_assessment.csv"
    CATS = ["Cellular", "Labyrinthine", "Islands"]

    dframe_manual, dframe_automated = load_and_parse(MANUAL_ASSESSMENT_FILE, AUTOMATED_ASSESSMENT_FILE)

    # Filter results
    # Remove manual multi-label classifications
    dframe_merged = dframe_manual.join(dframe_automated)


    # Remove automated filtered out

    # Turn classifications into matrices

    automated_classifications_cnn = np.array(dframe_automated["CNN Mean"].tolist())[:, 3:]
    automated_classifications_euler = np.array(dframe_automated["Euler Mean"].tolist())[:, 3:]
    manual_classifications_true = np.array(dframe_manual[["cell", "Labrynthine", "particle"]])

    # Get classification report and graphs