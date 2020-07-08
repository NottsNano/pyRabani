from ast import literal_eval

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from Analysis.model_stats import confusion_matrix, ROC_one_vs_all, PR_one_vs_all
from Filters.screening import FileFilter
from Analysis.plot_rabani import show_image


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
    # df_manual["Labrynthine"] = df_manual[["Labrynthine", "worm"]].max(axis=1)
    # df_manual = df_manual.drop(["worm"], axis=1)

    df_automated[["CNN Mean", "CNN std", "Euler Mean", "Euler std"]] = _restore_stored_list(
        df_automated[["CNN Mean", "CNN std", "Euler Mean", "Euler std"]])

    return df_manual, df_automated


def _restore_stored_list(df):
    """Allows pandas cells saved as lists to be acted on as lists instead of strings"""
    df = df.copy()  # Avoid mutability issues

    for col in list(df.columns):
        df[col] = df[col].str.replace("\s{1,10}", ",")
        df[col] = df[col].str.replace(",]", "]")

        list_len = int(df[col].str.count(",").max()) + 1
        strng = str([0.] * list_len)
        df[col] = df[col].replace(np.nan, strng)

        df[col] = df[col].apply(literal_eval)  # pd.eval only works with 100 vals - recommended workaround

    return df


def filter_multi_label(df_merged):
    filtered = df_merged.drop(
        df_merged[df_merged[["cell", "Labrynthine", "particle", "porous", "finger"]].sum(axis=1) != 1].index)
    return filtered.drop(filtered[(filtered[["finger", "porous"]] > 0).any(axis=1)].index)


def filter_automated_failing(df_merged, reason):
    if reason == "any":
        return df_merged.drop(df_merged[pd.notnull(df_merged["Fail Reasons"])].index)
    elif reason == "euler":
        return df_merged.drop(df_merged[df_merged["Fail Reasons"].str.contains("euler", case=False) & pd.notnull(
            df_merged["Fail Reasons"])].index)
    elif reason == "cnn":
        return df_merged.drop(df_merged[df_merged["Fail Reasons"].str.contains("cnn", case=False) & pd.notnull(
            df_merged["Fail Reasons"])].index)
    elif reason == "preprocessing":
        return df_merged.drop(df_merged[df_merged["Fail Reasons"].str.contains(
            "binaris|flatten|noisy|incomplete|corrupt|unexpected", case=False) & pd.notnull(
            df_merged["Fail Reasons"])].index)
    else:
        raise ValueError("reason must be one of ['any', 'euler', 'cnn', 'preprocessing']")


def _classifications_to_matrix(df_merged):
    """Converts the classification columns of the dframes to a matrix"""
    auto_classes_cnn = np.array(df_merged["CNN Mean"].tolist())[:, 2:]
    auto_classes_euler = np.array(df_merged["Euler Mean"].tolist())[:, 2:-1]
    manual_classes_truth = np.array(df_merged[["cell", "Labrynthine", "particle"]])

    return auto_classes_cnn, auto_classes_euler, manual_classes_truth


def _onehot_encode(arr):
    return np.argmax(arr, axis=1)


def stats_filtering(df_multiclass):
    df_preproc_fail = filter_automated_failing(df_multiclass, reason="preprocessing")
    print((f"{len(df_multiclass) - len(df_preproc_fail)}/{len(df_multiclass)} "
           "multi-class human classifications could not be preprocessed"))

    df_euler_fail = filter_automated_failing(df_multiclass, reason="euler")
    print(f"{len(df_multiclass) - len(df_euler_fail)}/{len(df_preproc_fail)} "
          "successfully preprocessed multi-class human classifications could not be confidently Euler classified")

    df_cnn_fail = filter_automated_failing(df_multiclass, reason="cnn")
    print(f"{len(df_multiclass) - len(df_cnn_fail)}/{len(df_preproc_fail)} "
          "successfully preprocessed multi-class human classifications could not be confidently CNN classified")

    df_filter_fail = filter_automated_failing(df_multiclass, reason="any")
    print(f"{len(df_multiclass) - len(df_filter_fail)}/{len(df_multiclass)} "
          "total multi-class human classifications could not be preprocessed/classified")


def compare_classifications(df):
    ngridpts = int(np.sqrt(len(df)))
    fig, axs = plt.subplots(ngridpts, ngridpts)
    axs = axs.reshape((-1,))

    for i, file in enumerate(df.iterrows()):
        filterer = FileFilter()
        filterer.assess_file(file[1]["File Path"])

        show_image(filterer.binarized_data, axis=axs[i],
                   title=f"Steff = {file[1]['Regime']} | CNN = {file[1]['CNN Classification']}")


if __name__ == '__main__':
    AUTOMATED_ASSESSMENT_FILE = "/home/mltest1/tmp/pycharm_project_883/Data/ClassificationWithoutFullDenoising.csv"
    MANUAL_ASSESSMENT_FILE = "/home/mltest1/tmp/pycharm_project_883/Data/steff_assessment.csv"
    CATS = ["Cellular", "Labyrinthine", "Islands"]

    dframe_manual, dframe_automated = load_and_parse(MANUAL_ASSESSMENT_FILE, AUTOMATED_ASSESSMENT_FILE)

    # Filter results
    dframe_merged = dframe_manual.join(dframe_automated)
    dframe_multilabel_filtered = filter_multi_label(dframe_merged)
    dframe_failing_filtered = filter_automated_failing(dframe_multilabel_filtered, reason="any")

    # Get stats
    pred_cnn, pred_euler, truth = _classifications_to_matrix(dframe_failing_filtered)
    stats_filtering(dframe_multilabel_filtered)

    confusion_matrix(y_truth=_onehot_encode(truth), y_pred=_onehot_encode(pred_cnn),
                     cats=CATS, title="CNN Preds")
    confusion_matrix(y_truth=_onehot_encode(truth), y_pred=_onehot_encode(pred_euler),
                     cats=CATS, title="Euler Preds")

    ROC_one_vs_all(pred_cnn, truth, cats=CATS, title="CNN ROC")
    ROC_one_vs_all(pred_euler, truth, cats=CATS, title="Euler ROC")

    PR_one_vs_all(pred_cnn, truth, cats=CATS, title="CNN ROC")
    PR_one_vs_all(pred_euler, truth, cats=CATS, title="Euler ROC")

    classification_report(y_pred=_onehot_encode(pred_cnn), y_true=_onehot_encode(truth))
    classification_report(y_pred=_onehot_encode(pred_euler), y_true=_onehot_encode(truth))
