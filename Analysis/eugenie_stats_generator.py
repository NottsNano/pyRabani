import pandas as pd
from tqdm import tqdm

from Analysis.get_stats import calculate_normalised_stats
from Classify.CNN_training import h5RabaniDataGenerator

datadir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/TrainFinal"
batch_size = 128

y_params = ["kT", "mu"]
y_cats = ["liquid", "hole", "cellular", "labyrinth", "island"]

df_summary = pd.DataFrame(columns=["label", "SIA", "SIP", "SIH0", "SIH1"])

img_generator = h5RabaniDataGenerator(datadir, network_type="classifier", batch_size=batch_size, is_train=False,
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
        df_summary.loc[num_img, "label"] = y_cats[y[j].argmax()]
        df_summary.loc[num_img, "SIA"] = SIA
        df_summary.loc[num_img, "SIP"] = SIP
        df_summary.loc[num_img, "SIH0"] = SIH0
        df_summary.loc[num_img, "SIH1"] = SIH1

df_summary.to_csv("/home/mltest1/tmp/pycharm_project_883/Data/Classical_Stats/simulated_train.csv")
