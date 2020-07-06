import numpy as np
import pandas as pd
from skimage.measure import regionprops, label
from skimage.morphology import closing, square
from tqdm import tqdm

from CNN.CNN_training import h5RabaniDataGenerator

datadir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/NewTest"
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
    x_inv = np.abs(1 - x)

    # For each image/inverse image
    for j in range(batch_size):
        # Segment objects
        label_img = label(closing(x[j, :, :, 0], square(3)))
        label_img_inv = label(closing(x_inv[j, :, :, 0], square(3)))

        # Get stats
        H0 = label_img.max()
        H1 = label_img_inv.max()

        tot_perimeter = 0
        if H0 > H1:
            average_particle_size = np.sum(label_img > 0)
        else:
            average_particle_size = np.sum(label_img_inv > 0)

        tot_area = np.sum(label_img > 0)

        for region, region_inv in zip(regionprops(label_img), regionprops(label_img_inv)):
            tot_perimeter += region["perimeter"] + region_inv["perimeter"]

        # Make stats size invariant
        SIA = average_particle_size / np.size(label_img)
        SIP = tot_perimeter / (H0 * np.sqrt(average_particle_size))
        SIH0 = H0 * average_particle_size
        SIH1 = H1 * average_particle_size

        # Place in dframe
        num_img = (i * batch_size) + j
        df_summary.loc[num_img, "label"] = y_cats[y[j].argmax()]
        df_summary.loc[num_img, "SIA"] = SIA
        df_summary.loc[num_img, "SIP"] = SIP
        df_summary.loc[num_img, "SIH0"] = SIH0
        df_summary.loc[num_img, "SIH1"] = SIH1

df_summary.to_csv("/home/mltest1/tmp/pycharm_project_883/Data/Classical_Stats/simulated_test.csv")