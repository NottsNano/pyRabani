import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from skimage import measure



root_dir="/home/mltest1/tmp/pycharm_project_883/Images/2020-02-14/21-16"
files = os.listdir(root_dir)

# Find image details
axis_res = int(np.sqrt(len(files)))
kT_range_all = np.zeros((len(files),))
mu_range_all = np.zeros((len(files),))
m_all = np.zeros((len(files),))
for i, file in enumerate(files):
    img_file = h5py.File(f"{root_dir}/{file}", "r")
    kT_range_all[i] = img_file.attrs["kT"]
    mu_range_all[i] = img_file.attrs["mu"]
    m_all[i] = img_file.attrs["num_mc_steps"]
    img_res = len(img_file["image"])

kT_range = [np.min(kT_range_all), np.max(kT_range_all)]
mu_range = [np.min(mu_range_all), np.max(mu_range_all)]

# Place each image in an array
big_img_arr = np.zeros((img_res * axis_res, img_res * axis_res))
kT_vals = np.linspace(kT_range[0], kT_range[1], axis_res)
mu_vals = np.linspace(mu_range[0], mu_range[1], axis_res)
eulers = np.zeros((axis_res, axis_res))
for i, file in enumerate(files):
    img_file = h5py.File(f"{root_dir}/{file}", "r")

    kT_ind = np.searchsorted(kT_vals, img_file.attrs["kT"])
    mu_ind = np.searchsorted(mu_vals, img_file.attrs["mu"])
    big_img_arr[(kT_ind * 128):((kT_ind + 1) * 128), (mu_ind * 128):((mu_ind + 1) * 128)] = np.flipud(
        img_file["image"])

    eulers[kT_ind, mu_ind] = measure.regionprops((img_file["image"][()]!=0).astype(int)+1)[0]["euler_number"]/np.sum(img_file["image"][()]==2)


# Plot
cmap = colors.ListedColormap(["black", "white", "orange"])
boundaries = [0, 0.5, 1]
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

fig1, ax1 = plt.subplots()
plt.imshow(big_img_arr, cmap=cmap, origin="lower")

def nice_axis(axis, res, grid=True):
    plt.xticks(rotation=90)
    plt.xticks(np.arange(len(mu_vals)) * res + res/2, [f"{mu_val:.2f}" for mu_val in mu_vals])
    plt.yticks(np.arange(len(kT_vals)) * res + res/2, [f"{kT_val:.2f}" for kT_val in kT_vals])

    axis.set_xticks([x * res for x in range(axis_res)], minor=True)
    axis.set_yticks([y * res for y in range(axis_res)], minor=True)
    # TODO: Don't write EVERY box as unit - be more intelligent!
    axis.set_xlabel("mu")
    axis.set_ylabel("kT")

    if grid:
        plt.grid(which="minor", ls="-", lw=2, color="r")

nice_axis(ax1, img_res)

fig2, ax2 = plt.subplots()
plt.imshow(eulers, origin="lower", cmap="jet")

nice_axis(ax2, 1, grid=False)
cbar = plt.colorbar()
cbar.set_label('Euler Characteristic', rotation=270)
cbar.ax.get_yaxis().labelpad = 15