import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
from skimage import measure

root_dir = "/home/mltest1/tmp/pycharm_project_883/Images/2020-02-14/21-16"
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

    eulers[kT_ind, mu_ind] = measure.regionprops((img_file["image"][()] != 0).astype(int) + 1)[0][
                                 "euler_number"] / np.sum(img_file["image"][()] == 2)

# Plot
cmap = colors.ListedColormap(["black", "white", "orange"])
boundaries = [0, 0.5, 1]
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

fig1, ax1 = plt.subplots()
plt.imshow(big_img_arr, cmap=cmap, origin="lower")

plt.xticks(np.arange(len(mu_vals)) * img_res + img_res / 2, [f"{mu_val:.2f}" for mu_val in mu_vals], rotation=90)
plt.yticks(np.arange(len(kT_vals)) * img_res + img_res / 2, [f"{kT_val:.2f}" for kT_val in kT_vals])
ax1.set_xticks([x * img_res for x in range(axis_res)], minor=True)
ax1.set_yticks([y * img_res for y in range(axis_res)], minor=True)

ax1.set_xlabel("mu")
ax1.set_ylabel("kT")

plt.grid(which="minor", ls="-", lw=2, color="r")

# Euler
fig2, ax2 = plt.subplots()
cax2 = ax2.matshow(eulers, origin="lower", cmap="jet")
X, Y = np.meshgrid(np.arange(len(eulers)), np.arange(len(eulers)))
cnts2 = plt.contour(X, Y, eulers, colors="w", linestyles="solid")

# Remove small contour circles
for level in cnts2.collections:
    for kp, path in reversed(list(enumerate(level.get_paths()))):
        verts = path.vertices
        diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
        if diameter < 10:
            del (level.get_paths()[kp])

# Pretty up axis
num_tick_skip = 3
ax2.xaxis.tick_bottom()
ax2.set_xticklabels([''] + [f"{mu_val:.2f}" for mu_val in mu_vals][::num_tick_skip], rotation=90)
ax2.set_yticklabels([''] + [f"{kT_val:.2f}" for kT_val in kT_vals][::num_tick_skip])
ax2.xaxis.set_major_locator(MultipleLocator(num_tick_skip))
ax2.yaxis.set_major_locator(MultipleLocator(num_tick_skip))

cbar = fig2.colorbar(cax2)
cbar.add_lines(cnts2)
cbar.set_label('Euler Characteristic', rotation=270)
cbar.ax.get_yaxis().labelpad = 15

ax2.set_xlabel("mu")
ax2.set_ylabel("kT")
