import matplotlib.pyplot as plt
from matplotlib import colors
import os
import numpy as np

root_dir = "/home/mltest1/tmp/pycharm_project_883/Images/2020-02-13/10"
cmap = colors.ListedColormap(["black", "white", "orange"])
boundaries = [0, 0.5, 1]
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

fig1, f1_axes = plt.subplots(ncols=25, nrows=25, sharex=True, sharey=True)

for kT in np.linspace(0, 6, 25):
    for mu in np.linspace(0, 6, 25):
        img = np.loadtxt(f"{root_dir}/rabani_kT={kT:.2f}_mu={mu:.2f}_nrep=1_0.txt")
        plt.imshow(img, cmap=cmap)
