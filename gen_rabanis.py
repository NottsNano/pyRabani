import os
import re
import time
from datetime import datetime
from os import listdir
import h5py
import numpy as np
from matplotlib import colors
from numba import prange, jit

from rabani import rabani_single


def call_rabani_sweep(kT_mus, image_reps):
    assert 0 not in kT_mus

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"{current_time} - Beginning generation of {axis_res * axis_res * image_reps} rabanis")

    for image_rep in range(image_reps):
        imgs, m_all = run_rabani_sweep(kT_mus)
        imgs = np.swapaxes(imgs, 0, 2)
        save_rabanis(imgs, m_all, image_rep)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(
            f"{current_time} - Successfully completed block {image_rep + 1} of {image_reps} ({axis_res * axis_res} rabanis)")


@jit(nopython=True, parallel=True, fastmath=True)
def run_rabani_sweep(kT_mus):
    axis_steps = len(kT_mus)
    runs = np.zeros((128, 128, axis_steps))
    m_all = np.zeros((axis_steps,))

    for i in prange(axis_steps):
        runs[:, :, i], m_all[i] = rabani_single(kT_mus[i, 0], kT_mus[i, 1])

    return runs, m_all


def make_storage_folder(date):
    if os.path.isdir(f"{root_dir}/{date}/1") is False:
        lastrun = 0
    else:
        allrun_str = list(filter(re.compile("[0-9]").match,
                                 listdir(f"{root_dir}/{date}")))
        lastrun = int(np.max(list(map(int, allrun_str))))

    storage_folder = f"{root_dir}/{date}/{lastrun + 1}"
    os.makedirs(storage_folder)

    return storage_folder


def save_rabanis(imgs, m_all, image_rep, savetype="txt"):
    if savetype is "txt":
        for rep, img in enumerate(imgs):
            np.savetxt(
                f"{save_dir}/rabani_kT={kT_mus[rep, 0]:.2f}_mu={kT_mus[rep, 1]:.2f}_nsteps={int(m_all[rep]):d}_rep={image_rep}.txt",
                img, fmt="%01d")
    elif savetype is "hdf5":
        pass


if __name__ == '__main__':
    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    start = time.time()
    root_dir = "Images"
    total_image_reps = 1
    axis_res = 5
    kT_range = [0.001, 2]
    mu_range = [2, 4]

    # Make folders
    date = datetime.now().strftime("%Y-%m-%d")
    save_dir = make_storage_folder(date)

    # Shitty code to figure out order of pranges
    kT_mus = np.zeros((axis_res ** 2, 2))
    kT_mus_cnt = 0
    for kT_val in np.linspace(kT_range[0], kT_range[1], axis_res):
        for mu_val in np.linspace(mu_range[0], mu_range[1], axis_res):
            kT_mus[kT_mus_cnt, :] = [kT_val, mu_val]
            kT_mus_cnt += 1

    call_rabani_sweep(kT_mus, total_image_reps)
