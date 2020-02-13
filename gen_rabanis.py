import os
import re
import time
from datetime import datetime
from os import listdir

import numpy as np
from matplotlib import colors
from numba import prange, jit

from rabani import rabani_single


@jit(nopython=True, parallel=True)
def gen_rabani_sweep(kT_min, kT_max, mu_min, mu_max, axis_steps):
    runs = np.zeros((128, 128, axis_steps ** 2))
    m_all = np.ones((axis_steps ** 2,))
    kT_all = np.linspace(kT_min, kT_max, axis_steps)
    mu_all = np.linspace(mu_min, mu_max, axis_steps)
    cnt = 0

    for i in prange(axis_steps):
        cnt += 1
        for j in prange(axis_steps):
            mu = mu_all[i]
            kT = kT_all[j]
            runs[:, :, cnt + i + j - 1], m_all[cnt + i + j - 1] = rabani_single(kT, mu)
            cnt += 1

    return runs, m_all


if __name__ == '__main__':
    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    start = time.time()
    root_dir = "Images"
    total_image_reps = 1
    axis_res = 25
    kT_range = [0, 6]
    mu_range = [0, 6]

    # Make folders
    date = datetime.now().strftime("%Y-%m-%d")
    if os.path.isdir(f"{root_dir}/{date}/1") is False:
        lastrun = 0
    else:
        allrun_str = list(filter(re.compile("[0-9]").match,
                                 listdir(f"{root_dir}/{date}")))
        lastrun = int(np.max(list(map(int, allrun_str))))
    os.makedirs(f"{root_dir}/{date}/{lastrun + 1}")

    # Shitty code to figure out order of pranges
    kT_mus = np.zeros((axis_res ** 2, 2))
    kT_mus_cnt = 0
    for kT_val in np.linspace(kT_range[0], kT_range[1], axis_res):
        for mu_val in np.linspace(mu_range[0], mu_range[1], axis_res):
            kT_mus[kT_mus_cnt, :] = [kT_val, mu_val]
            kT_mus_cnt += 1

    # Do the thing!
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"{current_time} - Beginning generation of {axis_res * axis_res * total_image_reps} rabanis")

    for image_rep in range(total_image_reps):
        imgs, m_all = gen_rabani_sweep(kT_range[0], kT_range[1], mu_range[0], mu_range[1], axis_res)
        imgs = np.swapaxes(imgs, 0, 2)

        for rep, img in enumerate(imgs):
            np.savetxt(
                f"{root_dir}/{date}/{lastrun + 1}/rabani_kT={kT_mus[rep, 0]:.1f}_mu={kT_mus[rep, 1]:.1f}_nsteps{int(m_all[rep]):d}_{image_rep}.txt",
                img, fmt="%01d")

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(
            f"{current_time} - Successfully completed block {image_rep + 1} of {total_image_reps} ({axis_res * axis_res} rabanis)")
