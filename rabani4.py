"""
rabani.py

rabani model in python - adapted from:

Rabani.m - Andrew Stannard 27/06/19, Rabani model in Matlab
"""
import os
import re
import time
from datetime import datetime
from math import exp
from os import listdir

import numpy as np
from matplotlib import colors
from numba import jit, prange


@jit(nopython=True)
def rabani(kT, mu):
    L = 128  # System length
    N = L ** 2  # System volume

    MCS = 1000  # Max mc steps
    MR = 1  # Mobility ratio
    C = 0.30  # Nano-particle coverage

    # kT = 0.6
    B = 1 / kT

    e_nl = 1.5  # nanoparticle-liquid interaction energy
    e_nn = 2.0  # nanoparticle-nanoparticle interaction energy
    # mu = 2.8  # liquid chemical potential

    # Seed system array
    I = np.random.choice(N, int(C * N), replace=False)
    nano_particles = np.zeros((N,))
    nano_particles[I] = 1
    nano_particles = nano_particles.reshape((L, L))
    liquid_array = np.abs(1 - nano_particles)

    # Set up checkpointing
    I_tmp = np.random.choice(N, int(C * N), replace=False)
    checkpoint_out = np.zeros((N,))
    checkpoint_out[I_tmp] = 1
    checkpoint_out = checkpoint_out.reshape((L, L))
    out = 2 * nano_particles + liquid_array

    perc_similarities = np.random.random((4,))
    perc_similarities_std = np.std(perc_similarities)
    m = 1
    for m in range(MCS):
        # random position arrays for the evaporation/condensation loop
        x = np.ceil(L * np.random.random((N,))) - 1
        y = np.ceil(L * np.random.random((N,))) - 1

        # nearest neighbour arrays with periodic boundaries
        xp1 = (x + 1) % L
        yp1 = (y + 1) % L

        xm1 = (x - 1) % L
        ym1 = (y - 1) % L

        r = np.random.rand(N)  # random number array for Metropolis acceptance

        for i in range(N):  # start of evaporation/condensation loop
            if nano_particles[int(x[i]), int(int(y[i]))] == 0:
                if liquid_array[int(x[i]), int(int(y[i]))] == 0:
                    # change in energy if condensation occurs
                    dE = -(
                            liquid_array[int(xp1[i]), int(y[i])] +
                            liquid_array[int(xm1[i]), int(y[i])] +
                            liquid_array[int(x[i]), int(yp1[i])] +
                            liquid_array[int(x[i]), int(ym1[i])]
                    ) - e_nl * (
                                 nano_particles[int(xp1[i]), int(y[i])] +
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(yp1[i])] +
                                 nano_particles[int(x[i]), int(ym1[i])]
                         ) + mu

                    if r[i] < exp(-B * dE):  # Metropolis acceptance
                        liquid_array[int(x[i]), int(y[i])] = 1  # condensation
                else:  # i.e. if LQ(x,y) == 1
                    # change in energy if evaporation occurs
                    dE = (
                                 liquid_array[int(xp1[i]), int(y[i])] +
                                 liquid_array[int(xm1[i]), int(y[i])] +
                                 liquid_array[int(x[i]), int(yp1[i])] +
                                 liquid_array[int(x[i]), int(ym1[i])]
                         ) + e_nl * (
                                 nano_particles[int(xp1[i]), int(y[i])] +
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(yp1[i])] +
                                 nano_particles[int(x[i]), int(ym1[i])]
                         ) - mu

                    if r[i] < exp(-B * dE):  # Metropolis acceptance
                        liquid_array[int(x[i]), int(y[i])] = 0  # evaporation

        # random number arrays for the nanoparticle diffusion loop
        x = np.ceil(L * np.random.random((N * MR,))) - 1
        y = np.ceil(L * np.random.random((N * MR,))) - 1

        # nearest and next nearest neighbour arrays with periodic boundaries
        xp1 = (x + 1) % L
        xp2 = (x + 2) % L

        yp1 = (y + 1) % L
        yp2 = (y + 2) % L

        xm1 = (x - 1) % L
        xm2 = (x - 2) % L

        ym1 = (y - 1) % L
        ym2 = (y - 2) % L

        r = np.random.rand(N * MR)  # random number array for Metropolis acceptance
        # random number array for nanoparticle movement direction
        d = np.random.randint(1, 5, size=N * MR)  # 1 = left, 2 = right, 3 = down, 4 = up

        for i in range(N * MR):  # start of nanoparticle diffusion loop
            if nano_particles[int(x[i]), int(y[i])] == 1:
                if (d[i] == 1) and liquid_array[int(xm1[i]), int(y[i])]:
                    # change in energy if nanoparticle moves left
                    dE = -(  # here we are repeating indices 4 times - totally unneccessary headache
                            liquid_array[int(x[i]), int(ym1[i])] +
                            liquid_array[int(xp1[i]), int(y[i])] +
                            liquid_array[int(x[i]), int(yp1[i])] -
                            liquid_array[int(xm2[i]), int(y[i])] -
                            liquid_array[int(xm1[i]), int(yp1[i])] -
                            liquid_array[int(xm1[i]), int(ym1[i])]
                    ) - e_nl * (
                                 liquid_array[int(xm2[i]), int(y[i])] +
                                 liquid_array[int(xm1[i]), int(yp1[i])] +
                                 liquid_array[int(xm1[i]), int(ym1[i])] -
                                 liquid_array[int(x[i]), int(ym1[i])] -
                                 liquid_array[int(xp1[i]), int(y[i])] -
                                 liquid_array[int(x[i]), int(yp1[i])]
                         ) - e_nl * (
                                 nano_particles[int(x[i]), int(ym1[i])] +
                                 nano_particles[int(xp1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(yp1[i])] -
                                 nano_particles[int(xm2[i]), int(y[i])] -
                                 nano_particles[int(xm1[i]), int(yp1[i])] -
                                 nano_particles[int(xm1[i]), int(ym1[i])]
                         ) - e_nn * (
                                 nano_particles[int(xm2[i]), int(y[i])] +
                                 nano_particles[int(xm1[i]), int(yp1[i])] +
                                 nano_particles[int(xm1[i]), int(ym1[i])] -
                                 nano_particles[int(x[i]), int(ym1[i])] -
                                 nano_particles[int(xp1[i]), int(y[i])] -
                                 nano_particles[int(x[i]), int(yp1[i])]
                         )
                    if r[i] < exp(-B * dE):  # Metropolis acceptance
                        # move nanoparticles left
                        nano_particles[int(xm1[i]), int(y[i])] = 1
                        nano_particles[int(x[i]), int(y[i])] = 0
                        # move liquid right
                        liquid_array[int(x[i]), int(y[i])] = 1
                        liquid_array[int(xm1[i]), int(y[i])] = 0
                elif (d[i] == 2) and (liquid_array[int(xp1[i]), int(y[i])] == 1):
                    # fixed bug here
                    dE = -(
                            liquid_array[int(x[i]), int(ym1[i])] +
                            liquid_array[int(xm1[i]), int(y[i])] +
                            liquid_array[int(x[i]), int(yp1[i])] -
                            liquid_array[int(xp2[i]), int(y[i])] -
                            liquid_array[int(xp1[i]), int(yp1[i])] -
                            liquid_array[int(xp1[i]), int(ym1[i])]
                    ) + e_nl * (
                                 liquid_array[int(x[i]), int(ym1[i])] +
                                 liquid_array[int(xm1[i]), int(y[i])] +
                                 liquid_array[int(x[i]), int(yp1[i])] -
                                 liquid_array[int(xp2[i]), int(y[i])] -
                                 liquid_array[int(xp1[i]), int(yp1[i])] -
                                 liquid_array[int(xp1[i]), int(ym1[i])]
                         ) - e_nl * (
                                 nano_particles[int(x[i]), int(ym1[i])] +
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(yp1[i])] -
                                 nano_particles[int(xp2[i]), int(y[i])] -
                                 nano_particles[int(xp1[i]), int(yp1[i])] -
                                 nano_particles[int(xp1[i]), int(ym1[i])]
                         ) + e_nn * (
                                 nano_particles[int(x[i]), int(ym1[i])] +
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(yp1[i])] -
                                 nano_particles[int(xp2[i]), int(y[i])] -
                                 nano_particles[int(xp1[i]), int(yp1[i])] -
                                 nano_particles[int(xp1[i]), int(ym1[i])]
                         )
                    if r[i] < exp(-B * dE):  # Metropolis acceptance
                        # move nano right
                        nano_particles[int(xp1[i]), int(y[i])] = 1
                        nano_particles[int(x[i]), int(y[i])] = 0
                        # move liquid left
                        liquid_array[int(x[i]), int(y[i])] = 1
                        liquid_array[int(xp1[i]), int(y[i])] = 0
                elif (d[i] == 3) and (liquid_array[int(x[i]), int(ym1[i])] == 1):
                    dE = -(
                            liquid_array[int(xm1[i]), int(y[i])] +
                            liquid_array[int(x[i]), int(yp1[i])] +
                            liquid_array[int(xp1[i]), int(y[i])] -
                            liquid_array[int(xm1[i]), int(ym1[i])] -
                            liquid_array[int(x[i]), int(ym2[i])] -
                            liquid_array[int(xp1[i]), int(ym1[i])]
                    ) + e_nl * (
                                 liquid_array[int(xm1[i]), int(y[i])] +
                                 liquid_array[int(x[i]), int(yp1[i])] +
                                 liquid_array[int(xp1[i]), int(y[i])] -
                                 liquid_array[int(xm1[i]), int(ym1[i])] -
                                 liquid_array[int(x[i]), int(ym2[i])] -
                                 liquid_array[int(xp1[i]), int(ym1[i])]
                         ) - e_nl * (
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(yp1[i])] +
                                 nano_particles[int(xp1[i]), int(y[i])] -
                                 nano_particles[int(xm1[i]), int(ym1[i])] -
                                 nano_particles[int(x[i]), int(ym2[i])] -
                                 nano_particles[int(xp1[i]), int(ym1[i])]
                         ) + e_nn * (
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(yp1[i])] +
                                 nano_particles[int(xp1[i]), int(y[i])] -
                                 nano_particles[int(xm1[i]), int(ym1[i])] -
                                 nano_particles[int(x[i]), int(ym2[i])] -
                                 nano_particles[int(xp1[i]), int(ym1[i])]
                         )
                    if r[i] < exp(-B * dE):  # Metropolis acceptance
                        # nano down
                        nano_particles[int(x[i]), int(ym1[i])] = 1
                        nano_particles[int(x[i]), int(y[i])] = 0
                        # liquid up
                        liquid_array[int(x[i]), int(y[i])] = 1
                        liquid_array[int(x[i]), int(ym1[i])] = 0
                elif (d[i] == 4) and (liquid_array[int(x[i]), int(yp1[i])] == 1):
                    dE = -(
                            liquid_array[int(xm1[i]), int(y[i])] +
                            liquid_array[int(x[i]), int(ym1[i])] +
                            liquid_array[int(xp1[i]), int(y[i])] -
                            liquid_array[int(xm1[i]), int(yp1[i])] -
                            liquid_array[int(x[i]), int(yp2[i])] -
                            liquid_array[int(xp1[i]), int(yp1[i])]
                    ) + e_nl * (
                                 liquid_array[int(xm1[i]), int(y[i])] +
                                 liquid_array[int(x[i]), int(ym1[i])] +
                                 liquid_array[int(xp1[i]), int(y[i])] -
                                 liquid_array[int(xm1[i]), int(yp1[i])] -
                                 liquid_array[int(x[i]), int(yp2[i])] -
                                 liquid_array[int(xp1[i]), int(yp1[i])]
                         ) - e_nl * (
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(ym1[i])] +
                                 nano_particles[int(xp1[i]), int(y[i])] -
                                 nano_particles[int(xm1[i]), int(yp1[i])] -
                                 nano_particles[int(x[i]), int(yp2[i])] -
                                 nano_particles[int(xp1[i]), int(yp1[i])]
                         ) + e_nn * (
                                 nano_particles[int(xm1[i]), int(y[i])] +
                                 nano_particles[int(x[i]), int(ym1[i])] +
                                 nano_particles[int(xp1[i]), int(y[i])] -
                                 nano_particles[int(xm1[i]), int(yp1[i])] -
                                 nano_particles[int(x[i]), int(yp2[i])] -
                                 nano_particles[int(xp1[i]), int(yp1[i])]
                         )
                    if r[i] < exp(-B * dE):  # Metropolis acceptance
                        # nano up
                        nano_particles[int(x[i]), int(yp1[i])] = 1
                        nano_particles[int(x[i]), int(y[i])] = 0
                        liquid_array[int(x[i]), int(y[i])] = 1
                        liquid_array[int(x[i]), int(yp1[i])] = 0

        # Early stopping
        out = 2 * nano_particles + liquid_array
        if m % 25 == 0:  # Check every 25th iteration
            # plt.imsave(f"ImagesWORKING/rabani_{m}.png", out, cmap=cmap)
            perc_similarities[-1] = np.mean(checkpoint_out == out)
            perc_similarities = np.roll(perc_similarities, -1)
            perc_similarities_std = np.std(np.diff(perc_similarities))
            # print(perc_similarities, perc_similarities_std)
            checkpoint_out = 2 * nano_particles + liquid_array

        if 0 < perc_similarities_std < 0.005 and m < 100:
            break

    return out, m


# Numba doesn't like lists, generators, etc, so have to do this!
@jit(nopython=True, parallel=True)
def run_rabani(kT_min, kT_max, mu_min, mu_max, axis_steps):
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
            runs[:, :, cnt + i + j - 1], m_all[cnt + i + j - 1] = rabani(kT, mu)
            cnt += 1

    return runs, m_all


def plot_rabani(root_dir):
    pass


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
        imgs, m_all = run_rabani(kT_range[0], kT_range[1], mu_range[0], mu_range[1], axis_res)
        imgs = np.swapaxes(imgs, 0, 2)

        for rep, img in enumerate(imgs):
            np.savetxt(
                f"{root_dir}/{date}/{lastrun + 1}/rabani_kT={kT_mus[rep, 0]:.1f}_mu={kT_mus[rep, 1]:.1f}_nsteps{int(m_all[rep]):d}_{image_rep}.txt",
                img, fmt="%01d")

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(
            f"{current_time} - Successfully completed block {image_rep + 1} of {total_image_reps} ({axis_res * axis_res} rabanis)")
