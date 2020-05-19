"""
rabani model in python - adapted from Rabani.m - Andrew Stannard 27/06/19, Rabani cnn_model in Matlab
"""

from math import exp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from numba import jit, prange


@jit(nopython=True, fastmath=True, cache=True)
def rabani_single(kT, mu, MR, C, e_nl, e_nn, L):
    # L = 128  # System length
    N = L ** 2  # System volume

    MCS = 20000  # Max mc steps
    # MR = 1  # Mobility ratio
    # C = 0.30  # Nano-particle coverage

    # kT = 0.6
    B = 1 / kT

    # e_nl = 1.5  # nanoparticle-liquid interaction energy
    # e_nn = 2.0  # nanoparticle-nanoparticle interaction energy
    # mu = 2.8  # liquid chemical potential

    # Seed system array
    I = np.random.choice(N, int(C * N), replace=False)
    nano_particles = np.zeros((N,))
    nano_particles[I] = 1
    nano_particles = nano_particles.reshape((L, L))
    liquid_array = np.abs(1 - nano_particles)

    # Set up checkpointing
    checkpoint_out = np.ones((L, L))
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
            perc_similarities[-1] = np.mean(checkpoint_out == out)
            perc_similarities = np.roll(perc_similarities, -1)
            perc_similarities_std = np.std(np.diff(perc_similarities))
            checkpoint_out = 2 * nano_particles + liquid_array

        if 0 < perc_similarities_std < 0.002 and m > 200:
            break

    return out, m


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _run_rabani_sweep(params):
    axis_steps = len(params)
    runs = np.zeros((int(params[0, 6]), int(params[0, 6]), axis_steps))
    m_all = np.zeros((axis_steps,))

    for i in prange(axis_steps):
        runs[:, :, i], m_all[i] = rabani_single(kT=params[i, 0], mu=params[i, 1], MR=int(params[i, 2]), C=params[i, 3],
                                                e_nl=params[i, 4], e_nn=params[i, 5], L=int(params[i, 6]))

    return runs, m_all


if __name__ == '__main__':
    img, num_steps = rabani_single(kT=0.6, mu=2.8, MR=1, C=0.3, e_nl=1.5, e_nn=2, L=128)

    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.imshow(img, cmap=cmap)
