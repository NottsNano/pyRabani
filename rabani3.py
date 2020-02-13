"""
rabani.py

rabani model in python - adapted from:

Rabani.m - Andrew Stannard 27/06/19, Rabani model in Matlab
"""
import time
from math import exp

import numpy as np
from matplotlib import colors, pyplot as plt
from numba import jit, prange

@jit(nopython=True)
def rabani(num_runs):
    L = 128  # System length
    N = L ** 2  # System volume

    MCS = 1000  # Max mc steps
    MR = 1  # Mobility ratio
    C = 0.30  # Nano-particle coverage

    kT = 0.6
    B = 1 / kT

    e_nl = 1.5  # nanoparticle-liquid interaction energy
    e_nn = 2.0  # nanoparticle-nanoparticle interaction energy
    mu = 2.8  # liquid chemical potential

    all_outs = np.zeros((L, L, num_runs))

    # Seed system array
    big_I = np.random.choice(N*num_runs, int(C * N*num_runs), replace=False)
    big_nano_particles = np.zeros((N*num_runs))
    big_nano_particles[big_I] = 1
    big_nano_particles = big_nano_particles.reshape((L, L, num_runs))
    big_liquid_array = np.abs(1 - big_nano_particles)

    # Set up checkpointing
    big_I_tmp = np.random.choice(N*num_runs, int(C * N*num_runs), replace=False)
    big_checkpoint_out = np.zeros((N*num_runs,))
    big_checkpoint_out[big_I_tmp] = 1
    big_checkpoint_out = big_checkpoint_out.reshape((L, L, num_runs))
    big_out = 2 * big_nano_particles + big_liquid_array


    for num_run in prange(num_runs):
        liquid_array = big_liquid_array[:, :, num_run]
        nano_particles = big_nano_particles[:, :, num_run]
        checkpoint_out = big_checkpoint_out[:, :, num_run]
        out = big_out[:, :, num_run]

        perc_similarities = np.random.random((4,))
        perc_similarities_std = np.std(perc_similarities)

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
                checkpoint_out = 2 * nano_particles + liquid_array

            if 0 < perc_similarities_std < 0.015:
                all_outs[:, :, num_run] = out
                break

    return all_outs


if __name__ == '__main__':
    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    start = time.time()
    run = rabani(2)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    start = time.time()
    run = rabani(2)
    end = time.time()
    print("Elapsed (without compilation) = %s" % (end - start))

    # plt.imshow(run, cmap=cmap)
    # plt.axis("square")
    # plt.axis("off")
    # plt.pause(0.001)
