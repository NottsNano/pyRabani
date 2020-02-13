import time

import numpy as np
from matplotlib import colors, pyplot as plt
from numba import jit, prange


@jit(nopython=True)
def rabani(num_reps, mu, kT):
    L = 1028  # System length
    N = L ** 2  # System size
    MCS = 1000  # Total num. of Monte Carlo steps
    MR = 30  # Mobility ratio
    C = 0.30  # Fractional coverage of nanoparticles
    kT = kT  # Thermal energy
    B = 1 / kT  # Inverse thermal energy
    e_nl = 1.5  # Interaction energy nanoparticle-liquid
    e_nn = 2.0  # Interaction energy nanoparticle-nanoparticle
    mu = mu  # Liquid chemical potential
    NUM_REPS = num_reps

    # Random initial nanoparticle positions
    I = np.random.choice(N * NUM_REPS, int(C * N * NUM_REPS), replace=False)
    NP = np.zeros((N * NUM_REPS,))
    NP[I] = 1
    NP = NP.reshape((NUM_REPS, L, L))
    LQ = np.abs(1 - NP)
    lastrun = np.zeros((NUM_REPS, L, L))

    # Random position arrays for the evaporation/condensation loop
    X1 = np.ceil(L * np.random.random((NUM_REPS, MCS, N))) - 1
    Y1 = np.ceil(L * np.random.random((NUM_REPS, MCS, N))) - 1

    # Random number array for Metropolis acceptance
    R1 = np.random.random((NUM_REPS, MCS, N))
    R2 = np.random.random((NUM_REPS, MCS, N * MR))

    # Random number arrays for the nanoparticle diffusion loop
    X2 = np.ceil(L * np.random.random((NUM_REPS, MCS, N * MR))) - 1
    Y2 = np.ceil(L * np.random.random((NUM_REPS, MCS, N * MR))) - 1

    # Random number array for nanoparticle movement direction
    D = np.ceil(4 * np.random.random((NUM_REPS, MCS, N * MR)))  # 1 = left, 2 = right, 3 = down, 4 = up

    for rep in prange(NUM_REPS):
        for m in range(MCS):

            # Nearest and next nearest neighbour arrays with periodic boundaries
            XP1 = X1[rep, m, :] + 1 - L * (X1[rep, m, :] == (L - 1))
            XP2 = X1[rep, m, :] + 2 - L * (X1[rep, m, :] >= (L - 2))
            YP1 = Y1[rep, m, :] + 1 - L * (Y1[rep, m, :] == (L - 1))
            YP2 = Y1[rep, m, :] + 2 - L * (Y1[rep, m, :] >= (L - 2))
            XM1 = X1[rep, m, :] - 1 + L * (X1[rep, m, :] == 0)
            XM2 = X1[rep, m, :] - 2 + L * (X1[rep, m, :] <= 1)
            YM1 = Y1[rep, m, :] - 1 + L * (Y1[rep, m, :] == 0)
            YM2 = Y1[rep, m, :] - 2 + L * (Y1[rep, m, :] <= 1)

            for i in range(N):  # Start of evaporation/condensation loop

                x = int(X1[rep, m, i])
                xp1 = int(XP1[i])
                xp2 = int(XP2[i])
                xm1 = int(XM1[i])
                xm2 = int(XM2[i])
                y = int(Y1[rep, m, i])
                yp1 = int(YP1[i])
                yp2 = int(YP2[i])
                ym1 = int(YM1[i])
                ym2 = int(YM2[i])
                r = R1[rep, m, i]
                # print(r, R[i])

                if NP[rep, x, y] == 0:
                    if LQ[rep, x, y] == 0:
                        # change in energy if condensation occurs
                        dE = -(LQ[rep, xp1, y] + LQ[rep, xm1, y] + LQ[rep, x, yp1] + LQ[rep, x, ym1]) - e_nl * (
                                NP[rep, xp1, y] + NP[rep, xm1, y] + NP[rep, x, yp1] + NP[rep, x, ym1]) + mu

                        if r < np.exp(-B * dE):  # Metropolis acceptance
                            LQ[rep, x, y] = 1  # condensation
                    else:
                        # change in energy if evaporation occurs
                        dE = (LQ[rep, xp1, y] + LQ[rep, xm1, y] + LQ[rep, x, yp1] + LQ[rep, x, ym1]) + e_nl * (
                                NP[rep, xp1, y] + NP[rep, xm1, y] + NP[rep, x, yp1] + NP[rep, x, ym1]) - mu
                        if r < np.exp(-B * dE):  # Metropolis acceptance
                            LQ[rep, x, y] = 0  # evaporation

            # Nearest and next nearest neighbour arrays with periodic boundaries
            XP1 = X2[rep, m, :] + 1 - L * (X2[rep, m, :] == (L - 1))
            XP2 = X2[rep, m, :] + 2 - L * (X2[rep, m, :] >= (L - 2))
            YP1 = Y2[rep, m, :] + 1 - L * (Y2[rep, m, :] == (L - 1))
            YP2 = Y2[rep, m, :] + 2 - L * (Y2[rep, m, :] >= (L - 2))
            XM1 = X2[rep, m, :] - 1 + L * (X2[rep, m, :] == 0)
            XM2 = X2[rep, m, :] - 2 + L * (X2[rep, m, :] <= 1)
            YM1 = Y2[rep, m, :] - 1 + L * (Y2[rep, m, :] == 0)
            YM2 = Y2[rep, m, :] - 2 + L * (Y2[rep, m, :] <= 1)

            for j in range(N * MR):  # start of nanoparticle diffusion loop
                x = int(X2[rep, m, j])
                xp1 = int(XP1[j])
                xp2 = int(XP2[j])
                xm1 = int(XM1[j])
                xm2 = int(XM2[j])
                y = int(Y2[rep, m, j])
                yp1 = int(YP1[j])
                yp2 = int(YP2[j])
                ym1 = int(YM1[j])
                ym2 = int(YM2[j])
                r = R2[rep, m, j]
                d = int(D[rep, m, j])

                if NP[rep, x, y] == 1:
                    if (d == 1) and (LQ[rep, xm1, y] == 1):
                        # change in energy if nanoparticle moves left
                        dE = -(LQ[rep, x, ym1] + LQ[rep, xp1, y] + LQ[rep, x, yp1] - LQ[rep, xm2, y] - LQ[
                            rep, xm1, yp1] - LQ[rep, xm1, ym1]) - e_nl * (
                                     LQ[rep, xm2, y] + LQ[rep, xm1, yp1] + LQ[rep, xm1, ym1] - LQ[rep, x, ym1] - LQ[
                                 rep, xp1, y] - LQ[rep, x, yp1]) - e_nl * (
                                     NP[rep, x, ym1] + NP[rep, xp1, y] + NP[rep, x, yp1] - NP[rep, xm2, y] - NP[
                                 rep, xm1, yp1] - NP[
                                         rep, xm1, ym1]) - e_nn * (
                                     NP[rep, xm2, y] + NP[rep, xm1, yp1] + NP[rep, xm1, ym1] - NP[rep, x, ym1] - NP[
                                 rep, xp1, y] - NP[rep, x, yp1])

                        if r < np.exp(-B * dE):  # Metropolis acceptance
                            NP[rep, xm1, y] = 1
                            NP[rep, x, y] = 0  # nanoparticle moves left
                            LQ[rep, x, y] = 1
                            LQ[rep, xm1, y] = 0  # liquid moves right

                    elif (d == 2) and (LQ[rep, xp1, y] == 1):
                        # Change in energy if nanoparticle moves right
                        dE = -(LQ[rep, x, ym1] + LQ[rep, xp1, y] + LQ[rep, x, yp1] - LQ[rep, xp2, y] - LQ[
                            rep, xp1, yp1] - LQ[rep, xp1, ym1]) - e_nl * (
                                     LQ[rep, xp2, y] + LQ[rep, xp1, yp1] + LQ[rep, xp1, ym1] - LQ[rep, x, ym1] - LQ[
                                 rep, xm1, y] - LQ[rep, x, yp1]) - e_nl * (
                                         NP[rep, x, ym1] + NP[rep, xm1, y] + NP[rep, x, yp1] - NP[rep, xp2, y] - NP[
                                     rep, xp1, yp1] - NP[rep, xp1, ym1]) - e_nn * (
                                     NP[rep, xp2, y] + NP[rep, xp1, yp1] + NP[rep, xp1, ym1] - NP[rep, x, ym1] - NP[
                                 rep, xm1, y] - NP[rep, x, yp1])

                        if r < np.exp(-B * dE):  # Metropolis acceptance
                            NP[rep, xp1, y] = 1
                            NP[rep, x, y] = 0  # Nanoparticle moves right
                            LQ[rep, x, y] = 1
                            LQ[rep, xp1, y] = 0  # Liquid moves left

                    elif (d == 3) and (LQ[rep, x, ym1] == 1):
                        # change in energy if nanoparticle moves down
                        dE = -(LQ[rep, xm1, y] + LQ[rep, x, yp1] + LQ[rep, xp1, y] - LQ[rep, xm1, ym1] - LQ[
                            rep, x, ym2] - LQ[
                                   rep, xp1, ym1]) - e_nl * (
                                     LQ[rep, xm1, ym1] + LQ[rep, x, ym2] + LQ[rep, xp1, ym1] - LQ[rep, xm1, y] - LQ[
                                 rep, x, yp1] - LQ[
                                         rep, xp1, y]) - e_nl * (
                                     NP[rep, xm1, y] + NP[rep, x, yp1] + NP[rep, xp1, y] - NP[rep, xm1, ym1] - NP[
                                 rep, x, ym2] - NP[
                                         rep, xp1, ym1]) - e_nn * (
                                     NP[rep, xm1, ym1] + NP[rep, x, ym2] + NP[rep, xp1, ym1] - NP[rep, xm1, y] - NP[
                                 rep, x, yp1] - NP[rep, xp1, y])

                        if r < np.exp(-B * dE):  # Metropolis acceptance
                            NP[rep, x, ym1] = 1
                            NP[rep, x, y] = 0  # Nanoparticle moves down
                            LQ[rep, x, y] = 1
                            LQ[rep, x, ym1] = 0  # Liquid moves up

                    elif (d == 4) and (LQ[rep, x, yp1] == 1):
                        # Change in energy if nanoparticle moves up
                        dE = -(LQ[rep, xm1, y] + LQ[rep, x, ym1] + LQ[rep, xp1, y] - LQ[rep, xm1, yp1] - LQ[
                            rep, x, yp2] - LQ[
                                   rep, xp1, yp1]) - e_nl * (
                                     LQ[rep, xm1, yp1] + LQ[rep, x, yp2] + LQ[rep, xp1, yp1] - LQ[rep, xm1, y] - LQ[
                                 rep, x, ym1] - LQ[
                                         rep, xp1, y]) - e_nl * (
                                     NP[rep, xm1, y] + NP[rep, x, ym1] + NP[rep, xp1, y] - NP[rep, xm1, yp1] - NP[
                                 rep, x, yp2] - NP[
                                         rep, xp1, yp1]) - e_nn * (
                                     NP[rep, xm1, yp1] + NP[rep, x, yp2] + NP[rep, xp1, yp1] - NP[rep, xm1, y] - NP[
                                 rep, x, ym1] - NP[rep, xp1, y])

                        if r < np.exp(-B * dE):  # Metropolis acceptance
                            NP[rep, x, yp1] = 1
                            NP[rep, x, y] = 0  # Nanoparticle moves up
                            LQ[rep, x, y] = 1
                            LQ[rep, x, yp1] = 0  # Liquid moves down

            # Check if equilibrium reached
            last = 2 * NP[rep, :, :] + LQ[rep, :, :]
            lastrun[rep, :, :] = last

    return lastrun


if __name__ == '__main__':
    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    num_reps = 2
    mus = np.arange(0.01, 6.1, 1)
    kTs = np.arange(0.01, 3.01, 0.5)
    for kT in kTs:
        for mu in mus:
            runs = rabani(num_reps, mu, kT)
            for rep, img in enumerate(runs):
                plt.imsave(f"ImagesOriginal/rabani_kT={kT:.1f}_mu={mu:.1f}_{rep}.png", img, cmap=cmap)

    # plt.imshow(run, cmap=cmap)
    # plt.axis("square")
    # plt.axis("off")
    # plt.pause(0.001)
