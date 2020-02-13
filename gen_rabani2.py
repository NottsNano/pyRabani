import numpy as np
from matplotlib import colors, pyplot as plt
from numba import jit, njit, prange

@jit(nopython=True)
def rabani():
    L = 128  # System length
    N = L ** 2  # System size
    MCS = 30  # Total num. of Monte Carlo steps
    MR = 30  # Mobility ratio
    C = 0.30  # Fractional coverage of nanoparticles
    kT = 0.25  # Thermal energy
    B = 1 / kT  # Inverse thermal energy
    e_nl = 1.5  # Interaction energy nanoparticle-liquid
    e_nn = 2.0  # Interaction energy nanoparticle-nanoparticle
    mu = 2.5  # Liquid chemical potential

    # Random initial nanoparticle positions
    I = np.random.choice(N, int(C * N), replace=False)
    NP = np.zeros((N,))
    NP[I] = 1
    NP = NP.reshape((L, L))
    LQ = np.abs(1 - NP)

    for m in range(MCS):
        # Random position arrays for the evaporation/condensation loop
        X = np.ceil(L * np.random.random((N,))) - 1
        Y = np.ceil(L * np.random.random((N,))) - 1

        # Nearest and next nearest neighbour arrays with periodic boundaries
        XP1 = X + 1 - L * (X == (L - 1))
        XP2 = X + 2 - L * (X >= (L - 2))
        YP1 = Y + 1 - L * (Y == (L - 1))
        YP2 = Y + 2 - L * (Y >= (L - 2))
        XM1 = X - 1 + L * (X == 0)
        XM2 = X - 2 + L * (X <= 1)
        YM1 = Y - 1 + L * (Y == 0)
        YM2 = Y - 2 + L * (Y <= 1)

        # Random number array for Metropolis acceptance
        R = np.random.random((N,))

        for i in range(N):  # Start of evaporation/condensation loop

            x = int(X[i])
            xp1 = int(XP1[i])
            xp2 = int(XP2[i])
            xm1 = int(XM1[i])
            xm2 = int(XM2[i])
            y = int(Y[i])
            yp1 = int(YP1[i])
            yp2 = int(YP2[i])
            ym1 = int(YM1[i])
            ym2 = int(YM2[i])
            r = R[i]
            # print(r, R[i])

            if NP[x, y] == 0:
                if LQ[x, y] == 0:
                    # change in energy if condensation occurs
                    dE = -(LQ[xp1, y] + LQ[xm1, y] + LQ[x, yp1] + LQ[x, ym1]) - e_nl * (
                            NP[xp1, y] + NP[xm1, y] + NP[x, yp1] + NP[x, ym1]) + mu

                    if r < np.exp(-B * dE):  # Metropolis acceptance
                        LQ[x, y] = 1  # condensation
                else:
                    # change in energy if evaporation occurs
                    dE = (LQ[xp1, y] + LQ[xm1, y] + LQ[x, yp1] + LQ[x, ym1]) + e_nl * (
                            NP[xp1, y] + NP[xm1, y] + NP[x, yp1] + NP[x, ym1]) - mu
                    if r < np.exp(-B * dE):  # Metropolis acceptance
                        LQ[x, y] = 0  # evaporation

        # Random number arrays for the nanoparticle diffusion loop
        X = np.ceil(L * np.random.random((N * MR,))) - 1
        Y = np.ceil(L * np.random.random((N * MR,))) - 1

        # Nearest and next nearest neighbour arrays with periodic boundaries
        XP1 = X + 1 - L * (X == (L - 1))
        XP2 = X + 2 - L * (X >= (L - 2))
        YP1 = Y + 1 - L * (Y == (L - 1))
        YP2 = Y + 2 - L * (Y >= (L - 2))
        XM1 = X - 1 + L * (X == 0)
        XM2 = X - 2 + L * (X <= 1)
        YM1 = Y - 1 + L * (Y == 0)
        YM2 = Y - 2 + L * (Y <= 1)

        R = np.random.random((N * MR,))  # random number array for Metropolis acceptance

        # Random number array for nanoparticle movement direction
        D = np.ceil(4 * np.random.random((N * MR,)))  # 1 = left, 2 = right, 3 = down, 4 = up

        for i in range(N * MR):  # start of nanoparticle diffusion loop
            x = int(X[i])
            xp1 = int(XP1[i])
            xp2 = int(XP2[i])
            xm1 = int(XM1[i])
            xm2 = int(XM2[i])
            y = int(Y[i])
            yp1 = int(YP1[i])
            yp2 = int(YP2[i])
            ym1 = int(YM1[i])
            ym2 = int(YM2[i])
            r = R[i]
            d = int(D[i])

            if NP[x, y] == 1:

                if (d == 1) and (LQ[xm1, y] == 1):

                    # change in energy if nanoparticle moves left
                    dE = -(LQ[x, ym1] + LQ[xp1, y] + LQ[x, yp1] - LQ[xm2, y] - LQ[xm1, yp1] - LQ[xm1, ym1]) - e_nl * (
                            LQ[xm2, y] + LQ[xm1, yp1] + LQ[xm1, ym1] - LQ[x, ym1] - LQ[xp1, y] - LQ[x, yp1]) - e_nl * (
                                 NP[x, ym1] + NP[xp1, y] + NP[x, yp1] - NP[xm2, y] - NP[xm1, yp1] - NP[
                             xm1, ym1]) - e_nn * (
                                 NP[xm2, y] + NP[xm1, yp1] + NP[xm1, ym1] - NP[x, ym1] - NP[xp1, y] - NP[x, yp1])

                    if r < np.exp(-B * dE):  # Metropolis acceptance
                        NP[xm1, y] = 1
                        NP[x, y] = 0  # nanoparticle moves left
                        LQ[x, y] = 1
                        LQ[xm1, y] = 0  # liquid moves right

                elif (d == 2) and (LQ[xp1, y] == 1):
                    # Change in energy if nanoparticle moves right
                    dE = -(LQ[x, ym1] + LQ[xp1, y] + LQ[x, yp1] - LQ[xp2, y] - LQ[xp1, yp1] - LQ[xp1, ym1]) - e_nl * (
                            LQ[xp2, y] + LQ[xp1, yp1] + LQ[xp1, ym1] - LQ[x, ym1] - LQ[xm1, y] - LQ[x, yp1]) - e_nl * (
                                 NP[x, ym1] + NP[xm1, y] + NP[x, yp1] - NP[xp2, y] - NP[xp1, yp1] - NP[
                             xp1, ym1]) - e_nn * (
                                    NP[xp2, y] + NP[xp1, yp1] + NP[xp1, ym1] - NP[x, ym1] - NP[xm1, y] - NP[x, yp1])

                    if r < np.exp(-B * dE):  # Metropolis acceptance
                        NP[xp1, y] = 1
                        NP[x, y] = 0  # Nanoparticle moves right
                        LQ[x, y] = 1
                        LQ[xp1, y] = 0  # Liquid moves left

                elif (d == 3) and (LQ[x, ym1] == 1):
                    # change in energy if nanoparticle moves down
                    dE = -(LQ[xm1, y] + LQ[x, yp1] + LQ[xp1, y] - LQ[xm1, ym1] - LQ[x, ym2] - LQ[xp1, ym1]) - e_nl * (
                            LQ[xm1, ym1] + LQ[x, ym2] + LQ[xp1, ym1] - LQ[xm1, y] - LQ[x, yp1] - LQ[xp1, y]) - e_nl * (
                                 NP[xm1, y] + NP[x, yp1] + NP[xp1, y] - NP[xm1, ym1] - NP[x, ym2] - NP[
                             xp1, ym1]) - e_nn * (
                                 NP[xm1, ym1] + NP[x, ym2] + NP[xp1, ym1] - NP[xm1, y] - NP[x, yp1] - NP[xp1, y])

                    if r < np.exp(-B * dE):  # Metropolis acceptance
                        NP[x, ym1] = 1
                        NP[x, y] = 0  # Nanoparticle moves down
                        LQ[x, y] = 1
                        LQ[x, ym1] = 0  # Liquid moves up

                elif (d == 4) and (LQ[x, yp1] == 1):
                    # Change in energy if nanoparticle moves up
                    dE = -(LQ[xm1, y] + LQ[x, ym1] + LQ[xp1, y] - LQ[xm1, yp1] - LQ[x, yp2] - LQ[xp1, yp1]) - e_nl * (
                            LQ[xm1, yp1] + LQ[x, yp2] + LQ[xp1, yp1] - LQ[xm1, y] - LQ[x, ym1] - LQ[xp1, y]) - e_nl * (
                                 NP[xm1, y] + NP[x, ym1] + NP[xp1, y] - NP[xm1, yp1] - NP[x, yp2] - NP[
                             xp1, yp1]) - e_nn * (
                                 NP[xm1, yp1] + NP[x, yp2] + NP[xp1, yp1] - NP[xm1, y] - NP[x, ym1] - NP[xp1, y])

                    if r < np.exp(-B * dE):  # Metropolis acceptance
                        NP[x, yp1] = 1
                        NP[x, y] = 0  # Nanoparticle moves up
                        LQ[x, y] = 1
                        LQ[x, yp1] = 0  # Liquid moves down

        # Check if equilibrium reached
        lastrun = 2 * NP + LQ

# if __name__ == '__main__':
#     cmap = colors.ListedColormap(["black", "white", "orange"])
#     boundaries = [0, 0.5, 1]
#     norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
#     plt.imshow(lastrun, cmap=cmap)
#     plt.axis("square")
#     plt.axis("off")
#     plt.pause(0.001)

