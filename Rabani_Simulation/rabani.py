"""
rabani model in python - adapted from Rabani.m - Andrew Stannard 27/06/19, Rabani model in Matlab
"""

from math import exp

import numpy as np
from numba import jit, prange

from Analysis.plot_rabani import show_image


@jit(nopython=True, fastmath=True, cache=True)
def rabani_single(kT, mu, MR, C, e_nl, e_nn, L, MCS_max, early_stop):
    """A single rabani simulation """

    N = L ** 2  # System volume
    B = 1 / kT

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
    for m in range(MCS_max+1):
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

        out = 2 * nano_particles + liquid_array

        if early_stop:
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
    """Create multiple rabanis in parallel

    Parameters
    ----------
    params : ndarray
        (Nx9) array of the N simulations to run. The 9 values are kT, mu, MR, C, e_nl, e_nn, L, MCS_max, early_stop

    Returns
    -------
    runs : ndarray
        (NxLxL) array of simulations
    m_all : ndarray
        1D array of length N showing the number of MC steps taken in each of the N simulations

    See Also
    --------
    Rabani_Simulation.gen_rabanis.RabaniSweeper
    """
    axis_steps = len(params)
    runs = np.zeros((axis_steps, int(params[0, 6]), int(params[0, 6])))
    m_all = np.zeros((axis_steps,))

    for i in prange(axis_steps):
        runs[i, :, :], m_all[i] = rabani_single(kT=float(params[i, 0]), mu=float(params[i, 1]),
                                                MR=int(params[i, 2]), C=float(params[i, 3]),
                                                e_nl=float(params[i, 4]), e_nn=float(params[i, 5]), L=int(params[i, 6]),
                                                MCS_max=int(params[i, 7]), early_stop=bool(params[i, 8]))

    return runs, m_all


if __name__ == '__main__':
    # for MCS in np.linspace(100, 2000, 5):
    img, num_steps = rabani_single(kT=0.3, mu=2.55, MR=1, C=0.3, e_nl=1.5,
                               e_nn=2, L=128, MCS_max=600, early_stop=False)
    show_image(img)
