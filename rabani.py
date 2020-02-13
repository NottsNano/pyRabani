"""
rabani.py

rabani model in python - adapted from:

Rabani.m - Andrew Stannard 27/06/19, Rabani model in Matlab
"""
from math import exp

import numpy as np
import matplotlib.pyplot as plt

L = 128
N = L**2 # system size

MCS = 500 # mc steps
MR = 1 # mobility ratio
C = 0.30 # covarage of nano particles

kT = 0.6
B = 1/kT

e_nl = 1.5 # nanoparticle-liquid interaction energy
e_nn = 2.0 # nanoparticle-nanoparticle interaction energy
mu = 2.8 # liquid chemical potential
# substrate = black, liquid = white, nanoparticle = orange
# TODO: do this with matplotlib
#colormap([0 0 0; 1 1 1; 1 0.5 0])

# want to have random nano particles as 1s and remove from liquid
nano_particles = np.zeros(N, dtype=int)
nano_particles[:round(C*N)] = 1
np.random.shuffle(nano_particles)
nano_particles = nano_particles.reshape(L, L)
liquid_array = 1 - nano_particles

fig, ax = plt.subplots()
ax.set_axis_off()

for m in range(MCS):
    # random position arrays for the evaporation/condensation loop
    x = np.random.randint(L, size=N)
    y = np.random.randint(L, size=N)

    # nearest neighbour arrays with periodic boundaries
    xp1 = (x + 1) % L
    yp1 = (y + 1) % L

    xm1 = (x - 1) % L
    ym1 = (y - 1) % L


    r = np.random.rand(N) # random number array for Metropolis acceptance

    for i in range(N): # start of evaporation/condensation loop
        if nano_particles[x[i], y[i]] == 0:
            if liquid_array[x[i], y[i]] == 0:
                # change in energy if condensation occurs
                dE = -(
                    liquid_array[xp1[i], y[i]] +
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], yp1[i]] +
                    liquid_array[x[i], ym1[i]]
                ) -e_nl * (
                    nano_particles[xp1[i], y[i]] +
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], yp1[i]] +
                    nano_particles[x[i], ym1[i]]
                ) + mu

                if r[i] < exp(-B * dE): # Metropolis acceptance
                    liquid_array[x[i], y[i]] = 1 # condensation
            else: # i.e. if LQ(x,y) == 1
                # change in energy if evaporation occurs
                dE = (
                    liquid_array[xp1[i], y[i]] +
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], yp1[i]] +
                    liquid_array[x[i], ym1[i]]
                ) + e_nl*(
                    nano_particles[xp1[i], y[i]] +
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], yp1[i]] +
                    nano_particles[x[i], ym1[i]]
                ) - mu

                if r[i] < exp(-B*dE): # Metropolis acceptance
                    liquid_array[x[i], y[i]] = 0 # evaporation

    # random number arrays for the nanoparticle diffusion loop
    x = np.random.randint(L, size=N*MR)
    y = np.random.randint(L, size=N*MR)

    # nearest and next nearest neighbour arrays with periodic boundaries
    xp1 = (x + 1) % L
    xp2 = (x + 2) % L

    yp1 = (y + 1) % L
    yp2 = (y + 2) % L

    xm1 = (x - 1) % L
    xm2 = (x - 2) % L

    ym1 = (y - 1) % L
    ym2 = (y - 2) % L

    r = np.random.rand(N*MR) # random number array for Metropolis acceptance
    # random number array for nanoparticle movement direction
    d = np.random.randint(1, 5, size=N*MR) # 1 = left, 2 = right, 3 = down, 4 = up

    for i in range(N*MR): # start of nanoparticle diffusion loop
        if nano_particles[x[i], y[i]] == 1:
            if (d[i] == 1) and liquid_array[xm1[i], y[i]]:
                # change in energy if nanoparticle moves left
                dE = -( # here we are repeating indices 4 times - totally unneccessary headache
                    liquid_array[x[i], ym1[i]] +
                    liquid_array[xp1[i], y[i]] +
                    liquid_array[x[i], yp1[i]] -
                    liquid_array[xm2[i], y[i]] -
                    liquid_array[xm1[i], yp1[i]] -
                    liquid_array[xm1[i], ym1[i]]
                ) - e_nl * (
                    liquid_array[xm2[i], y[i]] +
                    liquid_array[xm1[i], yp1[i]] +
                    liquid_array[xm1[i], ym1[i]] -
                    liquid_array[x[i], ym1[i]] -
                    liquid_array[xp1[i], y[i]] -
                    liquid_array[x[i], yp1[i]]
                ) -e_nl * (
                    nano_particles[x[i], ym1[i]] +
                    nano_particles[xp1[i], y[i]] +
                    nano_particles[x[i], yp1[i]] -
                    nano_particles[xm2[i], y[i]] -
                    nano_particles[xm1[i], yp1[i]] -
                    nano_particles[xm1[i], ym1[i]]
                ) - e_nn * (
                    nano_particles[xm2[i], y[i]] +
                    nano_particles[xm1[i], yp1[i]] +
                    nano_particles[xm1[i], ym1[i]] -
                    nano_particles[x[i], ym1[i]] -
                    nano_particles[xp1[i], y[i]] -
                    nano_particles[x[i], yp1[i]]
                )
                if r[i] < exp(-B*dE): # Metropolis acceptance
                    # move nanoparticles left
                    nano_particles[xm1[i], y[i]] = 1
                    nano_particles[x[i], y[i]] = 0
                    # move liquid right
                    liquid_array[x[i], y[i]] = 1
                    liquid_array[xm1[i], y[i]] = 0
            elif (d[i] == 2) and (liquid_array[xp1[i], y[i]] == 1):
                # fixed bug here
                dE = -(
                    liquid_array[x[i], ym1[i]] +
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], yp1[i]] -
                    liquid_array[xp2[i], y[i]] -
                    liquid_array[xp1[i], yp1[i]] -
                    liquid_array[xp1[i], ym1[i]]
                ) + e_nl * (
                    liquid_array[x[i], ym1[i]] +
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], yp1[i]] -
                    liquid_array[xp2[i], y[i]] -
                    liquid_array[xp1[i], yp1[i]] -
                    liquid_array[xp1[i], ym1[i]]
                ) - e_nl * (
                    nano_particles[x[i], ym1[i]] +
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], yp1[i]] -
                    nano_particles[xp2[i], y[i]] -
                    nano_particles[xp1[i], yp1[i]] -
                    nano_particles[xp1[i], ym1[i]]
                ) + e_nn * (
                    nano_particles[x[i], ym1[i]] +
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], yp1[i]] -
                    nano_particles[xp2[i], y[i]] -
                    nano_particles[xp1[i], yp1[i]] -
                    nano_particles[xp1[i], ym1[i]]
                )
                if r[i] < exp(-B*dE): # Metropolis acceptance
                    # move nano right
                    nano_particles[xp1[i], y[i]] = 1
                    nano_particles[x[i], y[i]] = 0
                    # move liquid left
                    liquid_array[x[i], y[i]] = 1
                    liquid_array[xp1[i], y[i]] = 0
            elif (d[i] == 3) and (liquid_array[x[i], ym1[i]] == 1):
                dE = -(
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], yp1[i]] +
                    liquid_array[xp1[i], y[i]] -
                    liquid_array[xm1[i], ym1[i]] -
                    liquid_array[x[i], ym2[i]] -
                    liquid_array[xp1[i], ym1[i]]
                ) + e_nl * (
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], yp1[i]] +
                    liquid_array[xp1[i], y[i]] -
                    liquid_array[xm1[i], ym1[i]] -
                    liquid_array[x[i], ym2[i]] -
                    liquid_array[xp1[i], ym1[i]]
                ) - e_nl * (
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], yp1[i]] +
                    nano_particles[xp1[i], y[i]] -
                    nano_particles[xm1[i], ym1[i]] -
                    nano_particles[x[i], ym2[i]] -
                    nano_particles[xp1[i], ym1[i]]
                ) + e_nn * (
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], yp1[i]] +
                    nano_particles[xp1[i], y[i]] -
                    nano_particles[xm1[i], ym1[i]] -
                    nano_particles[x[i], ym2[i]] -
                    nano_particles[xp1[i], ym1[i]]
                )
                if r[i] < exp(-B*dE): # Metropolis acceptance
                    # nano down
                    nano_particles[x[i], ym1[i]] = 1
                    nano_particles[x[i], y[i]] = 0
                    # liquid up
                    liquid_array[x[i], y[i]] = 1
                    liquid_array[x[i], ym1[i]] = 0 
            elif (d[i] == 4) and (liquid_array[x[i], yp1[i]] == 1):
                dE = -(
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], ym1[i]] +
                    liquid_array[xp1[i], y[i]] -
                    liquid_array[xm1[i], yp1[i]] -
                    liquid_array[x[i], yp2[i]] -
                    liquid_array[xp1[i], yp1[i]]
                ) + e_nl * (
                    liquid_array[xm1[i], y[i]] +
                    liquid_array[x[i], ym1[i]] +
                    liquid_array[xp1[i], y[i]] -
                    liquid_array[xm1[i], yp1[i]] -
                    liquid_array[x[i], yp2[i]] -
                    liquid_array[xp1[i], yp1[i]]
                ) - e_nl * (
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], ym1[i]] +
                    nano_particles[xp1[i], y[i]] -
                    nano_particles[xm1[i], yp1[i]] -
                    nano_particles[x[i], yp2[i]] -
                    nano_particles[xp1[i], yp1[i]]
                ) + e_nn * (
                    nano_particles[xm1[i], y[i]] +
                    nano_particles[x[i], ym1[i]] +
                    nano_particles[xp1[i], y[i]] -
                    nano_particles[xm1[i], yp1[i]] -
                    nano_particles[x[i], yp2[i]] -
                    nano_particles[xp1[i], yp1[i]]
                )
                if r[i] < exp(-B*dE): # Metropolis acceptance
                    # nano up
                    nano_particles[x[i], yp1[i]] = 1
                    nano_particles[x[i], y[i]] = 0
                    liquid_array[x[i], y[i]] = 1
                    liquid_array[x[i], yp1[i]] = 0

    if ((m+1) % 50) == 0:
        ax.imshow(2*nano_particles + liquid_array)
        zfill_no = f"{m+1}".zfill(5)
        fig.savefig(f"test{zfill_no}.png")

