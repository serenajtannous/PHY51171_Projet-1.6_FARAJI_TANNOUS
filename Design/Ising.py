# We import librairies:

import numpy as np
import numpy.random as rnd
import itertools
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure
from IPython import display
import pandas as pd
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


#Ising model class
class Configuration:
    def __init__(self, L, T, J, H=0.0):
        self.length=L         # Length of the 1D Ising chain
        self.spins = rnd.choice([-1,1], size=L)  # Initialize spins randomly to ±1
        self.T=T   # Temperature of the system
        self.beta = 1/T  # Inverse temperature β = 1/T
        self.J=J  # Coupling constant between neighboring spins
        self.H = H # External magnetic field

    def get_magnetization(self):
        """ Return the total magnetization M = Σ_i s_i"""
        m=np.sum(self.spins)
        return m

    def metropolis_step(self):
        J = self.J
        L = self.length
        T = self.T
        H = self.H
        # Select a random spin to propose a flip
        i = rnd.randint(L)

        # Periodic boundary conditions: neighbors of site i
        i_minus_1 = (i - 1) % L
        i_plus_1 = (i + 1) % L

        # Compute energy change ΔE associated with flipping spin i
        delta_energy = 2 * self.spins[i] * (J * (self.spins[i_minus_1] + self.spins[i_plus_1]) + H)

        # If the energy decreases or stays the same → always accept the flip
        if delta_energy <= 0:
            self.spins[i] *= -1

        # If T is close to 0, no thermal fluctuations → reject all energy-increasing flips
        elif T < 1e-9:
            pass

        else:
            # Compute Metropolis acceptance probability: p = exp(-ΔE / T)
            p_accept = np.exp(-delta_energy / T)

            # Accept with probability p, otherwise keep the spin unchanged
            if rnd.random() < p_accept:
                self.spins[i] *= -1


def config_to_image(config):
    # Extract the spin configuration and chain length
    spins = config.spins
    L = config.length

    # Create a horizontal visualization of the chain
    plt.figure(figsize=(L/5, 1))
    ax = plt.gca()

    # Display the spins
    plt.pcolormesh(
        np.array([spins]),
        cmap='bwr',
        vmin=-1,
        vmax=1
    )


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Ising Chain (L={L})')
    plt.show()