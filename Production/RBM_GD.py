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
from Design.Ising import Configuration
from Exact_Energy import exact_1d_ising_energy, compute_relative_error


class RBM:
    """
    Restricted Boltzmann Machine for 1D Ising model
    Psi(S) = exp(sum_j a_j s_j) * prod_i 2*cosh(b_i + sum_j W_ij s_j)
    """
    def __init__(self, n_visible, n_hidden, seed=None):
        """n_visible:   Number of visible units (physical spins in the system)
        n_hidden:  Number of hidden units (hidden neurons defining correlations)
        seed: Setting a seed makes all random initializations reproducible.
        """
        self.N = n_visible
        self.M = n_hidden
        self.alpha = n_hidden / n_visible  # Calculate alpha from n_hidden/n_visible

        #Random number generator
        self.rng = np.random.default_rng(seed)

        # initialize parameters small
        self.a = 0.01 * self.rng.normal(size=self.N)           # visible biases (shape N,)
        self.b = 0.01 * self.rng.normal(size=self.M)           # hidden biases (shape M,)
        self.W = 0.01 * self.rng.normal(size=(self.M, self.N)) # weights (M x N)  --> Weight matrix W_{ij} connecting visible spin j to hidden neuron i.

    def psi(self, s):
        """Wavefunction amplitude for configuration s"""
        # linear visible term
        v = np.dot(self.a, s)
        term1=np.exp(v)
        #hidden contributions
        x = self.b + self.W @ s   # shape (M,)
        term2=np.prod(2*np.cosh(x))
        return term1*term2

    def log_psi(self, s):
        """Log of wavefunction amplitude"""
        visible_term = np.dot(self.a, s)
        x = self.b + self.W @ s
        hidden_term = np.sum(np.log(2 * np.cosh(x)))

        return visible_term + hidden_term

    def log_psi_derivs(self, s):
        """
        Log-derivatives d ln psi / d param for a single configuration s.
        Returns dict of arrays for a, b, W (same shapes).

        d ln psi / d a_j = s_j
        d ln psi / d b_i = tanh(b_i + sum_j W_ij s_j)
        d ln psi / d W_ij = s_j * tanh(...)
        """
        x = self.b + self.W @ s            # shape (M,)
        th = np.tanh(x)                    # shape (M,)
        da = s.copy()                      # shape (N,)
        db = th.copy()                     # shape (M,)
        dW = np.outer(th, s)               # shape (M, N)
        return da, db, dW




    def get_log_psi_ratio(self, sigma, i):
        """
        Calcule le ratio R_i = Psi_RBM(sigma^(i)) / Psi_RBM(sigma)
        R_i = exp( log(Psi_RBM(sigma^(i))) - log(Psi_RBM(sigma)) )

        sigma^(i) est la configuration sigma avec le spin i flippé.
        """

        # Pour le spin flippé, l'action est: sigma[i] -> -sigma[i]
        sigma_i_initial = sigma[i]
        sigma_i_flipped = -sigma_i_initial

        # Différence des termes de biases visibles
        # d(log_term_v) = a_i * (sigma_i_flipped - sigma_i_initial) = a_i * (-2 * sigma_i_initial)
        diff_log_term_v = self.a[i] * (-2 * sigma_i_initial)

        # Différence des termes de biases cachés
        # h_j_flipped = b_j + sum_{k!=i} W_jk * sigma_k + W_ji * sigma_i_flipped
        # h_j_initial = b_j + sum_{k!=i} W_jk * sigma_k + W_ji * sigma_i_initial
        # L'argument de la j-ème cosh change de :
        # Delta_h_j = h_j_flipped - h_j_initial = W_ji * (sigma_i_flipped - sigma_i_initial)
        Delta_h_j = self.W[:, i] * (-2 * sigma_i_initial)

        # Calcul des arguments de cosh initiaux
        h_initial = self.b + self.W @ sigma

        # Calcul des arguments de cosh flippés
        h_flipped = h_initial + Delta_h_j

        # Différence des log des termes cachés
        diff_log_term_h = np.sum(np.log(np.cosh(h_flipped)) - np.log(np.cosh(h_initial)))

        # La différence des log-fonctions d'onde est:
        log_ratio = diff_log_term_v + diff_log_term_h

        # Le ratio est l'exponentielle de cette différence
        return np.exp(log_ratio)

    def get_log_psi_ratio(self, sigma, i):
        """
        Calcule le ratio R_i = Psi_RBM(sigma^(i)) / Psi_RBM(sigma)
        R_i = exp( log(Psi_RBM(sigma^(i))) - log(Psi_RBM(sigma)) )

        sigma^(i) est la configuration sigma avec le spin i flippé.
        """

        # Pour le spin flippé, l'action est: sigma[i] -> -sigma[i]
        sigma_i_initial = sigma[i]
        sigma_i_flipped = -sigma_i_initial

        # Différence des termes de biases visibles
        # d(log_term_v) = a_i * (sigma_i_flipped - sigma_i_initial) = a_i * (-2 * sigma_i_initial)
        diff_log_term_v = self.a[i] * (-2 * sigma_i_initial)

        # Différence des termes de biases cachés
        # h_j_flipped = b_j + sum_{k!=i} W_jk * sigma_k + W_ji * sigma_i_flipped
        # h_j_initial = b_j + sum_{k!=i} W_jk * sigma_k + W_ji * sigma_i_initial
        # L'argument de la j-ème cosh change de :
        # Delta_h_j = h_j_flipped - h_j_initial = W_ji * (sigma_i_flipped - sigma_i_initial)
        Delta_h_j = self.W[:, i] * (-2 * sigma_i_initial)

        # Calcul des arguments de cosh initiaux
        h_initial = self.b + self.W @ sigma

        # Calcul des arguments de cosh flippés
        h_flipped = h_initial + Delta_h_j

        # Différence des log des termes cachés
        diff_log_term_h = np.sum(np.log(np.cosh(h_flipped)) - np.log(np.cosh(h_initial)))

        # La différence des log-fonctions d'onde est:
        log_ratio = diff_log_term_v + diff_log_term_h

        # Le ratio est l'exponentielle de cette différence
        return np.exp(log_ratio)


    def get_energy_local(self, config):
        """
        Calcule l'énergie locale E_L(sigma) = <sigma|H|Psi> / <sigma|Psi>
        """
        N = self.N # Nombre de spins
        sigma=config.spins
        # 1. Contribution diagonale (Terme d'interaction H_z)
        # E_classique(sigma) = -J * sum(sigma_i * sigma_{i+1})
        # Conditions aux bords périodiques: sigma_{N+1} = sigma_1

        # Calcul des produits des voisins (jusqu'à N-1 et N)
        interaction_sum = np.sum(sigma[:-1] * sigma[1:])

        # Ajout du terme périodique sigma_N * sigma_1
        interaction_sum += sigma[-1] * sigma[0]

        E_classique = -config.J * interaction_sum

        # 2. Contribution non-diagonale (Terme de champ transverse H_x)
        # Terme_non_diag = -H * sum_i [ Psi(sigma^(i)) / Psi(sigma) ]

        sum_of_ratios = 0.0 + 0.0j # Initialisation pour les complexes

        for i in range(N):
            # Calcule le ratio de la fonction d'onde R_i pour le flip au site i
            ratio_i = self.get_log_psi_ratio(sigma, i)
            sum_of_ratios += ratio_i

        E_non_diag = -config.H * sum_of_ratios

        # 3. Énergie Locale Totale
        E_local = E_classique + E_non_diag

        return E_local



    def local_energy(self, config):
        """
        Local energy using existing Configuration class
        """
        return self.get_energy_local(config)


    def metropolis_step(self, config):
        """Single Metropolis-Hastings step using |psi|^2"""
        # Create a copy to test spin flip
        new_config = Configuration(config.length, config.T, config.J, config.H)
        new_config.spins = config.spins.copy()

        # Flip a random spin
        i = self.rng.integers(config.length)
        new_config.spins[i] *= -1

        # Acceptance probability: |psi(new)|^2 / |psi(old)|^2
        log_p_accept = 2 * (self.log_psi(new_config.spins) - self.log_psi(config.spins))

        if log_p_accept >= 0 or self.rng.random() < np.exp(log_p_accept):
            config.spins = new_config.spins.copy()
            return True
        return False


    def sample(self, n_samples, n_burnin=1000, T=1e-10, J=1.0, H=0.0):
        """Generate samples using Metropolis-Hastings from |psi|^2 (proba of distribution)"""
        # Initialize configuration
        config = Configuration(self.N, T, J, H)
        config.spins = self.rng.choice([-1, 1], size=self.N)

        # Burn-in phase: the system will forgot its initial configuration and converge to the distribtion target |psi|^2
        for _ in range(n_burnin):
            self.metropolis_step(config)

        # Sampling phase
        samples = []
        energies = []
        acceptances = 0
        total_steps = 0

        while len(samples) < n_samples:
            accepted = self.metropolis_step(config)
            if accepted:
                samples.append(config.spins.copy())
                energies.append(self.get_energy_local(config))
            acceptances += int(accepted)
            total_steps += 1

        acceptance_rate = acceptances / total_steps
        return np.array(samples), np.array(energies), acceptance_rate


    def compute_gradients(self, samples, J=1.0, H=0.0):
        n_samples = len(samples)
    # Initialize accumulators
        da_avg = np.zeros_like(self.a)
        db_avg = np.zeros_like(self.b)
        dW_avg = np.zeros_like(self.W)

        E_local_list = []  # collect scalar energies
        O_list = []        # collect concatenated derivatives

    # Temporary configuration for energy evaluation
        temp_config = Configuration(self.N, 1.0, J, H)

        for spins in samples:
            temp_config.spins = spins
            E_local = float(self.local_energy(temp_config))  # ensure scalar
            da, db, dW = self.log_psi_derivs(spins)

        # Accumulate averages
            da_avg += da
            db_avg += db
            dW_avg += dW

            E_local_list.append(E_local)
            O_list.append(np.concatenate([da, db, dW.flatten()]))

    # Convert lists to arrays for vectorized mean
        E_local_array = np.array(E_local_list)  # shape (n_samples,)
        O_array = np.array(O_list)              # shape (n_samples, N+M+M*N)

    # Averages
        E_local_avg = np.mean(E_local_array)
        O_avg = np.mean(O_array, axis=0)

    # Compute gradients using covariance formula
        grad = 2 * np.mean((E_local_array[:, None] - E_local_avg) * (O_array - O_avg), axis=0)

    # Split gradients into a, b, W
        idx = 0
        grad_a = grad[idx:idx + self.N]
        idx += self.N
        grad_b = grad[idx:idx + self.M]
        idx += self.M
        grad_W = grad[idx:idx + self.M * self.N].reshape(self.M, self.N)

        return grad_a, grad_b, grad_W, E_local_avg


# The training of our model :

    def train(self, n_epochs, n_samples_per_epoch=1000, learning_rate=0.01, J=1.0, H=0.0, verbose=True):
        """Train the RBM using Stochastic Reconfiguration"""
        energies = []
        rel_errors = []

        #E_exact = exact_1d_ising_energy(self.N, J, H)
        E_exact = exact_1d_ising_energy(self.N, J, H)[0]


        for epoch in range(n_epochs):
            # Generate samples from current wavefunction
            samples, sample_energies, acceptance_rate = self.sample(n_samples_per_epoch, J=J, H=H)

            # Compute gradients and energy
            grad_a, grad_b, grad_W, energy = self.compute_gradients(samples, J, H)
            # Update parameters
            self.a -= learning_rate * grad_a
            self.b -= learning_rate * grad_b
            self.W -= learning_rate * grad_W

            # Compute relative error
            rel_error = compute_relative_error(energy, E_exact)

            energies.append(energy)
            rel_errors.append(rel_error)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:3d}: E_NQS = {energy:8.4f}, "f"E_exact = {E_exact:8.4f}, "f"ε_rel = {rel_error:8.6f}, ", f"Accept = {acceptance_rate:.3f}")

        return energies, rel_errors