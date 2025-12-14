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


# Pauli matrices
sx = sp.csr_matrix(np.array([[0,1],[1,0]], dtype=complex))
sz = sp.csr_matrix(np.array([[1,0],[0,-1]], dtype=complex))
id2 = sp.identity(2, format='csr', dtype=complex)

def kron_n(ops):
    """Construct the tensor (Kronecker) product of a list of local operators.

    Each operator in the list 'ops' acts on a single site (spin) of the chain.
    This function combines them into a global operator acting on the full
    Hilbert space (dimension 2^L for L spins)."""

    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format='csr')
    return out

# Build TFIM Hamiltonian
def build_tfim(L, J=1.0, Gamma=1.0, periodic=True):
    H = sp.csr_matrix((2**L, 2**L), dtype=complex)

    # Interaction term -J Σ σ^z_j σ^z_{j+1}
    for j in range(L):
        jp = (j+1) % L
        if not periodic and jp == 0:
            continue
        ops = []
        for site in range(L):
            if site == j or site == jp:
                ops.append(sz)
            else:
                ops.append(id2)
        H -= J * kron_n(ops)

    # Transverse field term -Γ Σ σ^x_j
    for j in range(L):
        ops = []
        for site in range(L):
            ops.append(sx if site == j else id2)
        H -= Gamma * kron_n(ops)

    return H

# Compute ground state
def exact_1d_ising_energy(L, J=1.0, Gamma=1.0, periodic=True):
    H = build_tfim(L, J, Gamma, periodic)
    dim = 2**L

    # If small enough: exact dense diagonalization
    if dim <= 2000:
        H_dense = H.toarray()
        evals, evecs = np.linalg.eigh(H_dense)
        return evals[0], evecs[:,0]
    else:
        eval, vec = spla.eigsh(H, k=1, which='SA')
        return eval[0].real, vec[:,0]

# Relative error

def compute_relative_error(E_nqs, E_exact):
    return (E_nqs - E_exact) / np.abs(E_exact)
