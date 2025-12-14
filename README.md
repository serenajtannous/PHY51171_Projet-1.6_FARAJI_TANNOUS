# PHY51171_Projet-1.6_FARAJI_TANNOUS
Neural networks for quantum systems

# Neural Networks for Quantum Systems: 1D Ising Model

## Project description
This project implements a Neural-Network Quantum State (NQS) ansatz using a **Restricted Boltzmann Machine (RBM)** to find the ground state of the **1D Transverse-Field Ising Model**.

The wavefunction optimization is performed using **Variational Monte Carlo (VMC)**. Two optimization strategies are implemented and compared:
1.  **Standard Gradient Descent (GD)**
2.  **Stochastic Reconfiguration (SR)**, which accounts for the geometry of the variational manifold.

The project validates results against Exact Diagonalization for small systems (N=10) and uses the **V-score** metric to validate large-scale simulations (N=50) where exact methods are intractable.

## Usage
To train the RBM using Stochastic Reconfiguration (recommended method), run the main script:

    python3 RBM_SR.py

To run the Gradient Descent version for comparison:

    python3 RBM_GD.py

To generate the convergence plots and heatmap analysis used in the report, run the Jupyter notebooks located in the `analysis` directory.

## Directory structure

* **Configuration**
      * `Ising.py`: Defines the spin configuration, Hamiltonian, and the classical Metropolis sampler for initialization

*  **Production**
    * `RBM_GD.py`: Implements the RBM ansatz optimized via standard Gradient Descent.
    * `RBM_SR.py`: Implements the RBM ansatz optimized via Stochastic Reconfiguration.
    * `Exact_Energy.py`: Computes the exact ground state energy using sparse diagonalization.

* **analysis**
    * Contains Jupyter notebooks used to analyze hyperparameters (alpha, N_samples, LR) and generate the plots. All the results are included.

* **figures**
    * Contains the generated plots used in the report (e.g., convergence curves, V-score scaling).

* **report**
    * `Report_EA.pdf`: The final project report detailing the theoretical background, implementation, and results.
