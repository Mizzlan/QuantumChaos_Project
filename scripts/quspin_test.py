import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import time

def compute_r_stats(L, delta, h_defect):
    """
    Calculates the mean r-statistic for the XXZ model with a defect.
    Standard metric used in PRL/PRB papers for Quantum Chaos.
    """
    # 1. Basis construction (Zero-magnetization sector)
    basis = spin_basis_1d(L, m=0)
    
    # 2. Hamiltonian components (Periodic Boundary Conditions)
    # H = sum[ J(S+_i S-_{i+1} + h.c.) + Delta (Sz_i Sz_{i+1}) ] + h_0 Sz_0
    nn_list = [[1.0, i, (i+1)%L] for i in range(L)]
    zz_list = [[delta, i, (i+1)%L] for i in range(L)]
    defect  = [[h_defect, 0]]
    
    static = [
        ["+-", nn_list], 
        ["-+", nn_list], 
        ["zz", zz_list], 
        ["z",  defect]
    ]
    
    # 3. Diagonalization
    H = hamiltonian(static, [], basis=basis, dtype=np.float64, check_herm=False)
    energies = H.eigvalsh()
    
    # 4. R-statistic calculation
    # r_n = min(s_n, s_{n-1}) / max(s_n, s_{n-1}) where s_n is the level spacing
    spacings = np.diff(energies)
    r_n = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    
    return np.mean(r_n)

# --- Execution ---
L_values = [10, 12, 14,16] # N=16 will use your 16GB RAM effectively
results = []

print("Starting Journal-Grade Simulation...")
for L in L_values:
    start = time.time()
    r_avg = compute_r_stats(L, delta=1.0, h_defect=0.5)
    elapsed = time.time() - start
    results.append(r_avg)
    print(f"N={L} | <r>={r_avg:.4f} | Time: {elapsed:.2f}s")

# --- Plotting (Harvard/MIT Publication Style) ---
plt.style.use('seaborn-v0_8-paper') # Professional style
plt.figure(figsize=(6, 4))
plt.plot(L_values, results, 'o-', color='black', label='XXZ with Defect')
plt.axhline(y=0.536, color='r', linestyle='--', label='Wigner-Dyson (Chaos)')
plt.axhline(y=0.386, color='b', linestyle='--', label='Poisson (Integrable)')

plt.xlabel(r'System Size $L$', fontsize=12)
plt.ylabel(r'Mean $r$-statistic $\langle r \rangle$', fontsize=12)
plt.title('Transition to Chaos in XXZ Chain')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()