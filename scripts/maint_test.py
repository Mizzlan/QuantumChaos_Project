import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

def get_chaotic_xxz_levels(L, J=1.0, delta=1.0, h_defect=0.5):
    si, sx, sy, sz = qt.qeye(2), 0.5*qt.sigmax(), 0.5*qt.sigmay(), 0.5*qt.sigmaz()
    H = 0
    # Nearest-neighbor interactions
    for i in range(L - 1):
        for op in [sx, sy]:
            ops = [si]*L; ops[i], ops[i+1] = op, op
            H += J * qt.tensor(ops)
        ops = [si]*L; ops[i], ops[i+1] = sz, sz
        H += delta * qt.tensor(ops)
    
    # Add defect at site 0 to break symmetry and cause chaos
    ops = [si]*L; ops[0] = sz
    H += h_defect * qt.tensor(ops)
    
    # Extract eigenvalues (energies)
    return eigvalsh(H.full())

# --- MAIN ANALYSIS ---
L = 12  # Matrix size: 4096 x 4096
print(f"Starting analysis for L={L}...")
energies = get_chaotic_xxz_levels(L)

# 1. Truncate the spectrum (we usually ignore the very top and bottom 10%)
energies = np.sort(energies)
trim = int(len(energies) * 0.1)
energies = energies[trim:-trim]

# 2. Unfold the spectrum using a polynomial fit
# This transforms energies into a uniform distribution
fit = np.polyval(np.polyfit(energies, np.linspace(0, 1, len(energies)), 10), energies)
spacings = np.diff(fit)
s = spacings / np.mean(spacings) # Normalized spacings

# 3. Plotting
plt.hist(s, bins=40, density=True, alpha=0.6, color='navy', label='XXZ Defect Model')
x = np.linspace(0, 4, 100)
plt.plot(x, (np.pi*x/2)*np.exp(-np.pi*x**2/4), 'r--', label='Wigner-Dyson (Chaos)')
plt.plot(x, np.exp(-x), 'g:', label='Poisson (Non-Chaos)')
plt.xlabel("s (Normalized Spacing)")
plt.ylabel("P(s)")
plt.legend()
plt.title(f"Level Spacing Statistics for L={L}")
plt.show()