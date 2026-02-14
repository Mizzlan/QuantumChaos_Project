import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from model_spinchain00 import build_xxz_chain
from src.spectrum import get_sorted_eigenvalues
from src.rstat import r_statistic

L = 10
J2_values = np.linspace(0.0, 1.0, 11)
r_J2 = []

for J2 in J2_values:
    H, _ = build_xxz_chain(
        L=L,
        h=None,
        J2=J2
    )
    E = get_sorted_eigenvalues(H)
    r_J2.append(r_statistic(E))

for J2, r in zip(J2_values, r_J2):
    print(f"J2={J2:.2f}  r={r:.4f}")
