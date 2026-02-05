import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.model_spinchain import build_xxz_chain
from src.spectrum import get_sorted_eigenvalues
from src.rstat import r_statistic

L_values = [8, 10, 12]
h = 1.0
n_realizations = 30

for L in L_values:
    r_vals = []
    for seed in range(n_realizations):
        H, _ = build_xxz_chain(
            L=L,
            h=h,
            seed=seed
        )
        E = get_sorted_eigenvalues(H)
        r_vals.append(r_statistic(E))

    print(f"L={L}  <r>={np.mean(r_vals):.4f}")
