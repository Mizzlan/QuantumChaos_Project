import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.model_spinchain import xxz_chain
from src.spectrum import get_middle_spectrum
from src.rstat import r_statistic

# ------------------------
# PHYSICS PARAMETERS
# ------------------------
L_list = [14]
W_values = np.linspace(0.1, 6.0, 15)
n_realizations = 50

# Numerical thresholds
EPS = 1e-8
MIN_FRACTION = 0.3 # minimum number of valid r-values per W

for L in L_list:
    print(f"\nRunning L={L}")
    results = []

    for W in W_values:
        r_vals = []

        for seed in range(n_realizations):
            H = xxz_chain(L=L, Delta=1.0, W=W, J2=0.5, seed=seed)
            evals = H.eigvalsh()
            E_mid = get_middle_spectrum(evals, fraction=0.5)

            r = r_statistic(E_mid, eps=EPS)

            if not np.isnan(r):
                r_vals.append(r)

        # --------- STATISTICS ----------
        n_valid = len(r_vals)
        if n_valid < MIN_FRACTION * n_realizations:
            mean_r = np.nan
            sem_r = np.nan
        else:
            mean_r = np.mean(r_vals)
            sem_r = np.std(r_vals, ddof=1) / np.sqrt(n_valid)
        


        results.append({
            "L": L,
            "W": W,
            "mean_r": mean_r,
            "sem_r": sem_r,
            "n_valid": len(r_vals),
            "n_valid": n_valid
        })

    df = pd.DataFrame(results)
    df.to_csv(f"results/data/results_L{L}.csv", index=False)
    print(f"Saved results_L{L}.csv")
