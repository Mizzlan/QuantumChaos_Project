import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.model_spinchain import xxz_chain
from src.model_spinchain import NNN, jxy, ising, nrealise
from src.spectrum import get_middle_spectrum, frac
from src.rstat import r_statistic
from src.ipr import disorder_averaged_ipr

# ------------------------
# PHYSICS PARAMETERS
# ------------------------
L_list = [8,10,12]
W_values = np.linspace(0.1, 6.0, 15)
n_realizations = nrealise

# Numerical thresholds
EPS = 1e-8
MIN_FRACTION = 0.3 # minimum number of valid r-values per W

for L in L_list:
    print(f"\nRunning L={L}")
    results = []

    for W in W_values:
        r_vals = []
        ipr_list = []

        for seed in range(n_realizations):
            H = xxz_chain(L=L, Delta=ising, W=W, J2=NNN, seed=seed)
            evals, evecs = H.eigh()
            E_mid = get_middle_spectrum(evals, fraction=frac)

            r = r_statistic(E_mid, eps=EPS)
            
            ipr_val = disorder_averaged_ipr(evals,evecs)
            ipr_list.append(ipr_val)

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
        
        mean_ipr = np.mean(ipr_list)
        sem_ipr = np.std(ipr_list, ddof=1) / np.sqrt(len(ipr_list))


        results.append({
            "L": L,
            "W": W,
            "mean_r": mean_r,
            "sem_r": sem_r,
            "mean_ipr": mean_ipr,
            "sem_ipr": sem_ipr,
            "n_valid": len(r_vals),
            "n_valid": n_valid
        })

    df = pd.DataFrame(results)
    df.to_csv(f"results/data/resultsipr_L{L}_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}.csv", index=False)
    print(f"Saved results_L{L}.csv")
