import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.spectrum import get_middle_spectrum
from src.rstat import r_statistic

data_dir = "data/spectra"
out_dir = "data/rstat"
os.makedirs(out_dir, exist_ok=True)

L_list = [8, 10, 12, 14]
J2_values = [0.0, 0.5]

for L in L_list:
    for J2 in J2_values:
        rows = []

        files = [f for f in os.listdir(data_dir)
                 if f.startswith(f"L{L}_") and f"_J2{J2}_" in f]

        grouped = {}
        for f in files:
            W = float(f.split("_W")[1].split("_")[0])
            grouped.setdefault(W, []).append(f)

        for W, flist in grouped.items():
            r_vals = []

            for fname in flist:
                data = np.load(os.path.join(data_dir, fname))
                evals = data["evals"]

                E_mid = get_middle_spectrum(evals)
                r_vals.append(r_statistic(E_mid))

            rows.append({
                "L": L,
                "J2": J2,
                "W": W,
                "mean_r": np.mean(r_vals),
                "sem_r": np.std(r_vals, ddof=1)/np.sqrt(len(r_vals)),
                "n_valid": len(r_vals)
            })

        df = pd.DataFrame(rows).sort_values("W")
        df.to_csv(f"{out_dir}/rstat_L{L}_J2{J2}.csv", index=False)
        print(f"Saved r-stat for L={L}, J2={J2}")
