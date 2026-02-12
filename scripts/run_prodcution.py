import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
import pandas as pd

from src.model_spinchain import xxz_chain
from src.rstat import r_statistic
from src.ipr import mean_ipr
from src.spectrum import mid_spectrum_indices

# -------------------- PARAMETERS --------------------
L_list = [14]            # Run ONLY L=14 now
W_values = np.linspace(0.5, 8.0, 16)
J2 = 0.0                 # Change to 0.0 or 0.5 as needed
N_E = 50                 # Number of realizations
fraction = 0.2           # Energy window
J_val = 1.0              # J strength
Delta_val = 1.0          # Delta strength
# ---------------------------------------------------

filename = (f"data/ipr_rstat_L{L_list}_J2{J2}_W{W_values.min()}-{W_values.max()}_"
            f"D{Delta_val}_Frac{fraction}_NE{N_E}.csv")

os.makedirs("data", exist_ok=True)


results_list = []
print(f"\n--- Starting Simulation: J2 = {J2} ---")
    
for L in L_list:
        for W in W_values:
            r_vals = []
            ipr_vals = []

            for seed in tqdm(range(N_E), desc=f"L={L}, W={W:.2f}"):
                # Build Hamiltonian
                H = xxz_chain(L=L, Delta=1.0, W=W, J2=J2, seed=seed)

                # Diagonalize (Eigh returns both evals and evecs)
                evals, evecs = H.eigh()

                # Get middle 20% indices
                idx = mid_spectrum_indices(evals, fraction=fraction)

                # Spectral Chaos (r-statistic)
                r_vals.append(r_statistic(evals[idx]))

                # Spatial Localization (IPR)
                # Slicing the matrix here saves memory before the math operation
                ipr_vals.append(mean_ipr(evecs[:, idx]))

                # CRITICAL: Free memory for L=14
                del evecs

            results_list.append({
                "L": L,
                "W": W,
                "J2": J2,
                "r_mean": np.mean(r_vals),
                "r_sem": np.std(r_vals, ddof=1) / np.sqrt(N_E),
                "ipr_mean": np.mean(ipr_vals),
                "ipr_sem": np.std(ipr_vals, ddof=1) / np.sqrt(N_E)
            })

    # Save each J2 case to a separate CSV for easy comparison
df = pd.DataFrame(results_list)
df.to_csv(filename, index=False)
print(f"\n✅ Data saved to: {filename}")    