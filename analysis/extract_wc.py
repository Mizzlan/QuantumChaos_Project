import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from math import comb
import os, sys

# Ensure consistency with project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------- PARAMETERS --------------------
# Using your established master production file
INPUT_FILE = "data/master_fss_data_L[8,10,12,14]_J2[0.0,0.5]_NE50.csv"
J2_VALUES = [0.0, 0.5]

# Theoretical RMT benchmarks for r-statistic
R_GOE = 0.5307
R_POISSON = 0.386
# PHYSICS: The 'midpoint' where the system is 50% chaotic / 50% localized
R_STAR = 0.5 * (R_GOE + R_POISSON) 
# ----------------------------------------------------

def hilbert_dim(L):
    """Calculates dimensions of the Sz=0 sector (Half-filling)."""
    return float(comb(L, L // 2))

def run_wc_extraction():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    results = []

    for J2 in J2_VALUES:
        df_J = df[df["J2"] == J2]

        for L in sorted(df_J["L"].unique()):
            # PHYSICS: Sorting by W is required for valid interpolation/differentiation
            sub = df_J[df_J["L"] == L].sort_values("W")
            W = sub["W"].values
            r = sub["r_mean"].values

            # --- METHOD 1: r-statistic Crossover ---
            # Using linear interpolation to find exactly where r hits R_STAR
            f_r = interp1d(W, r, kind="linear", bounds_error=False)
            W_dense = np.linspace(W.min(), W.max(), 500)
            r_dense = f_r(W_dense)
            
            # Find index where the difference from R_STAR is minimized
            idx_r = np.argmin(np.abs(r_dense - R_STAR))
            Wc_r = W_dense[idx_r]

            # --- METHOD 2: IPR Inflection Point ---
            # PHYSICS: The transition is where the wavefunction structure changes fastest.
            D = hilbert_dim(L)
            scaled_ipr = sub["ipr_mean"].values * D
            
            # MATH: Savitzky-Golay smoothing to remove noise from NE=50 realizations
            # Note: window_length must be less than the number of data points
            window = 5 if len(W) > 5 else 3
            ipr_smooth = savgol_filter(scaled_ipr, window_length=window, polyorder=2)
            
            # MATH: Calculate the gradient (slope) to find the maximum change
            dipr_dW = np.gradient(ipr_smooth, W)
            idx_ipr = np.argmax(np.abs(dipr_dW))
            Wc_ipr = W[idx_ipr]

            results.append({
                "J2": J2, "L": L, 
                "inv_L": 1.0/L, # Pre-calculating for the scaling plot
                "Wc_r": Wc_r, 
                "Wc_ipr": Wc_ipr
            })

    # Save results to a clean CSV for the plotter
    os.makedirs("data", exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv("data/Wc_vs_L_drift.csv", index=False)
    print("✅ Wc(L) extraction complete. Saved to data/Wc_vs_L_drift.csv")

if __name__ == "__main__":
    run_wc_extraction()