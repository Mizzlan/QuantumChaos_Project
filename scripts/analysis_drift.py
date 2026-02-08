import sys
import os

# Adds the parent directory (project root) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.finite_size import extract_wc, extract_vL
from src.model_spinchain import NNN, jxy, ising, nrealise
from src.spectrum import frac
# --- PARAMETERS (Keep consistent with simulation) ---
L_list = [8, 10, 12]


drift_results = []

for L in L_list:
    # Construct filename based on your simulation naming convention
    fname = f"results/data/resultsipr_L{L}_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}.csv"
    
    # Load using pandas for better column handling
    df = pd.read_csv(fname)
    
    W = df["W"].values
    r = df["mean_r"].values
    
    # Extract Wc(L) where r is roughly halfway between GOE (0.53) and Poisson (0.39)
    # 0.46 is the common 'crossing' midpoint for MBL transitions
    Wc = extract_wc(W, r, r_mid=0.46)
    vL = extract_vL(W, r, Wc)
    
    print(f"L={L}: Extracted Wc = {Wc:.3f}, Slope v(L) = {vL:.3f}")
    
    drift_results.append({"L": L, "Wc": Wc, "vL": vL})

# --- DRIFT ANALYSIS (Journal Standard) ---
# Plotting Wc vs 1/L to see the thermodynamic limit
dr_df = pd.DataFrame(drift_results)
plt.plot(1/dr_df["L"], dr_df["Wc"], 'o-', label="Drift of $W_c$")
plt.xlabel("1/L")
plt.ylabel("$W_c(L)$")
plt.title("Scaling Drift Analysis")
plt.grid(True)
plt.savefig("results/figures/wc_drift_analysis.png")