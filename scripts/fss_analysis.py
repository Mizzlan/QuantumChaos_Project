import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from model_spinchain00 import NNN, jxy, ising, nrealise
from src.spectrum import frac

def run_fss_plotting():
    # 1. LOAD DATA: Ensure path matches your stable architecture
    data_path = f"results/data/fss_master_data_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}.csv"
    
    if not os.path.exists(data_path):
        print(f"CRITICAL ERROR: {data_path} not found!")
        return

    df = pd.read_csv(data_path)
    
    # Check if the dataframe actually has data
    if df.empty:
        print("ERROR: The CSV file is empty. Check your data_merger script.")
        return

    # -------------------------------
    # 1. Standard r vs W plot
    # # -------------------------------
    # plt.figure(figsize=(8, 5))
    # plt.axhline(0.536, ls="--", color='r', label="GOE") # Chaos limit
    # plt.axhline(0.386, ls=":", color='b', label="Poisson") # MBL limit

    # for L in sorted(df["L"].unique()):
    #     sub = df[df["L"] == L].sort_values("W")
    #     # Ensure we use error bars so the points are visible even if sparse
    #     plt.errorbar(
    #         sub["W"], sub["mean_r"], yerr=sub["sem_r"],
    #         marker="o", capsize=3, label=f"L={L}", markersize=4
    #     )

    # plt.xlabel(r"Disorder strength $W$")
    # plt.ylabel(r"Mean gap ratio $\langle r \rangle$")
    # plt.legend()
    # plt.grid(alpha=0.3)
    # # AUTOSCALE: Ensure the plot shows all W values from 0.1 to 6.0
    # plt.autoscale() 
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # filename = f"results/figures/r_vs_W_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}_{timestamp}.png"     
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Figure saved as: {filename}")                               
    # plt.show()
    # plt.close()
    # Important: Closes the plot memory so your RAM doesn't fill up

    # -------------------------------
    # 2. Scaling Collapse
    # -------------------------------
    Wc = 3.71111
    nu = 1.0

    plt.figure(figsize=(8, 5))
    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L].sort_values("W")
        # Scaling formula: collapses system sizes to a single curve
        x_scaled = (L ** (1 / nu)) * (sub["W"] - Wc)
        plt.plot(x_scaled, sub["mean_r"], "o-", label=f"L={L}", markersize=4)

    plt.xlabel(rf"$L^{{1/\nu}}(W - W_c)$")
    plt.ylabel(r"$\langle r \rangle$")
    plt.title(f"Scaling Collapse ($W_c={Wc}$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.autoscale() # Reset zoom to show all scaled data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"results/figures/r_vs_scale_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}_wc_{Wc}.png"     
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {filename}")                               
    plt.show()
    plt.close() # Important: Closes the plot memory so your RAM doesn't fill up




if __name__ == "__main__":
    run_fss_plotting()
    