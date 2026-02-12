import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import comb

df = pd.read_csv("data/master_fss_data_L[8,10,12,14]_J2[0.0,0.5]_NE50.csv")

def hilbert_dim(L):
    # Sz=0 sector
    return comb(L, L//2)

for J2 in [0.0, 0.5]:
    plt.figure(figsize=(7,5))

    subJ = df[df["J2"] == J2]

    for L in sorted(subJ["L"].unique()):
        sub = subJ[subJ["L"] == L].sort_values("W")

        scaled_ipr = sub["ipr_mean"] * hilbert_dim(L)

        plt.plot(sub["W"], scaled_ipr, marker='o', label=f"L={L}")

    plt.xlabel("Disorder strength $W$")
    plt.ylabel(r"$\mathrm{IPR} \times \dim(\mathcal{H})$")
    plt.title(f"ETH Test via IPR ($J_2={J2}$)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"results/figures/ipr_eth_J2_{J2}.pdf")
    plt.show()
