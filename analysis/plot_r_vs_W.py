import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt

# Load master CSV
df = pd.read_csv("data/master_fss_data_L[8,10,12,14]_J2[0.0,0.5]_NE50.csv")

for J2 in [0.0, 0.5]:
    plt.figure(figsize=(7,5))

    subset_J2 = df[df["J2"] == J2]

    for L in sorted(subset_J2["L"].unique()):
        sub = subset_J2[subset_J2["L"] == L].sort_values("W")

        plt.errorbar(
            sub["W"], sub["r_mean"], yerr=sub["r_sem"],
            marker='o', capsize=3, label=f"L={L}"
        )

    # RMT benchmarks
    plt.axhline(0.536, ls="--", color="red", label="GOE")
    plt.axhline(0.386, ls=":", color="blue", label="Poisson")

    plt.xlabel("Disorder strength $W$")
    plt.ylabel(r"$\langle r \rangle$")
    plt.title(f"Spectral Statistics ($J_2={J2}$)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"results/figures/r_vs_W_J2_{J2}.pdf")
    plt.show()
