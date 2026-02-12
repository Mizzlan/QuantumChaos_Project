import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from src.plotting import set_publication_style

# ===================== PARAMETERS =====================
INPUT_FILE = "data/master_fss_data_L[8,10,12,14]_J2[0.0,0.5]_NE50.csv"
OUTPUT_DATA = "data/processed_scaled_ipr.csv"
FIG_DIR = "results/figures"

J2_VALUES = [0.0, 0.5]
W_ETH = 1                 # Weak disorder point for ETH scaling check
ETH_IPR_CONST = 3.0          # GOE prediction for real eigenvectors
# =====================================================


def hilbert_dim(L):
    """
    Hilbert space dimension in the Sz = 0 (half-filling) sector.
    Cast to float to avoid overflow in future larger-L runs.
    """
    assert L % 2 == 0, "L must be even for Sz=0 sector."
    return float(comb(L, L // 2))


# ===================== STYLE =====================
set_publication_style()
os.makedirs(FIG_DIR, exist_ok=True)


# ===================== LOAD DATA =====================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

processed_rows = []


# =====================================================
# 1️⃣ MAIN IPR × D PLOTS (ETH vs MBL)
# =====================================================
for J2 in J2_VALUES:
    plt.figure(figsize=(7, 5))
    subJ = df[df["J2"] == J2]

    for L in sorted(subJ["L"].unique()):
        sub = subJ[subJ["L"] == L].sort_values("W")

        D = hilbert_dim(L)
        scaled_ipr = sub["ipr_mean"] * D
        scaled_err = sub["ipr_sem"] * D

        plt.errorbar(
            sub["W"], scaled_ipr,
            yerr=scaled_err,
            marker="o",
            markersize=4,
            capsize=2,
            label=f"L={L}"
        )

        # Save processed data
        for w, sipr in zip(sub["W"], scaled_ipr):
            processed_rows.append({
                "L": L,
                "W": w,
                "J2": J2,
                "scaled_ipr": sipr,
                "hilbert_dim": D
            })

    # ---- ETH reference line (GOE prediction) ----
    plt.axhline(
        ETH_IPR_CONST,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=r"GOE prediction ($\approx 3$)"
    )

    plt.yscale("log")
    plt.xlabel(r"Disorder strength $W$")
    plt.ylabel(r"$\langle \mathrm{IPR} \rangle \times \mathcal{D}$")
    plt.title(rf"Eigenstate ETH Test via IPR ($J_2={J2}$)")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/ipr_eth_test_J2_{J2}.png", dpi=300)
    plt.show()

    print(f"✅ Saved ETH–IPR plot for J2 = {J2}")


# =====================================================
# 2️⃣ ETH SIZE-SCALING CHECK (FIXED W)
# =====================================================
plt.figure(figsize=(6, 4))

for J2 in J2_VALUES:
    sub = df[(df["J2"] == J2) & (df["W"] == W_ETH)]

    L_vals = sorted(sub["L"].unique())
    scaled_ipr_vals = [
        (sub[sub["L"] == L]["ipr_mean"].values[0]) * hilbert_dim(L)
        for L in L_vals
    ]

    plt.plot(L_vals, scaled_ipr_vals, "o-", label=f"$J_2={J2}$")

plt.axhline(
    ETH_IPR_CONST,
    color="gray",
    linestyle="--",
    alpha=0.6,
    label=r"GOE prediction"
)

plt.xlabel(r"System size $L$")
plt.ylabel(r"$\langle \mathrm{IPR} \rangle \times \mathcal{D}$")
plt.title(f"ETH Scaling Check at Weak Disorder (W={W_ETH})")
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(f"{FIG_DIR}/ipr_eth_scaling_check_w{W_ETH}.png", dpi=300)
plt.show()

print("✅ Saved ETH size-scaling check")


# =====================================================
# 3️⃣ EXPORT PROCESSED DATA
# =====================================================
processed_df = pd.DataFrame(processed_rows)
processed_df.to_csv(OUTPUT_DATA, index=False)

print(f"✅ Exported processed IPR data to {OUTPUT_DATA}")
