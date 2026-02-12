import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting import set_publication_style

set_publication_style()

df = pd.read_csv("data/Wc_vs_L.csv")

plt.figure(figsize=(7,5))

for J2 in df["J2"].unique():
    sub = df[df["J2"] == J2]
    x = 1.0 / sub["L"]

    plt.plot(x, sub["Wc_r"], 'o-', label=fr"$r$-stat, $J_2={J2}$")
    plt.plot(x, sub["Wc_ipr"], 's--', label=fr"IPR, $J_2={J2}$")

plt.xlabel(r"$1/L$")
plt.ylabel(r"$W_c(L)$")
plt.title("Finite-size drift of MBL transition")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/figures/Wc_scaling_drift_11.png", dpi=300)
plt.show()
