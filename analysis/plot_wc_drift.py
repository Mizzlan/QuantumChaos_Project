import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from src.plotting import set_publication_style


def plot_Wc_finite_size_drift():
    set_publication_style()
    DATA_PATH = "data/Wc_vs_L_drift.csv"
    OUT_PATH = "results/figures/Wc_scaling_drift_final_pppppt.pdf"

    df = pd.read_csv(DATA_PATH)
    plt.figure(figsize=(7,5))

    for J2 in sorted(df["J2"].unique()):
        sub = df[df["J2"] == J2].sort_values("L")
        x = 1.0 / sub["L"].values

        for key, marker, label in [
            ("Wc_r", "o", r"$r$-stat"),
            ("Wc_ipr", "s", "IPR"),
        ]:
            y = sub[key].values

            slope, intercept, r, p, stderr = linregress(x, y)

            plt.plot(x, y, marker, linestyle="none",
                     label=fr"{label}, $J_2={J2}$")
            xfit = np.linspace(0, max(x), 200)
            plt.plot(xfit, slope*xfit + intercept, "--", alpha=0.6)

            print(
                f"J2={J2}, {label}: "
                f"Wc(L→∞) = {intercept:.3f} ± {stderr:.3f}"
            )

    plt.xlabel(r"$1/L$")
    plt.ylabel(r"$W_c(L)$")
    plt.title("Finite-size drift and thermodynamic extrapolation of $W_c$")
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_PATH)
    plt.show()

if __name__ == "__main__":
    plot_Wc_finite_size_drift()
