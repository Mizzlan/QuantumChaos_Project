import os, sys

# Ensure consistency with project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------- PARAMETERS --------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------
# USER INPUT
# ----------------------------
DATA_PATH = "data/master_fss_data_L[8,10,12,14]_J2[0.0,0.5]_NE50.csv"
GOE_VALUE = 0.5307  # GOE <r> value
OUTPUT_TABLE = "results/Wc_r_extrapolated_test.csv"

# ----------------------------
# Linear scaling ansatz
# ----------------------------
def linear_scaling(inv_L, W_inf, a):
    return W_inf + a * inv_L


def find_Wc_from_r(W, r_vals, r_err):
    """
    Extract Wc by interpolation where r crosses GOE value.
    Returns (Wc, estimated_error)
    """

    # Find closest crossing
    diff = r_vals - GOE_VALUE
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    if len(idx) == 0:
        return None, None

    i = idx[0]

    # Linear interpolation
    x1, x2 = W[i], W[i+1]
    y1, y2 = r_vals[i], r_vals[i+1]

    Wc = x1 + (GOE_VALUE - y1) * (x2 - x1) / (y2 - y1)

    # Error estimate from slope
    slope = (y2 - y1) / (x2 - x1)
    sigma_r = np.mean(r_err[i:i+2])
    sigma_W = sigma_r / abs(slope)

    return Wc, sigma_W


def main():

    if not os.path.exists(DATA_PATH):
        print("Data file not found.")
        return

    df = pd.read_csv(DATA_PATH)

    os.makedirs("results/figures", exist_ok=True)

    final_results = []

    for J2 in sorted(df["J2"].unique()):

        subJ = df[df["J2"] == J2]

        L_vals = []
        Wc_vals = []
        Wc_errs = []

        for L in sorted(subJ["L"].unique()):
            sub = subJ[subJ["L"] == L].sort_values("W")

            W = sub["W"].values
            r_vals = sub["r_mean"].values
            r_err = sub["r_sem"].values

            Wc, sigma = find_Wc_from_r(W, r_vals, r_err)

            if Wc is not None:
                L_vals.append(L)
                Wc_vals.append(Wc)
                Wc_errs.append(sigma)

        L_vals = np.array(L_vals)
        inv_L = 1.0 / L_vals
        Wc_vals = np.array(Wc_vals)
        Wc_errs = np.array(Wc_errs)

        # ----------------------------
        # Weighted linear regression
        # ----------------------------
        popt, pcov = curve_fit(
            linear_scaling,
            inv_L,
            Wc_vals,
            sigma=Wc_errs,
            absolute_sigma=True
        )

        W_inf, slope = popt
        W_inf_err = np.sqrt(pcov[0, 0])

        print(f"\nJ2 = {J2}")
        print(f"Wc(∞) = {W_inf:.3f} ± {W_inf_err:.3f}")

        final_results.append({
            "J2": J2,
            "Wc_infinite": W_inf,
            "Wc_error": W_inf_err
        })

        # ----------------------------
        # Plot
        # ----------------------------
        plt.figure(figsize=(6,5))

        plt.errorbar(inv_L, Wc_vals, yerr=Wc_errs,
                     fmt='o', capsize=3, label="Data")

        x_fit = np.linspace(0, max(inv_L)*1.1, 200)
        plt.plot(x_fit, linear_scaling(x_fit, *popt),
                 '--', label="Weighted fit")

        plt.xlabel(r"$1/L$")
        plt.ylabel(r"$W_c(L)$")
        plt.title(fr"$r$-stat Scaling, $J_2={J2}$")
        plt.legend(frameon=False)
        plt.tight_layout()

        plt.savefig(f"results/figures/Wc_r_scaling_J2_{J2}.pdf")
        plt.show()

    # Save extrapolated values
    pd.DataFrame(final_results).to_csv(OUTPUT_TABLE, index=False)
    print("\nSaved extrapolated results.")


if __name__ == "__main__":
    main()
