import os, sys

# Ensure consistency with project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# -------------------- USER PARAMETERS --------------------
DATA_PATH = "results/data/research_run_L12_J20.5_N100_k250.csv"
R_STAR = 0.458  # Midpoint between GOE and Poisson
N_BOOT = 2000   # Bootstrap samples
# ---------------------------------------------------------

# -------------------- SCALING MODEL ----------------------
def linear_scaling(inv_L, W_inf, a):
    return W_inf + a * inv_L
# ---------------------------------------------------------

def find_Wc_from_r(W, r_vals, r_err):
    diff = r_vals - R_STAR
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    if len(idx) == 0:
        return None, None

    i = idx[0]
    x1, x2 = W[i], W[i+1]
    y1, y2 = r_vals[i], r_vals[i+1]

    Wc = x1 + (R_STAR - y1) * (x2 - x1) / (y2 - y1)

    slope = (y2 - y1) / (x2 - x1)
    sigma_r = np.mean(r_err[i:i+2])
    sigma_W = sigma_r / abs(slope) if slope != 0 else 0

    return Wc, sigma_W

# -------------------- BOOTSTRAP --------------------------
def bootstrap_fit(inv_L, Wc_vals, Wc_errs):
    W_inf_samples = []

    for _ in range(N_BOOT):
        noisy = np.random.normal(Wc_vals, Wc_errs)
        popt, _ = curve_fit(linear_scaling, inv_L, noisy)
        W_inf_samples.append(popt[0])

    return np.array(W_inf_samples)
# ---------------------------------------------------------

def reduced_chi_square(inv_L, Wc_vals, Wc_errs, popt):
    residuals = Wc_vals - linear_scaling(inv_L, *popt)
    chi_sq = np.sum((residuals / Wc_errs)**2)
    dof = len(Wc_vals) - len(popt)
    return chi_sq / dof
# ---------------------------------------------------------

def stability_test(inv_L, Wc_vals, Wc_errs):
    results = []

    for i in range(len(inv_L)):
        mask = np.ones(len(inv_L), dtype=bool)
        mask[i] = False

        popt, _ = curve_fit(
            linear_scaling,
            inv_L[mask],
            Wc_vals[mask],
            sigma=Wc_errs[mask],
            absolute_sigma=True
        )
        results.append(popt[0])

    return results
# ---------------------------------------------------------

def main():
    if not os.path.exists(DATA_PATH):
        print("Data file not found.")
        return

    df = pd.read_csv(DATA_PATH)
    os.makedirs("results/figures/test", exist_ok=True)

    for J2 in sorted(df["J2"].unique()):
        print("\n==============================")
        print(f"Processing J2 = {J2}")
        print("==============================")

        subJ = df[df["J2"] == J2]

        L_vals, Wc_vals, Wc_errs = [], [], []

        for L in sorted(subJ["L"].unique()):
            sub = subJ[subJ["L"] == L].sort_values("W")

            Wc, sigma = find_Wc_from_r(
                sub["W"].values,
                sub["r_mean"].values,
                sub["r_sem"].values
            )

            if Wc is not None:
                L_vals.append(L)
                Wc_vals.append(Wc)
                Wc_errs.append(sigma)

        if len(L_vals) < 3:
            print("Not enough points for scaling.")
            continue

        inv_L = 1.0 / np.array(L_vals)
        Wc_vals = np.array(Wc_vals)
        Wc_errs = np.array(Wc_errs)

        # -------- Weighted Fit --------
        popt, pcov = curve_fit(
            linear_scaling,
            inv_L,
            Wc_vals,
            sigma=Wc_errs,
            absolute_sigma=True
        )

        W_inf, slope = popt
        W_inf_err = np.sqrt(pcov[0, 0])

        # -------- Chi-Square --------
        chi2_red = reduced_chi_square(inv_L, Wc_vals, Wc_errs, popt)

        # -------- Bootstrap --------
        boot = bootstrap_fit(inv_L, Wc_vals, Wc_errs)
        W_inf_boot_mean = np.mean(boot)
        W_inf_boot_std = np.std(boot)

        # -------- Stability --------
        stability = stability_test(inv_L, Wc_vals, Wc_errs)

        # -------- Print Diagnostics --------
        print(f"Wc(inf) weighted fit = {W_inf:.3f} ± {W_inf_err:.3f}")
        print(f"Wc(inf) bootstrap    = {W_inf_boot_mean:.3f} ± {W_inf_boot_std:.3f}")
        print(f"Reduced chi^2        = {chi2_red:.3f}")
        print(f"Leave-one-out W_inf  = {[round(x,3) for x in stability]}")

        # -------- Plot --------
        plt.figure(figsize=(7,6))

        plt.errorbar(
            inv_L,
            Wc_vals,
            yerr=Wc_errs,
            fmt='o',
            color='black',
            capsize=4,
            label="Extracted $W_c(L)$"
        )

        x_fit = np.linspace(0, max(inv_L)*1.1, 200)
        plt.plot(
            x_fit,
            linear_scaling(x_fit, *popt),
            'r--',
            label=fr"Fit: $W_c(\infty)={W_inf:.2f}\pm{W_inf_err:.2f}$"
        )

        plt.xlabel(r"$1/L$")
        plt.ylabel(r"$W_c(L)$")
        plt.title(fr"Finite-Size Scaling ($J_2={J2}$)")
        plt.legend(frameon=False)
        plt.tight_layout()

        plt.savefig(f"results/figures/test/Wc_scaling_J2_{J2}.pdf")
        plt.show()

# ---------------------------------------------------------

if __name__ == "__main__":
    main()
