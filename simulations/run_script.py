import sys
import os
import time
import json
import platform
import subprocess
import numpy as np
import pandas as pd
import scipy
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

# ---- Project imports ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_spinchain import xxz_chain
from src.rstat import r_statistic
from src.ipr import mean_ipr


# =========================
# ===== PARAMETERS ========
# =========================

L_list = [16]          # change to [14,16,18] when scaling
W_values = np.linspace(0.5, 8.0, 16)

J_val = 1.0
Delta_val = 1.0
J2 = 0.0

L_params = {
    8:  {"N_real": 200, "k_eigs": 70},
    10: {"N_real": 150, "k_eigs": 150},
    12: {"N_real": 100, "k_eigs": 250},
    14: {"N_real": 60,  "k_eigs": 300},
    16: {"N_real": 30,  "k_eigs": 200},
    18: {"N_real": 10,  "k_eigs": 150},
}

for L in L_list:
    N_real = L_params[L]["N_real"]
    k_eigs = L_params[L]["k_eigs"]
         # safe for L=18
window = 80

ARPACK_TOL = 1e-10
MAX_ITER = 5000

if L_list[0] >= 18:
    n_jobs = 1
elif L_list[0] >= 16:
    n_jobs = 2 
else:
    n_jobs = -1  
        # use all cores
base_seed = 12345

t_vals = np.linspace(0, 10, 50)

os.makedirs("results/data", exist_ok=True)
os.makedirs("results/metadata", exist_ok=True)
os.makedirs("results/sff", exist_ok=True)


# =========================
# ===== UTILITIES =========
# =========================

def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("ascii").strip()
    except:
        return "not_available"


def spectral_form_factor(evals, t_vals, avg_spacing):
    """
    Compute unfolded Spectral Form Factor.

    Parameters
    ----------
    evals : array
        Eigenvalues in spectral window.
    t_vals : array
        Physical time values.
    avg_spacing : float
        Mean level spacing in window.

    Returns
    -------
    tau_vals : array
        Dimensionless unfolded time.
    K_tau : array
        Normalized spectral form factor.
    """

    # ----- 1. Center spectrum -----
    evals = evals - np.mean(evals)

    # ----- 2. Unfold time -----
    # Heisenberg time: t_H = 2π / Δ
    t_H = 2 * np.pi / avg_spacing
    tau_vals = t_vals / t_H

    # ----- 3. Partition function -----
    Z_t = np.array([
        np.sum(np.exp(-1j * t * evals))
        for t in t_vals
    ])

    # ----- 4. Normalize by Hilbert dimension -----
    dim = len(evals)
    K_tau = (np.abs(Z_t) ** 2) / dim

    return tau_vals, K_tau



# =========================
# ==== CORE COMPUTE =======
# =========================

def compute_one_realization(L, W, seed):

    np.random.seed(seed)

    H = xxz_chain(L=L, Delta=Delta_val, W=W, J2=J2, seed=seed)
    dim = H.basis.Ns

    # ----- Diagonalization -----
    if k_eigs >= dim - 2:
        diag_mode = "dense"
        evals, evecs = H.eigh()
        arpack_converged = True
        nconv = dim
    else:
        diag_mode = "sparse"
        H_sparse = H.tocsr()

        try:
            evals, evecs = eigsh(
                H_sparse,
                k=k_eigs,
                sigma=1e-6,
                which='LM',
                tol=ARPACK_TOL,
                maxiter=MAX_ITER
            )
            arpack_converged = True
            nconv = k_eigs
        except ArpackNoConvergence as e:
            evals = e.eigenvalues
            evecs = e.eigenvectors
            arpack_converged = False
            nconv = len(evals)

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # ----- Spectral window -----
    mid = len(evals) // 2
    start = max(0, mid - window // 2)
    end = min(len(evals), mid + window // 2)

    evals_mid = evals[start:end]
    evecs_mid = evecs[:, start:end]

    # ----- Observables -----
    r = r_statistic(evals_mid)
    ipr = mean_ipr(evecs_mid)

    level_spacings = np.diff(evals_mid)
    avg_spacing = np.mean(level_spacings)

    emin = float(evals[0])
    emax = float(evals[-1])

    K_t = spectral_form_factor(evals_mid, t_vals, avg_spacing)

    del H

    return (
        r,
        ipr,
        emin,
        emax,
        avg_spacing,
        K_t,
        dim,
        diag_mode,
        arpack_converged,
        nconv
    )


# =========================
# ===== MAIN LOOP =========
# =========================

if __name__ == "__main__":

    start_time = time.time()
    results = []

    print("\n==== RESEARCH-GRADE RUN STARTED ====\n")

    for L in L_list:

        for W in W_values:

            print(f"L={L}, W={W:.2f}")

            seeds = [base_seed + i for i in range(N_real)]

            output = Parallel(n_jobs=n_jobs)(
                delayed(compute_one_realization)(L, W, seed)
                for seed in seeds
            )

            (
                r_vals,
                ipr_vals,
                emin_vals,
                emax_vals,
                spacing_vals,
                K_vals,
                dims,
                diag_modes,
                converged_flags,
                nconv_vals
            ) = zip(*output)

            K_array = np.array(K_vals)
            K_mean = np.mean(K_array, axis=0)

            # Save SFF separately
            np.save(
                f"results/sff/SFF_L{L}_W{W:.2f}_N{N_real}_k{k_eigs}.npy",
                K_mean
            )

            results.append({
                "L": L,
                "W": W,
                "J2": J2,
                "hilbert_dim": dims[0],
                "window_size": window,
                "diag_mode": diag_modes[0],
                "arpack_tol": ARPACK_TOL,
                "arpack_maxiter": MAX_ITER,
                "arpack_converged_all": all(converged_flags),
                "mean_nconv": np.mean(nconv_vals),
                "emin_mean": np.mean(emin_vals),
                "emax_mean": np.mean(emax_vals),
                "avg_spacing_mean": np.mean(spacing_vals),
                "r_mean": np.mean(r_vals),
                "r_sem": np.std(r_vals, ddof=1)/np.sqrt(N_real),
                "ipr_mean": np.mean(ipr_vals),
                "ipr_sem": np.std(ipr_vals, ddof=1)/np.sqrt(N_real),
                "n_realizations": N_real
            })

            df = pd.DataFrame(results)
            df.to_csv(
                f"results/data/research_run_L{L}_J2{J2}_N{N_real}_k{k_eigs}.csv",
                index=False
            )

    # ========================
    # ===== METADATA =========
    # ========================

    metadata = {
        "timestamp": time.ctime(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "joblib_version": joblib.__version__,
        "platform": platform.platform(),
        "git_commit": get_git_hash(),
        "parameters": {
            "L_list": L_list,
            "W_values": W_values.tolist(),
            "J2": J2,
            "Delta": Delta_val,
            "J": J_val,
            "N_real": N_real,
            "k_eigs": k_eigs,
            "window": window,
            "ARPACK_TOL": ARPACK_TOL,
            "MAX_ITER": MAX_ITER,
            "base_seed": base_seed,
            "t_vals": t_vals.tolist()
        }
    }

    with open("results/metadata/run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    total_time = time.time() - start_time
    print(f"\nRun completed in {total_time/60:.2f} minutes.")
    print("==== DONE ====\n")
