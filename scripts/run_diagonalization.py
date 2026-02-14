import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from model_spinchain00 import xxz_chain

# PARAMETERS
L_list = [8, 10, 12, 14]
W_values = np.linspace(0.1, 6.0, 15)
J2_values = [0.0, 0.5]
n_realizations = 50

save_dir = "data/spectra"
os.makedirs(save_dir, exist_ok=True)

for L in L_list:
    for J2 in J2_values:
        for W in W_values:
            for seed in range(n_realizations):
                H = xxz_chain(L=L, Delta=1.0, W=W, J2=J2, seed=seed)
                evals, evecs = np.linalg.eigh(H.toarray())

                fname = f"L{L}_W{W:.3f}_J2{J2}_seed{seed}.npz"
                np.savez(
                    os.path.join(save_dir, fname),
                    evals=evals,
                    evecs=evecs
                )

            print(f"Saved spectra for L={L}, J2={J2}, W={W:.2f}")
