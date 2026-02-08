import numpy as np
from src.finite_size import extract_wc, extract_vL

data = np.loadtxt("results/data/resultsipr_L{L}_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}.csv", delimiter=",", skiprows=1)

L_vals = np.unique(data[:,0])

for L in L_vals:
    mask = data[:,0] == L
    W = data[mask,1]
    r = data[mask,2]

    Wc = extract_wc(W, r)
    vL = extract_vL(W, r, Wc)

    print(f"L={L}, Wc={Wc:.3f}, v(L)={vL:.3f}")
