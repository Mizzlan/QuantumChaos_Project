import numpy as np

def extract_wc(W_vals, r_vals, r_mid=0.46):
    """
    Extract pseudo-critical disorder W_c(L)
    """
    return np.interp(r_mid, r_vals[::-1], W_vals[::-1])


def extract_vL(W_vals, r_vals, Wc):
    """
    Extract slope v(L) at W_c
    """
    idx = np.argmin(np.abs(W_vals - Wc))
    if idx == 0 or idx == len(W_vals) - 1:
        return np.nan
    return (r_vals[idx+1] - r_vals[idx-1]) / (W_vals[idx+1] - W_vals[idx-1])
