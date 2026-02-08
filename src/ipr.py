import numpy as np

def compute_ipr(evecs):
    """
    Compute IPR for each eigenstate.
    evecs: columns are eigenvectors
    """
    return np.sum(np.abs(evecs)**4, axis=0)


def mid_spectrum_indices(evals, fraction=0.2):
    n = len(evals)
    k = int(fraction * n)
    start = n//2 - k//2
    end = n//2 + k//2
    return np.arange(start, end)


def disorder_averaged_ipr(evals, evecs, fraction=0.2):
    idx = mid_spectrum_indices(evals, fraction)
    ipr_vals = compute_ipr(evecs)
    return np.mean(ipr_vals[idx])
