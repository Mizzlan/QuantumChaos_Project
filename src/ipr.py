import numpy as np

def compute_ipr(evecs):
    """
    Computes IPR for each eigenstate.
    IPR_n = sum_i |psi_n(i)|^4
    """
    return np.sum(np.abs(evecs)**4, axis=0)


def microcanonical_ipr(evals, evecs, fraction=0.5):
    """
    Computes mean IPR in the middle of the spectrum.
    """
    N = len(evals)
    start = int((1 - fraction) / 2 * N)
    end   = int((1 + fraction) / 2 * N)

    ipr_vals = compute_ipr(evecs[:, start:end])
    return np.mean(ipr_vals)
