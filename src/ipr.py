import numpy as np

def mean_ipr(evecs_subset):
    """
    Computes IPR for a subset of states.
    IPR = sum(|psi|^4)
    """
    # Sum over the basis states (axis 0) for each eigenvector
    # then take the mean over the number of eigenvectors in the subset.
    return np.mean(np.sum(np.abs(evecs_subset)**4, axis=0))

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
