import numpy as np

frac = 0.5
def get_middle_spectrum(energies, fraction=frac):
    """
    Extracts the central fraction of the spectrum.

    Physics:
    - The middle of the spectrum best represents infinite-temperature behavior
    - Avoids edge effects and conserved-quantity artifacts
    """

    energies = np.sort(np.asarray(energies))  # 🔥 CRITICAL LINE

    n = len(energies)
    if n < 10:
        return np.array([])

    start = int(n * (1 - fraction) / 2)
    end = int(n * (1 + fraction) / 2)

    return energies[start:end]



def unfold_spectrum(energies, order=6):
    """
    Standardizes the energy spectrum for Universal comparison.
    Reference: Haake, 'Quantum Signatures of Chaos' (The 'Bible' of this field).
    """

    # 1. PYTHON: Sorts the energies in ascending order.
    #    PHYSICS: You cannot have a 'staircase' if the steps are out of order.
    #    IMPACT: Ensures we are counting levels from ground state upwards.
    energies = np.sort(energies)

    # 2. PYTHON: Creates an array [1, 2, 3, ..., n].
    #    PHYSICS: This represents the 'Staircase Function' N(E), which counts 
    #             how many states exist below energy E cumulative adding.
    #    IMPACT: This is our 'Raw' counting data before we smooth it out.
    n = len(energies)
    N_E = np.arange(1, n + 1)
    #N_E is the cumulative number of states 

    # 3. PYTHON: Chooses the smaller value between 6 and (total levels / 10).
    #    PHYSICS: Prevents 'Overfitting'. We want the 'Global' trend, not local noise.
    #    IMPACT: If you fit too closely, you 'delete' the chaos you are trying to find.
    poly_order = min(order, n // 10)

    # 4. PYTHON: Fits a polynomial to the curve of (energies vs N_E).
    coeffs = np.polyfit(energies, N_E, poly_order)
    #it becomes a staircase then this function uses a line of best fit and fit it to a curve

    # 5. PYTHON: Evaluates the polynomial at each original energy point.
    #    PHYSICS: This is the 'Unfolded Spectrum' (mapping E to a linear scale).
    #    IMPACT: After this line, the average gap between your levels is exactly 1.
    #            You can now compute the Spectral Form Factor (SFF).
    unfolded = np.polyval(coeffs, energies)

    return unfolded


def mid_spectrum_indices(evals, fraction=0.2):
    """
    Returns indices corresponding to the middle fraction of the spectrum.
    """
    N = len(evals)
    k = int(fraction * N)
    center = N // 2
    return np.arange(center - k // 2, center + k // 2)