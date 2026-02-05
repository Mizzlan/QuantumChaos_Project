import numpy as np

def r_statistic(energies, eps=1e-12):
    """
    Computes the average adjacent gap ratio <r>.

    Physics:
    - r ≈ 0.386 → Poisson (integrable / localized)
    - r ≈ 0.536 → GOE (chaotic)

    This implementation is robust for:
    - small system sizes
    - integrable limits
    - weak disorder
    """

    energies = np.asarray(energies)

    # Require minimum number of levels
    if len(energies) < 5:
        return np.nan

    # Sort energies
    energies = np.sort(energies)

    # Level spacings
    spacings = np.diff(energies)

    # Remove exact degeneracies ONLY
    spacings = spacings[spacings > eps]

    # Need at least two spacings
    if len(spacings) < 2:
        return np.nan

    # Adjacent gap ratios
    r_vals = np.minimum(spacings[1:], spacings[:-1]) / \
             np.maximum(spacings[1:], spacings[:-1])

    # Final safety check
    if len(r_vals) == 0:
        return np.nan

    return np.mean(r_vals)
