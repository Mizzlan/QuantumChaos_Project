"""
model_spinchain.py
------------------

Spin-1/2 XXZ chain with disorder and next-nearest-neighbor coupling.

Physics Model:
    H = J Σ_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y)
      + JΔ Σ_i S_i^z S_{i+1}^z
      + J2 Σ_i (S_i^x S_{i+2}^x + S_i^y S_{i+2}^y)
      + J2Δ Σ_i S_i^z S_{i+2}^z
      + Σ_i h_i S_i^z

Disorder:
    h_i ∈ Uniform[-W, W]

Symmetry:
    Total magnetization sector Sz = 0 (Nup = L/2)

Boundary Conditions:
    Open (default)
    Periodic (optional)

Designed for:
    Quantum chaos / MBL finite-size scaling studies.
"""

import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d


def xxz_chain(
    L: int,
    J: float = 1.0,
    Delta: float = 1.0,
    W: float = 0.0,
    J2: float = 0.0,
    seed: int | None = None,
    boundary: str = "open",
):
    """
    Constructs the disordered XXZ spin chain Hamiltonian.

    Parameters
    ----------
    L : int
        System size (must be even for Sz=0 sector).

    J : float
        Nearest-neighbor XY coupling strength.

    Delta : float
        Anisotropy parameter.

    W : float
        Disorder strength (uniform in [-W, W]).

    J2 : float
        Next-nearest-neighbor coupling (breaks integrability).

    seed : int or None
        Random seed for reproducibility.

    boundary : {"open", "periodic"}
        Boundary condition type.

    Returns
    -------
    H : quspin.operators.hamiltonian
        Sparse Hamiltonian object in Sz=0 sector.
    """

    # -----------------------------
    # 1. VALIDATION
    # -----------------------------
    if L % 2 != 0:
        raise ValueError("System size L must be even for Sz=0 sector.")

    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'.")

    # -----------------------------
    # 2. LOCAL RANDOM GENERATOR
    # -----------------------------
    rng = np.random.default_rng(seed)

    # -----------------------------
    # 3. SYMMETRY SECTOR
    # -----------------------------
    basis = spin_basis_1d(L, Nup=L // 2, pauli=False)

    # -----------------------------
    # 4. INTERACTION RANGES
    # -----------------------------
    if boundary == "open":
        nn_range = range(L - 1)
        nnn_range = range(L - 2)
    else:  # periodic
        nn_range = range(L)
        nnn_range = range(L)

    # -----------------------------
    # 5. NEAREST-NEIGHBOR TERMS
    # -----------------------------
    J_xy = [[0.5 * J, i, (i + 1) % L] for i in nn_range]
    J_zz = [[Delta * J, i, (i + 1) % L] for i in nn_range]

    # -----------------------------
    # 6. NEXT-NEAREST-NEIGHBOR TERMS
    # -----------------------------
    J2_xy = [[0.5 * J2, i, (i + 2) % L] for i in nnn_range]
    J2_zz = [[Delta * J2, i, (i + 2) % L] for i in nnn_range]

    # -----------------------------
    # 7. DISORDER FIELD
    # -----------------------------
    hz = [[rng.uniform(-W, W), i] for i in range(L)]

    # -----------------------------
    # 8. OPERATOR ASSEMBLY
    # -----------------------------
    static = [
        ["+-", J_xy],
        ["-+", J_xy],
        ["zz", J_zz],
        ["+-", J2_xy],
        ["-+", J2_xy],
        ["zz", J2_zz],
        ["z", hz],
    ]

    # -----------------------------
    # 9. BUILD HAMILTONIAN
    # -----------------------------
    H = hamiltonian(
        static,
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_symm=False  # Safe for controlled construction
    )

    return H
