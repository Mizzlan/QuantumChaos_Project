"""
Defines spin-chain Hamiltonians for quantum chaos studies.
Standard: Open Boundary Conditions (OBC) as per Gubin & Santos (2012).
"""

import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

def xxz_chain(L, J=1.0, Delta=1.0, W=0.0, J2=0.0, seed=None):
    """
    Constructs the XXZ spin chain Hamiltonian.
    """

    # 1. PYTHON: Reproducibility
    #    PHYSICS: Guarantees that the 'randomness' is consistent across runs.
    if seed is not None:
        np.random.seed(seed)

    # 2. PHYSICS: Symmetry Sector Selection.
    #    WHY: We must restrict to the Sz=0 sector (Nup = L // 2).
    #    IMPACT: Prevents level-crossing from different sectors that mimics integrability.
    basis = spin_basis_1d(L, Nup=L // 2, pauli=False) 

    # 3. PHYSICS: Nearest-neighbor (NN) XXZ terms.
    #    IMPACT: Uses OBC (range L-1) to avoid momentum conservation complexities.
    J_xy = [[J, i, i+1] for i in range(L-1)]
    J_zz = [[Delta * J, i, i+1] for i in range(L-1)]

    # 4. PHYSICS: Next-nearest-neighbor (NNN) coupling.
    #    IMPACT: The primary chaos trigger that breaks Bethe-Ansatz integrability.
    J2_xy = [[J2, i, i+2] for i in range(L-2)]
    J2_zz = [[Delta * J2, i, i+2] for i in range(L-2)]

    # 5. PHYSICS: Disorder fields (W).
    #    IMPACT: Random field at each site; used for studying Localization (MBL).
    hz = [[np.random.uniform(-W, W), i] for i in range(L)]

    # 6. PYTHON: Assembly of operator lists.
    #    '+-' and '-+' are the raising/lowering products (exchange terms).
    static = [
        ["+-", J_xy], ["-+", J_xy], ["zz", J_zz],
        ["+-", J2_xy], ["-+", J2_xy], ["zz", J2_zz],
        ["z", hz]
    ]

    # 7. PYTHON: Build the Hamiltonian object.
    #    RESEARCH: check_herm=False accelerates the calculation for large ensembles.
    return hamiltonian(static, [], basis=basis, dtype=np.float64, check_herm=False)