import numpy as np
from itertools import combinations
import time

# --- 1. BUILD THE BASIS ---
def sz0_basis(N):
    # This finds all states with total Sz=0 (half up, half down)
    n_up = N // 2
    basis_states = []
    print(f"  > Creating basis for N={N}...")
    for ups in combinations(range(N), n_up):
        state = 0
        for i in ups:
            state |= (1 << i)
        basis_states.append(state)
    
    state_index = {state: i for i, state in enumerate(basis_states)}
    print(f"  > Basis size: {len(basis_states)}")
    return basis_states, state_index

# --- 2. SPIN HELPERS ---
def spin_z(state, site):
    # Returns +1 for UP, -1 for DOWN
    return 1 if (state >> site) & 1 else -1

def flip_spins(state, i, j):
    # Flips the bits at i and j
    mask = (1 << i) | (1 << j)
    return state ^ mask

# --- 3. BUILD THE MATRIX ---
def build_xxz_hamiltonian(N, Delta=1.0, J2=0.0):
    basis, index = sz0_basis(N)
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=np.float64) 

    # We add a small 'Magnetic Field' to the first spin
    # This 'breaks' reflection symmetry so the levels don't overlap
    h_field = 0.1 

    for idx, state in enumerate(basis):
        # ADD THIS: Breaking symmetry with a local field
        H[idx, idx] += h_field * spin_z(state, 0)

        for i in range(N - 1): # Changed to N-1 for 'Open Boundary Conditions'
            j = i + 1 

            # Sz Sz term
            H[idx, idx] += Delta * spin_z(state, i) * spin_z(state, j)

            # Flip-flop (SxSy)
            if spin_z(state, i) != spin_z(state, j):
                flipped = flip_spins(state, i, j)
                H[idx, index[flipped]] += 2.0

            # Next-Nearest Neighbor (Chaos)
            if J2 != 0.0 and i < N - 2:
                k = i + 2
                if spin_z(state, i) != spin_z(state, k):
                    flipped = flip_spins(state, i, k)
                    H[idx, index[flipped]] += 2.0 * J2
                H[idx, idx] += J2 * spin_z(state, i) * spin_z(state, k)
    return H

# --- 4. THE STATISTICS ---
def r_statistics(evals):
    spacings = np.diff(evals)
    # Ratio of adjacent gaps
    # We add a tiny 1e-12 to avoid 'division by zero' errors
    r_vals = np.minimum(spacings[:-1], spacings[1:]) / (np.maximum(spacings[:-1], spacings[1:]) + 1e-12)
    return np.mean(r_vals)

# --- 5. MAIN EXECUTION ---
# Start with N=10 to ensure it works, then move to N=12
N = 10
print(f"=== PROJECT START: N={N} ===")

# --- INTEGRABLE RUN ---
start = time.time()
print("\n[Step 1] Building Integrable Model (J2=0)...")
H_int = build_xxz_hamiltonian(N, Delta=1.0, J2=0.0)
print(f"  > Diagonalizing...")
evals_int = np.linalg.eigvalsh(H_int)
rmean_int = r_statistics(evals_int)
print(f"  > Done. Time: {time.time()-start:.2f}s")

# --- CHAOTIC RUN ---
start = time.time()
print("\n[Step 2] Building Chaotic Model (J2=0.5)...")
H_chaos = build_xxz_hamiltonian(N, Delta=1.0, J2=0.5)
print(f"  > Diagonalizing...")
evals_chaos = np.linalg.eigvalsh(H_chaos)
rmean_chaos = r_statistics(evals_chaos)
print(f"  > Done. Time: {time.time()-start:.2f}s")

print("\n" + "="*40)
print(f"RESULT FOR N={N}:")
print(f"Integrable r: {rmean_int:.4f} (Target 0.386)")
print(f"Chaotic r:    {rmean_chaos:.4f} (Target 0.536)")
print("="*40)

input("\nSuccess! Press Enter to exit...")
# --- THE FINAL RESEARCH PLOT ---

