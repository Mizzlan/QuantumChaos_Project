import sys
import os

# Adds the parent directory (project root) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Disorder-induced chaos transition (Objective 1.2).
Investigates the transition from Chaos (GOE) to Localization (MBL) 
by averaging over multiple disorder realizations.
"""

import numpy as np
# Import our canonical modules
from src.model_spinchain import xxz_chain
from src.spectrum import get_middle_spectrum
from src.rstat import r_statistic
from src.plotting import plot_chaos_transition # Using your PRB-style plotter

# 1. PHYSICS: Setup parameters
#    L=14 is the 'Sweet Spot' for undergrad projects (large enough for chaos, small enough to run).
L = 10
Delta = 1.0
J2 = 0 # We keep J2 non-zero to start from a chaotic state
W_values = np.linspace(0.01, 6.0, 15) 
# W=0 is avoided to stay in the Sz=0 sector reliably
# W=0 is avoided because residual symmetries at zero disorder
# can artificially suppress level repulsion.
n_realizations = 50
assert n_realizations >= 30, "Too few disorder realizations for reliable averaging"

# 2. PYTHON: Storage for statistical results
r_means = []
r_errors = []

print(f"Starting Ensemble Scan for L={L}...")

for W in W_values:
    r_list = []
    
    # 3. PHYSICS: Ensemble Averaging (Objective 1.4)
    #    WHY: Disorder is random. One seed might be an outlier. 
    #    IMPACT: Averaging over 'n' seeds reveals the universal physics.
    for seed in range(n_realizations):
        # Build Hamiltonian with a unique seed for this realization
        H = xxz_chain(L, Delta=Delta, W=W, J2=J2, seed=seed)
        
        # Extract middle 50% of the spectrum (Santos standard)
        E_mid = get_middle_spectrum(H.eigvalsh())
        
        # Calculate r-statistic for this specific disorder configuration
        r_list.append(r_statistic(E_mid))

    # 4. PHYSICS: Statistical Reduction
    #    IMPACT: We need the Mean for the plot and the Error for credibility.
    mean_val = np.mean(r_list)
    # Standard Error of the Mean (SEM) = std / sqrt(N)
    error_val = np.std(r_list) / np.sqrt(n_realizations)
    
    r_means.append(mean_val)
    r_errors.append(error_val)
    
    print(f"Finished W={W:.2f} | <r> = {mean_val:.4f}")

# 5. PYTHON: Visualization
#    Uses the 'plot_chaos_transition' function we perfected earlier.
plot_chaos_transition(
    x_data=W_values,
    y_mean=r_means,
    y_err=r_errors,
    x_label="Disorder Strength $W/J$",
    title=f"Chaos-to-MBL Transition (L={L})",
    filename=f"r_vs_W_L{L}.pdf"
)