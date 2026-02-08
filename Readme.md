Files: model_spinchain.py, spectrum.py, rstat.py, plotting.py.

Hamiltonian: build_xxz_chain(L, Jxy, Delta, J2, disorder).

Metrics: get_middle_spectrum(energies, fraction), r_statistic(energies).

Convention: L is size, Delta is Ising strength, J2 is NNN coupling
J2 is NNN
disorder is disorder

Hamiltonian
   ↓
Spectrum (middle only)
   ↓
Level spacings
   ↓
r-statistic
  ↓
Ensemble averaging(he "Missing Link") * (((Why: If you have disorder, one random "seed" might give you $r=0.51$ and another $r=0.55$. To be precise, you must run it $N$ times and take the mean.)))
   ↓
Parameter scan (Vary J2 or W)
   ↓
Finite-size scaling
   ↓
Publication plot

hello world my name is milan chapagain
i am doing this project a
anydesk is working fine 
and its good speed4