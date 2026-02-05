from src.model_spinchain import build_xxz_chain
from src.spectrum import get_sorted_eigenvalues
from src.rstat import r_statistic

L = 13

# ---------- Integrable ----------
H_int, basis_int = build_xxz_chain(
    L=L,
    h=None,
    J2=0.0
)
E_int = get_sorted_eigenvalues(H_int)
r_int = r_statistic(E_int)

# ---------- Chaotic (disorder) ----------
H_dis, basis_dis = build_xxz_chain(
    L=L,
    h=1.0,
    J2=0.0,
    seed=53
)
E_dis = get_sorted_eigenvalues(H_dis)
r_dis = r_statistic(E_dis)

# ---------- Chaotic (NNN) ----------
H_nnn, basis_nnn = build_xxz_chain(
    L=L,
    h=None,
    J2=0.6
)
E_nnn = get_sorted_eigenvalues(H_nnn)
r_nnn = r_statistic(E_nnn)

print(f"L = {L}, Sz = 0 sector")
# print(f"Integrable NN XXZ      r = {r_int:.4f}  (Poisson ≈ 0.386)")
# print(f"Disorder-induced chaos r = {r_dis:.4f}  (GOE ≈ 0.536)")
print(f"NNN-induced chaos      r = {r_nnn:.4f}  (GOE ≈ 0.536)")
