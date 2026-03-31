"""Debug: check if phases are correctly applied to two-particle placements."""
from lfm import create_two_particles
from lfm.particles import get_particle
import numpy as np

e = get_particle("electron")
p = get_particle("positron")
print(f"Electron phase: {e.phase}")
print(f"Positron phase: {p.phase}")

pa_s, pb_s = create_two_particles("electron", "electron", separation=6, N=48)
pr_same = np.asarray(pa_s.sim.psi_real)
pi_same = pa_s.sim.psi_imag
if pi_same is not None:
    pi_same = np.asarray(pi_same)
c = 24
print(f"\nSame-charge: psi_real shape={pr_same.shape}")
print(f"  psi_real at y=z=24, x=18..30:")
for i in range(18, 31):
    val_r = pr_same[i, c, c] if pr_same.ndim == 3 else pr_same[0, i, c, c]
    print(f"    x={i}: psi_r={val_r:.4f}")
if pi_same is not None:
    print(f"  psi_imag max={pi_same.max():.6f}")

pa_o, pb_o = create_two_particles("electron", "positron", separation=6, N=48)
pr_opp = np.asarray(pa_o.sim.psi_real)
pi_opp = pa_o.sim.psi_imag
if pi_opp is not None:
    pi_opp = np.asarray(pi_opp)
print(f"\nOpp-charge: psi_real shape={pr_opp.shape}")
print(f"  psi_real at y=z=24, x=18..30:")
for i in range(18, 31):
    val_r = pr_opp[i, c, c] if pr_opp.ndim == 3 else pr_opp[0, i, c, c]
    val_i = 0.0
    if pi_opp is not None:
        val_i = pi_opp[i, c, c] if pi_opp.ndim == 3 else pi_opp[0, i, c, c]
    print(f"    x={i}: psi_r={val_r:.4f}, psi_i={val_i:.4f}")

psi_sq_same = pr_same**2
if pi_same is not None:
    psi_sq_same = psi_sq_same + pi_same**2
psi_sq_opp = pr_opp**2
if pi_opp is not None:
    psi_sq_opp = psi_sq_opp + pi_opp**2
print(f"\nTotal |Psi|^2 same={float(psi_sq_same.sum()):.4f}")
print(f"Total |Psi|^2 opp ={float(psi_sq_opp.sum()):.4f}")
