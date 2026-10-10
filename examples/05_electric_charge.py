"""05 — Historical phase-dependent separation demo.

Evolves two Gaussian preparations and compares their endpoint separation.
Absolute phase 0/pi does not establish positive/negative electric charge.
Endpoint displacement is not acceleration, force, or an inverse-square law.
This legacy implementation does not implement authoritative v39.0.2 dynamics.
See docs/coulomb_audit.md for quantitative gates and research limitations.
"""
import numpy as np

import lfm
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie

_args = parse_no_anim()
_OUT  = make_out_dir("05_electric_charge")

config = lfm.SimulationConfig(
    grid_size=48,
    field_level=lfm.FieldLevel.COMPLEX,
)

print("05 — Phase-dependent separation")
print("=" * 55)
print()

# --- Experiment A: Same phase (both θ=0) ---
sim_same = lfm.Simulation(config)
sim_same.place_soliton((17, 24, 24), amplitude=5.0, sigma=3.5, phase=0.0)
sim_same.place_soliton((31, 24, 24), amplitude=5.0, sigma=3.5, phase=0.0)
sim_same.equilibrate()

psi_sq = sim_same.psi_real**2 + sim_same.psi_imag**2
sep_same_0 = lfm.measure_separation(psi_sq)

sim_same.run(steps=3000)
psi_sq = sim_same.psi_real**2 + sim_same.psi_imag**2
sep_same_f = lfm.measure_separation(psi_sq)

# --- Experiment B: Opposite phase (θ=0 and θ=π) ---
sim_opp = lfm.Simulation(config)
sim_opp.place_soliton((17, 24, 24), amplitude=5.0, sigma=3.5, phase=0.0)
sim_opp.place_soliton((31, 24, 24), amplitude=5.0, sigma=3.5, phase=np.pi)
sim_opp.equilibrate()

psi_sq = sim_opp.psi_real**2 + sim_opp.psi_imag**2
sep_opp_0 = lfm.measure_separation(psi_sq)

sim_opp.run(steps=3000)
psi_sq = sim_opp.psi_real**2 + sim_opp.psi_imag**2
sep_opp_f = lfm.measure_separation(psi_sq)

# --- Report ---
print("Same-phase preparation:")
print(f"  Initial separation: {sep_same_0:.1f}")
print(f"  Final separation:   {sep_same_f:.1f}")
print(f"  Change: {sep_same_f - sep_same_0:+.1f} cells")
print()

print("Opposite-phase preparation:")
print(f"  Initial separation: {sep_opp_0:.1f}")
print(f"  Final separation:   {sep_opp_f:.1f}")
print(f"  Change: {sep_opp_f - sep_opp_0:+.1f} cells")
print()

delta_same = sep_same_f - sep_same_0
delta_opp = sep_opp_f - sep_opp_0

if delta_opp < delta_same:
    print("Opposite-phase separation change is smaller than same-phase change.")
else:
    print("Opposite-phase separation change is not smaller in this run.")
print("Coulomb law NOT ESTABLISHED: no signed-source or force-distance test.")
print()
print("Next: what is dark matter? (→ 06)")
# 3-D movie showing opposite-phase pair evolution
snaps, _movie = run_and_save_3d_movie(
    sim_opp, steps=1000, out_dir=_OUT, stem="electric_charge",
    field="psi_real", snapshot_every=20, no_anim=_args.no_anim,
)