"""27 — Electron vs Muon: Mass-Dependent Speed

Launch an electron and a muon at the same phase-gradient velocity and compare
how far each travels.

In LFM, the muon is 207× heavier than the electron.  From CALC-31:
  v_max = sqrt((chi0 - chi_min) / chi0)
A heavier particle sits in a deeper chi-well, so its v_max is higher.
But for the SAME encoded phase gradient, the heavier particle carries far
more momentum (E ~ chi_min * amplitude) and will travel at a similar apparent
velocity — the ratio of displacements converges toward 1 as mass grows.
"""

import numpy as np
import lfm

N = 64
V = 0.04    # phase gradient velocity (same for both)
STEPS = 6_000

print("27 — Electron vs Muon: Mass-Dependent Kinematics")
print("=" * 55)
print()
print(f"Grid: N={N}   Velocity: {V}c   Steps: {STEPS}")
print()

placed_e = lfm.create_particle("electron", N=N, velocity=(V, 0.0, 0.0))
placed_m = lfm.create_particle("muon",     N=N, velocity=(V, 0.0, 0.0))

pos0_e = lfm.measure_center_of_energy(placed_e.sim)
pos0_m = lfm.measure_center_of_energy(placed_m.sim)

placed_e.sim.run(STEPS, evolve_chi=False)
placed_m.sim.run(STEPS, evolve_chi=False)

pos1_e = lfm.measure_center_of_energy(placed_e.sim)
pos1_m = lfm.measure_center_of_energy(placed_m.sim)

disp_e = float(pos1_e[0] - pos0_e[0])
disp_m = float(pos1_m[0] - pos0_m[0])
expected  = V * STEPS * lfm.constants.DT_DEFAULT

print(f"{'Particle':<12} {'mass_ratio':>12} {'disp (cells)':>14} {'ratio to expected':>18}")
print("-" * 60)
print(
    f"{'electron':<12} {lfm.ELECTRON.mass_ratio:>12.1f}"
    f" {disp_e:>14.1f} {disp_e/expected:>18.2f}"
)
print(
    f"{'muon':<12} {lfm.MUON.mass_ratio:>12.1f}"
    f" {disp_m:>14.1f} {disp_m/expected:>18.2f}"
)
print()
print("Both particles encoded the same phase velocity; the LFM dispersion slightly")
print("modifies the group velocity based on each particle's chi-well depth.")
