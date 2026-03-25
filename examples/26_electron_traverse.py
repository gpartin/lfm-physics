"""26 — Electron Traverse

Boost an electron to v = 0.04c and watch it travel across the grid.

Momentum is encoded as a phase gradient in the complex psi field.  The
CALC-31 constraint sets the maximum speed: v_max = sqrt((chi0 - chi_min)/chi0).
For an electron with chi_min ~ 16, v_max ~ 0.39c.  At v = 0.04c we are well
inside the stable regime.
"""

import lfm

N = 64
V = 0.04  # in units of c
STEPS = 10_000

print("26 — Electron Traverse (v = 0.04c)")
print("=" * 55)
print()

print(f"Grid: N = {N}  Steps: {STEPS}")
print(f"Velocity: vx = {V}c")
print()

placed = lfm.create_particle("electron", N=N, velocity=(V, 0.0, 0.0))
sim = placed.sim

pos0 = lfm.measure_center_of_energy(sim)
print(f"Initial position:  x = {pos0[0]:.2f} cells")
print()

print(f"Running {STEPS} steps...")
sim.run(STEPS, evolve_chi=False)

pos1 = lfm.measure_center_of_energy(sim)
print(f"Final position:    x = {pos1[0]:.2f} cells")
print()

actual_disp = float(pos1[0] - pos0[0])
expected_disp = V * STEPS * lfm.constants.DT_DEFAULT  # v * N_steps * dt

print("Results:")
print(f"  Actual displacement:   {actual_disp:.1f} cells")
print(f"  Expected displacement: {expected_disp:.1f} cells  (v * steps * dt)")
if expected_disp > 0:
    ratio = actual_disp / expected_disp
    print(f"  Ratio actual/expected: {ratio:.2f}")
print()
print("The electron crossed the grid driven by the phase-gradient boost.")
print("No Newtonian force was injected — momentum came from the wave phase.")
print()
print("Next: compare electron vs muon at the same velocity (-> 27).")
