"""25 — Electron at Rest

Create a stable electron eigenmode using the SCF solver and verify it stays
put for 1000 steps.

A raw Gaussian blob placed in a uniform chi=19 lattice will disperse within
a few hundred steps — it is not a stable particle.  The eigenmode solver runs
GOV-01 + GOV-02 together until the wave packet and chi-well converge on a
self-consistent solution, producing a true soliton that does not disperse.
"""

import numpy as np

import lfm

print("25 — Electron at Rest (eigenmode verification)")
print("=" * 55)
print()

# Create a stable electron via eigenmode solver (use_eigenmode=True, default).
print("Solving electron eigenmode (N=48, 5000 GOV-01+GOV-02 steps)...")
placed = lfm.create_particle("electron", N=48, use_eigenmode=True)
sim = placed.sim

m0 = sim.metrics()
print("\nAfter eigenmode solve:")
print(f"  chi_min  = {m0['chi_min']:.3f}  (below 19 = gravity well present)")
print(f"  energy   = {m0['energy_total']:.4e}")
center0 = lfm.measure_center_of_energy(sim)
print(f"  position = ({center0[0]:.2f}, {center0[1]:.2f}, {center0[2]:.2f})")
print()

# Run 1000 more steps and check that energy is conserved and position is stable.
sim.run(steps=1000)
m1 = sim.metrics()
center1 = lfm.measure_center_of_energy(sim)

drift = float(np.linalg.norm(center1 - center0))
energy_rel = abs(m1["energy_total"] - m0["energy_total"]) / max(abs(m0["energy_total"]), 1e-30)

print("After 1000 more steps:")
print(f"  chi_min  = {m1['chi_min']:.3f}")
print(f"  energy   = {m1['energy_total']:.4e}")
print(f"  position = ({center1[0]:.2f}, {center1[1]:.2f}, {center1[2]:.2f})")
print()
print("Stability metrics:")
print(f"  position drift = {drift:.3f} cells  (target < 3.0)")
print(f"  energy drift   = {energy_rel * 100:.2f}%  (target < 5%)")
print()

stable = drift < 3.0 and energy_rel < 0.05
print(f"Eigenmode stable: {stable}")
print()
print("The electron soliton sits in its own chi-well and does not disperse.")
print("Without use_eigenmode=True, a Gaussian blob would spread and fade.")
print()
print("Next: move the electron (-> 26).")
