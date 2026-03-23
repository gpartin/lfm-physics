"""06 — Dark Matter

In example 02 we saw that energy creates a χ-well (gravity).
Here's the question: what happens when the energy LEAVES?

Answer: the well persists.  χ has its own dynamics (GOV-02 is
a wave equation) and it doesn't instantly snap back to 19.
The leftover well attracts other matter even though nothing is
there anymore.

This IS dark matter in LFM — not a new particle, but the
substrate's memory of where matter used to be.
"""

import numpy as np

import lfm

config = lfm.SimulationConfig(grid_size=48)
sim = lfm.Simulation(config)

center = (24, 24, 24)

print("06 — Dark Matter")
print("=" * 55)
print()

# --- Phase 1: Create a particle with a deep χ-well ---
sim.place_soliton(center, amplitude=8.0, sigma=4.0)
sim.equilibrate()
sim.run(steps=2000)

m1 = sim.metrics()
chi_center_before = sim.chi[24, 24, 24]
print(f"Phase 1 — Particle present:")
print(f"  χ at center = {chi_center_before:.2f}  (well depth: {lfm.CHI0 - chi_center_before:.2f})")
print(f"  χ_min       = {m1['chi_min']:.2f}")
print()

# --- Phase 2: Remove ALL matter ---
sim.psi_real = np.zeros_like(sim.psi_real)
if sim.psi_imag is not None:
    sim.psi_imag = np.zeros_like(sim.psi_imag)

psi_sq = sim.psi_real**2
print(f"Phase 2 — Matter removed:")
print(f"  |Ψ|² total  = {psi_sq.sum():.2e}  (zero)")
print(f"  χ at center = {sim.chi[24, 24, 24]:.2f}  (well still there!)")
print()

# --- Phase 3: Evolve with no matter — does the well persist? ---
print(f"Phase 3 — Evolution with no matter:")
for step in range(1, 6):
    sim.run(steps=1000)
    chi_c = sim.chi[24, 24, 24]
    chi_min = sim.chi.min()
    depth = lfm.CHI0 - chi_min
    print(
        f"  Step {step * 1000:5d}: χ_center = {chi_c:.2f}, χ_min = {chi_min:.2f}, depth = {depth:.2f}"
    )

chi_min_final = sim.chi.min()
final_depth = lfm.CHI0 - chi_min_final

print()
if final_depth > 0.5:
    print(f"RESULT: χ well persists!  Depth = {final_depth:.2f}")
    print("  → Gravitational well with NO matter present")
    print("  → This IS dark matter: substrate memory")
else:
    print(f"RESULT: Well has mostly dissipated (depth = {final_depth:.2f})")
    print("  → Try higher amplitude for longer persistence")

print()
print("Dark matter is not a substance — it's the lattice remembering")
print("where matter used to be.  The χ-well persists and attracts.")
print()
print("Next: can matter appear from nothing? (→ 07)")
