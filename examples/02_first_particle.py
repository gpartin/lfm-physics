"""02 — Your First Particle

Drop a lump of energy onto the lattice and watch gravity appear.

In example 01 we saw that empty space has χ = 19 everywhere.
Now we add energy: a Gaussian soliton (a smooth blob of wave
amplitude).  GOV-02 says that energy density |Ψ|² pulls χ
downward.  Low χ = gravitational well.

This is gravity emerging from nothing but the wave equations.
"""

import lfm

config = lfm.SimulationConfig(grid_size=48)
sim = lfm.Simulation(config)

print("02 — Your First Particle")
print("=" * 55)
print()

# Place one soliton at the grid center.
center = (24, 24, 24)
sim.place_soliton(center, amplitude=5.0, sigma=4.0)

print("Before equilibration:")
m = sim.metrics()
print(f"  χ_min  = {m['chi_min']:.2f}  (should be 19 — no well yet)")
print()

# Equilibrate: solve Poisson equation so chi adjusts to the energy.
sim.equilibrate()

print("After equilibration (GOV-02 created the well):")
m = sim.metrics()
print(f"  χ_min  = {m['chi_min']:.2f}  (below 19 → gravity well!)")
print(f"  wells  = {m['well_fraction'] * 100:.1f}%  of the grid is a well")
print()

# Evolve — the soliton sits in its own gravity well.
sim.run(steps=2000)
m = sim.metrics()
print(f"After 2000 steps of evolution:")
print(f"  χ_min  = {m['chi_min']:.2f}")
print(f"  wells  = {m['well_fraction'] * 100:.1f}%")
print(f"  energy = {m['energy_total']:.2e}")
print()
print("The energy blob created a χ-well and sits inside it.")
print("No Newton's law was injected — gravity emerged from GOV-02.")
print()
print("Next: measure the shape of this well (→ 03).")
