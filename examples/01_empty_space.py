"""01 — Empty Space

What does the vacuum look like?

In LFM, empty space isn't nothing — it's a lattice where every
point has a stiffness value chi (χ) equal to 19.  This number
comes from the geometry of a 3D cubic lattice:
    1 center mode + 6 face modes + 12 edge modes = 19.

Run this first.  It takes a few seconds and prints what "nothing"
looks like in the simulation.
"""

import lfm

# Create a 32³ lattice.  No particles, no energy — just vacuum.
config = lfm.SimulationConfig(grid_size=32)
sim = lfm.Simulation(config)

print("01 — Empty Space")
print("=" * 55)
print()
print("A 32×32×32 lattice, freshly created.")
print()

chi = sim.chi
print(f"  chi everywhere  = {chi.mean():.1f}")
print(f"  chi min         = {chi.min():.1f}")
print(f"  chi max         = {chi.max():.1f}")
print(f"  energy density  = {sim.energy_density.sum():.6f}")
print()
print("Every point is χ₀ = 19.  No wells, no voids, no structure.")
print("This is the emptiest possible universe.")
print()

# Evolve for a bit — nothing should change.
sim.run(steps=500)
m = sim.metrics()
print(f"After 500 steps of evolution:")
print(f"  chi mean  = {m['chi_mean']:.4f}")
print(f"  chi std   = {m['chi_std']:.6f}")
print(f"  wells     = {m['well_fraction']*100:.1f}%")
print()
print("Nothing happened — because there's no energy to drive change.")
print("Empty space is stable.  Now let's add something to it (→ 02).")
