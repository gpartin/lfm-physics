"""04 — Two Bodies

In examples 01–03 we worked with one particle.  Now we place two
solitons and watch them interact.  Each one creates a χ-well via
GOV-02, and the other soliton's wave equation (GOV-01) bends
toward the low-χ region.

No force law is injected — gravitational attraction emerges from
the coupled dynamics of Ψ and χ.
"""

import lfm

config = lfm.SimulationConfig(grid_size=48)
sim = lfm.Simulation(config)

# Two solitons, separated by 14 cells along the x-axis.
pos_a = (17, 24, 24)
pos_b = (31, 24, 24)
sim.place_soliton(pos_a, amplitude=5.0, sigma=3.5)
sim.place_soliton(pos_b, amplitude=5.0, sigma=3.5)
sim.equilibrate()

print("04 — Two Bodies")
print("=" * 55)
print()

# Track how the separation changes over time.
psi_sq = sim.psi_real ** 2
if sim.psi_imag is not None:
    psi_sq = psi_sq + sim.psi_imag ** 2
initial_sep = lfm.measure_separation(psi_sq)
print(f"Initial separation: {initial_sep:.1f} cells")
print()

print(f"  {'step':>6s}  {'separation':>10s}  {'χ_min':>8s}")
print(f"  {'------':>6s}  {'----------':>10s}  {'--------':>8s}")

separations = []
for i in range(10):
    sim.run(steps=500)
    psi_sq = sim.psi_real ** 2
    if sim.psi_imag is not None:
        psi_sq = psi_sq + sim.psi_imag ** 2
    sep = lfm.measure_separation(psi_sq)
    m = sim.metrics()
    separations.append(sep)
    step = (i + 1) * 500
    print(f"  {step:6d}  {sep:10.2f}  {m['chi_min']:8.2f}")

final_sep = separations[-1]
delta = final_sep - initial_sep

print()
if delta < -0.5:
    print(f"Separation decreased by {-delta:.1f} cells → ATTRACTION")
elif delta > 0.5:
    print(f"Separation increased by {delta:.1f} cells")
else:
    print(f"Separation roughly stable (Δ = {delta:.1f} cells)")
    print("  Solitons are orbiting or oscillating in mutual wells.")

print()
print("Each soliton curves toward the other's χ-well.")
print("No Newton, no force law — just waves in a lattice.")
print()
print("Next: what about electric charge? (→ 05)")
