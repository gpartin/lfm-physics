#!/usr/bin/env python3
"""
Soliton Gravity
===============

Place a single Gaussian soliton on the lattice and watch
GOV-02 carve a chi well around it. This is the simplest possible
LFM experiment: one lump of energy, and gravity emerges.

What to observe:
  - chi drops below 19 at the soliton center (gravitational well)
  - Energy is conserved (total stays constant within ~0.3%)
  - The soliton is a standing wave trapped in its own well

Usage:
  python soliton_gravity.py
"""

from __future__ import annotations

import lfm


def main() -> None:
    N = 64

    config = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.REAL,
        boundary_type=lfm.BoundaryType.FROZEN,
        report_interval=500,
    )

    sim = lfm.Simulation(config)

    # Place a single soliton at grid center
    center = (N // 2, N // 2, N // 2)
    sim.place_soliton(center, amplitude=8.0, sigma=5.0)
    sim.equilibrate()

    m0 = sim.metrics()
    print(f"Initial:  chi_min = {m0['chi_min']:.2f},  "
          f"energy = {m0['energy_total']:.4e}")

    # Evolve
    def report(sim: lfm.Simulation, step: int) -> None:
        m = sim.metrics()
        print(f"  Step {step:>6d} | chi_min = {m['chi_min']:.3f} | "
              f"wells = {m['well_fraction']*100:.1f}% | "
              f"energy = {m['energy_total']:.4e}")

    sim.run(steps=5000, callback=report)

    mf = sim.metrics()
    drift = abs(mf["energy_total"] - m0["energy_total"]) / max(abs(m0["energy_total"]), 1e-30) * 100
    print(f"\nFinal:    chi_min = {mf['chi_min']:.2f},  "
          f"energy = {mf['energy_total']:.4e}")
    print(f"Energy drift: {drift:.3f}%")
    print(f"Wells (chi < 17): {mf['well_fraction']*100:.1f}%")
    print(f"Clusters: {mf['n_clusters']}")


if __name__ == "__main__":
    main()
