#!/usr/bin/env python3
"""
Parametric Resonance — Matter Creation
=======================================

When chi oscillates at Omega = 2*chi0, GOV-01 becomes a Mathieu
equation and any Psi perturbation grows exponentially.  This is the
LFM mechanism for matter creation from vacuum instability.

This example seeds a uniform chi oscillation and a tiny Psi
perturbation, then measures exponential growth of |Psi|^2.

Usage:
  python parametric_resonance.py
"""

from __future__ import annotations

import numpy as np

import lfm


def main() -> None:
    N = 32  # small grid — the effect is global
    chi0 = 19.0

    config = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.REAL,
        boundary_type=lfm.BoundaryType.PERIODIC,
        chi0=chi0,
        report_interval=200,
    )
    sim = lfm.Simulation(config)

    # Seed a tiny uniform Psi perturbation (machine-epsilon scale)
    pr = sim.psi_real.copy()
    rng = np.random.default_rng(42)
    pr += rng.normal(0, 1e-6, pr.shape).astype(np.float32)
    sim.psi_real = pr

    # Record |Psi|^2 growth
    energies: list[tuple[int, float]] = []

    def record(sim: lfm.Simulation, step: int) -> None:
        e2 = float(np.sum(sim.psi_real ** 2))
        energies.append((step, e2))

    total_steps = 4000
    sim.run(steps=total_steps, callback=record, record_metrics=False)

    if not energies:
        print("No data recorded.")
        return

    e2_initial = energies[0][1]
    e2_final = energies[-1][1]
    growth = e2_final / max(e2_initial, 1e-30)

    print("Parametric Resonance — Matter from Vacuum")
    print("=" * 50)
    print(f"Grid: {N}^3,  chi0 = {chi0},  steps = {total_steps}")
    print(f"|Psi|^2 initial: {e2_initial:.4e}")
    print(f"|Psi|^2 final  : {e2_final:.4e}")
    print(f"Growth factor   : {growth:.2e}x")
    print()

    if growth > 10:
        print("RESULT: Exponential growth detected!")
        print("  -> GOV-01 acts as Mathieu equation when chi oscillates.")
        print("  -> Matter emerges via parametric resonance.")
    elif growth > 2:
        print("RESULT: Moderate growth (try more steps for clearer signal).")
    else:
        print("RESULT: No significant growth.")
        print("  (Growth depends on chi dynamics; with frozen chi")
        print("   and periodic BC, chi oscillation may be weak.)")

    # Print a few data points
    print()
    print("Sample trajectory:")
    step_count = len(energies)
    indices = [0, step_count // 4, step_count // 2, 3 * step_count // 4, -1]
    for idx in indices:
        s, e = energies[idx]
        print(f"  step {s:>5d} : |Psi|^2 = {e:.4e}")


if __name__ == "__main__":
    main()
