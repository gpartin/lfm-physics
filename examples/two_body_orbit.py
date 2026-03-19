#!/usr/bin/env python3
"""
Two-Body Gravitational Orbit
=============================

Place two solitons offset from the grid center and let GOV-01+GOV-02
dynamics create orbital motion. Each soliton's energy density |Ψ|²
sources χ wells (GOV-02) that attract the other.

We track the energy-weighted center-of-mass of each soliton to
measure orbital separation over time.

Usage:
  python two_body_orbit.py
"""

from __future__ import annotations

import numpy as np

import lfm


def com_half(psi_r: np.ndarray, axis: int, lo: int, hi: int) -> float:
    """Energy-weighted center-of-mass along axis within [lo, hi)."""
    if axis == 0:
        region = psi_r[lo:hi, :, :]
    elif axis == 1:
        region = psi_r[:, lo:hi, :]
    else:
        region = psi_r[:, :, lo:hi]

    e2 = region ** 2
    coords = np.arange(lo, hi, dtype=np.float64)
    profile = e2.sum(axis=tuple(i for i in range(3) if i != axis))
    total = profile.sum()
    if total < 1e-30:
        return (lo + hi) / 2.0
    return float(np.dot(coords, profile) / total)


def main() -> None:
    N = 64
    c = N // 2
    sep = 16  # initial separation along x-axis

    config = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.REAL,
        boundary_type=lfm.BoundaryType.FROZEN,
        report_interval=500,
    )
    sim = lfm.Simulation(config)

    # Two equal-mass solitons separated along x
    pos_a = (c - sep // 2, c, c)
    pos_b = (c + sep // 2, c, c)
    sim.place_soliton(pos_a, amplitude=8.0, sigma=4.0)
    sim.place_soliton(pos_b, amplitude=8.0, sigma=4.0)
    sim.equilibrate()

    print("Two-Body Gravitational Orbit")
    print("=" * 55)
    print(f"Grid: {N}³, initial separation: {sep} cells")
    print()

    # Track orbital separation
    trajectory: list[tuple[int, float]] = []

    def track(sim: lfm.Simulation, step: int) -> None:
        pr = sim.psi_real
        xa = com_half(pr, axis=0, lo=0, hi=c)
        xb = com_half(pr, axis=0, lo=c, hi=N)
        d = xb - xa
        trajectory.append((step, d))
        m = sim.metrics()
        print(f"  Step {step:>5d} | sep = {d:.2f} | "
              f"χ_min = {m['chi_min']:.2f} | "
              f"energy = {m['energy_total']:.4e}")

    sim.run(steps=5000, callback=track)

    if len(trajectory) < 2:
        print("Not enough data points.")
        return

    d0 = trajectory[0][1]
    d_final = trajectory[-1][1]
    d_min = min(d for _, d in trajectory)
    d_max = max(d for _, d in trajectory)

    print()
    print("-" * 55)
    print(f"Initial separation: {d0:.2f}")
    print(f"Final separation:   {d_final:.2f}")
    print(f"Range: [{d_min:.2f}, {d_max:.2f}]")

    delta = d_final - d0
    if delta < -0.5:
        print("RESULT: Solitons are approaching → gravitational attraction ✓")
    elif abs(delta) < 0.5:
        print("RESULT: Roughly stable orbit (minimal net displacement)")
    else:
        print("RESULT: Solitons drifting apart (radiation pressure dominates)")

    print()
    print("Physics: each soliton's |Ψ|² sources a χ well via GOV-02.")
    print("The other soliton's wave equation (GOV-01) curves toward")
    print("the low-χ region. No Newton's law injected — pure lattice dynamics.")


if __name__ == "__main__":
    main()
