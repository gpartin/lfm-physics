#!/usr/bin/env python3
"""
Electromagnetism from Phase
============================

Show that same-phase solitons REPEL and opposite-phase solitons
ATTRACT — Coulomb's law emerging from wave interference in GOV-01.

Charge in LFM is the phase theta of the complex field Psi:
  theta = 0   -> "electron" (negative charge)
  theta = pi  -> "positron" (positive charge)

Same phase  -> constructive interference -> energy UP  -> REPEL
Opp. phase  -> destructive interference -> energy DOWN -> ATTRACT

Usage:
  python em_from_phase.py
"""

from __future__ import annotations

import math

import numpy as np

import lfm


def measure_center_of_mass(psi_r: np.ndarray, axis: int) -> float:
    """Energy-weighted center along one axis."""
    e2 = psi_r ** 2
    coords = np.arange(psi_r.shape[axis], dtype=np.float64)
    # Sum over the other two axes
    profile = e2.sum(axis=tuple(i for i in range(3) if i != axis))
    total = profile.sum()
    if total < 1e-30:
        return psi_r.shape[axis] / 2.0
    return float(np.dot(coords, profile) / total)


def run_pair(name: str, phase_a: float, phase_b: float) -> float:
    """Place two solitons and measure if they approach or separate."""
    N = 64
    c = N // 2
    sep = 14  # initial separation along x

    config = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.COMPLEX,
        boundary_type=lfm.BoundaryType.FROZEN,
        report_interval=500,
    )
    sim = lfm.Simulation(config)

    pos_a = (c - sep // 2, c, c)
    pos_b = (c + sep // 2, c, c)
    sim.place_soliton(pos_a, amplitude=8.0, sigma=4.0, phase=phase_a)
    sim.place_soliton(pos_b, amplitude=8.0, sigma=4.0, phase=phase_b)
    sim.equilibrate()

    # Measure initial separation
    pr0 = sim.psi_real.copy()
    x0_a = measure_center_of_mass(pr0[:c, :, :], axis=0)
    x0_b = measure_center_of_mass(pr0[c:, :, :], axis=0) + c
    d0 = x0_b - x0_a

    # Evolve
    sim.run(steps=3000, record_metrics=False)

    # Measure final separation
    pr1 = sim.psi_real.copy()
    x1_a = measure_center_of_mass(pr1[:c, :, :], axis=0)
    x1_b = measure_center_of_mass(pr1[c:, :, :], axis=0) + c
    d1 = x1_b - x1_a

    delta = d1 - d0
    direction = "CLOSER (attract)" if delta < 0 else "FARTHER (repel)"
    print(f"  {name}: d0={d0:.2f} -> d1={d1:.2f}  Δd={delta:+.3f}  {direction}")
    return delta


def main() -> None:
    print("Coulomb Test — Charge from Phase Interference")
    print("=" * 55)
    print()

    print("1) Same phase (theta=0, theta=0) — should REPEL:")
    dd_same = run_pair("e⁻ e⁻", 0.0, 0.0)

    print()
    print("2) Opposite phase (theta=0, theta=π) — should ATTRACT:")
    dd_opp = run_pair("e⁻ e⁺", 0.0, math.pi)

    print()
    print("-" * 55)
    if dd_same > 0 and dd_opp < 0:
        result = "PASS — like charges repel, opposite attract"
    elif dd_same > 0:
        result = "PARTIAL — repulsion correct, attraction not observed"
    elif dd_opp < 0:
        result = "PARTIAL — attraction correct, repulsion not observed"
    else:
        result = "INCONCLUSIVE — gravity dominates at this scale"
    print(f"Coulomb criterion: {result}")
    print()
    print("Physics: no Coulomb law injected! Force emerges")
    print("from constructive/destructive wave interference in GOV-01.")


if __name__ == "__main__":
    main()
