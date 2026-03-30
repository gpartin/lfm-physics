#!/usr/bin/env python
"""Experiment 06: Mass Hierarchy (well depth ∝ soliton energy)
==============================================================

CLAIM
-----
In LFM, particle mass corresponds to soliton energy, which creates
a proportionally deeper χ-well via GOV-02.  Heavier solitons produce
deeper wells: higher amplitude → larger |Ψ|² → lower χ_min.

This is the mechanism behind m_e < m_μ < m_p: the mass hierarchy
emerges from the eigenmode structure, where each particle's angular
quantum number l determines its amplitude and well depth.

PROTOCOL
--------
Place three solitons with increasing amplitude (representing light,
medium, heavy particles) at the centre of separate grids.  Equilibrate
each via GOV-02 (Poisson solver) and verify:
  1. χ_min monotonically decreases with amplitude (deeper well = heavier)
  2. The library's electron catalog entry has a shallower well than
     a high-amplitude soliton (mass hierarchy from the framework)

PREDICTIONS:
  - χ_min(low amp) > χ_min(med amp) > χ_min(high amp)
  - electron (catalog) has shallower well than heavy soliton

LFM-ONLY: Mass = soliton eigenmode energy.  Wells from GOV-02.
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.config_presets import gravity_only

N = 48
AMPLITUDES = [3.0, 6.0, 12.0]  # light, medium, heavy solitons
LABELS = ["light (A=3)", "medium (A=6)", "heavy (A=12)"]


def run() -> dict:
    results: list[dict] = []
    mid = N // 2

    for amp, label in zip(AMPLITUDES, LABELS, strict=False):
        sim = Simulation(gravity_only(grid_size=N))
        sim.place_soliton((mid, mid, mid), amplitude=amp, sigma=3.0)
        sim.equilibrate()
        chi_min = float(np.asarray(sim.chi).min())
        psi_sq = float(np.asarray(sim.psi_real)[mid, mid, mid] ** 2)
        results.append({"label": label, "amp": amp, "chi_min": chi_min, "psi_sq": psi_sq})

    # Also test catalog electron
    sim_e = Simulation(gravity_only(grid_size=N))
    sim_e.place_particle("electron", (mid, mid, mid))
    sim_e.equilibrate()
    chi_min_e = float(np.asarray(sim_e.chi).min())

    # Verify monotonic ordering: higher amplitude → deeper well
    chi_ordered = all(
        results[i]["chi_min"] > results[i + 1]["chi_min"] for i in range(len(results) - 1)
    )
    # Verify electron (standard catalog) is lighter than heavy soliton
    electron_lighter = chi_min_e > results[-1]["chi_min"]
    passed = chi_ordered and electron_lighter

    print("═" * 60)
    print("EXPERIMENT 06: Mass Hierarchy (well depth ∝ soliton energy)")
    print("═" * 60)
    print("Equations: GOV-02 equilibrium. More energy → deeper χ-well.\n")
    print("MEASUREMENTS:")
    print(f"  {'Soliton':>18s}  {'Amplitude':>9s}  {'χ_min':>8s}  {'|Ψ|²_ctr':>10s}")
    print("  " + "─" * 50)
    for r in results:
        print(f"  {r['label']:>18s}  {r['amp']:9.1f}  {r['chi_min']:8.4f}  {r['psi_sq']:10.4f}")
    print(f"  {'electron (catalog)':>18s}  {'--':>9s}  {chi_min_e:8.4f}  {'--':>10s}")
    print()
    print("ORDERING:")
    print(f"  χ_min monotonically decreasing?  {'YES' if chi_ordered else 'NO'}")
    print(f"  electron lighter than A=12?       {'YES' if electron_lighter else 'NO'}")
    print()
    print("PHYSICS:")
    print("  Higher amplitude → more |Ψ|² → GOV-02 drives χ lower")
    print("  This is how m_e < m_μ < m_p arises: eigenmode energy ordering")
    verdict = (
        "✅ PASS — Mass hierarchy CONFIRMED (deeper well = heavier)"
        if passed
        else "❌ FAIL — Ordering broken"
    )
    print(f"\nVERDICT: {verdict}")
    print("═" * 60)

    return {
        "name": "Mass hierarchy",
        "measured": ", ".join(f"χ={r['chi_min']:.2f}" for r in results),
        "expected": "monotonically decreasing χ_min",
        "error": f"ordered={'YES' if chi_ordered else 'NO'}",
        "passed": passed,
    }


if __name__ == "__main__":
    run()
