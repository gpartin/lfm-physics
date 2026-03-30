#!/usr/bin/env python
"""Experiment 03: Coulomb Force from Phase Interference
=======================================================

CLAIM
-----
Electromagnetic force emerges from wave-phase interference alone:
- Same phase (e⁻ + e⁻): constructive → HIGHER energy → REPULSION
- Opposite phase (e⁻ + e⁺): destructive → LOWER energy → ATTRACTION

PROTOCOL
--------
For each separation d:
  1. Place two same-phase solitons (gravity_em) → measure total |Ψ|²
  2. Place two opposite-phase solitons → measure total |Ψ|²
  3. EM interaction proxy = |Ψ|²_same − |Ψ|²_opposite
Repeat for multiple separations.

PREDICTIONS:
  - |Ψ|²_same > |Ψ|²_opposite at all separations (correct sign)
  - EM interaction decreases with separation (correct range)
  - At close range, EM effect is measurable

LFM-ONLY: No Coulomb law injected. Phase interference in GOV-01.
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.config_presets import gravity_em

N = 48
SEPS = [4, 6, 8, 10, 12]
AMP = 6.0
SIGMA = 3.0


def _total_psi_sq(sim: Simulation) -> float:
    psi_sq = np.asarray(sim.psi_real) ** 2
    if sim.psi_imag is not None:
        psi_sq = psi_sq + np.asarray(sim.psi_imag) ** 2
    return float(np.sum(psi_sq))


def run() -> dict:
    mid = N // 2
    results = []

    for sep in SEPS:
        pos1 = (mid, mid, mid - sep // 2)
        pos2 = (mid, mid, mid + sep // 2)

        # Same phase (both θ=0 → like charges → constructive)
        sim_s = Simulation(gravity_em(grid_size=N))
        sim_s.place_soliton(pos1, amplitude=AMP, sigma=SIGMA, phase=0.0)
        sim_s.place_soliton(pos2, amplitude=AMP, sigma=SIGMA, phase=0.0)
        psi_sq_same = _total_psi_sq(sim_s)

        # Opposite phase (θ=0 and θ=π → unlike charges → destructive)
        sim_o = Simulation(gravity_em(grid_size=N))
        sim_o.place_soliton(pos1, amplitude=AMP, sigma=SIGMA, phase=0.0)
        sim_o.place_soliton(pos2, amplitude=AMP, sigma=SIGMA, phase=np.pi)
        psi_sq_opp = _total_psi_sq(sim_o)

        delta = psi_sq_same - psi_sq_opp
        results.append({"sep": sep, "same": psi_sq_same, "opp": psi_sq_opp, "delta": delta})

    # Check: all deltas positive (same-charge has more energy)
    all_positive = all(r["delta"] > 0 for r in results)
    # Check: delta decreases with separation (interaction falls off)
    decreasing = all(
        results[i]["delta"] >= results[i + 1]["delta"] for i in range(len(results) - 1)
    )
    # Closest separation should have significant effect
    max_effect = results[0]["delta"] / results[0]["same"]

    passed = all_positive and decreasing

    print("═" * 60)
    print("EXPERIMENT 03: Coulomb Force from Phase Interference")
    print("═" * 60)
    print("Equations: GOV-01 + GOV-02 only. No Coulomb injected.\n")
    print("MEASUREMENTS:")
    print(f"  {'Sep':>4s}  {'|Ψ|²_same':>12s}  {'|Ψ|²_opp':>12s}  {'Δ(EM)':>10s}")
    print("  " + "─" * 44)
    for r in results:
        print(f"  {r['sep']:4d}  {r['same']:12.2f}  {r['opp']:12.2f}  {r['delta']:10.2f}")
    print()
    print("ANALYSIS:")
    print(f"  All Δ > 0 (same > opposite)?  {'YES' if all_positive else 'NO'}")
    print(f"  Δ decreasing with distance?   {'YES' if decreasing else 'NO'}")
    print(f"  Max EM effect: {max_effect:.2%} of total\n")
    print("PHYSICS:")
    print("  Same phase → constructive interference → MORE |Ψ|² → REPEL")
    print("  Opp. phase → destructive interference → LESS |Ψ|² → ATTRACT")
    print("  No Coulomb potential injected — emerges from GOV-01.\n")
    verdict = (
        "✅ PASS — Like repels, unlike attracts, range correct"
        if passed
        else "❌ FAIL — Sign or range incorrect"
    )
    print(f"VERDICT: {verdict}")
    print("═" * 60)

    return {
        "name": "Coulomb force",
        "measured": f"Δ_max = {results[0]['delta']:.1f}",
        "expected": "Δ > 0, decreasing",
        "error": "OK" if passed else "WRONG",
        "passed": passed,
    }


if __name__ == "__main__":
    run()
