#!/usr/bin/env python
"""Experiment 03: historical initial phase-overlap control.

Places Gaussian seeds and measures sum(|Psi|^2) WITHOUT evolution.
Same/opposite phase are preparation labels, not established signed charges.
The norm difference is not a Hamiltonian, potential, force, or acceleration.
A positive decreasing overlap does not establish inverse-square scaling.
The historical filename and measurements are retained for reproducibility.
This legacy implementation is not the authoritative v39.0.2 model.
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

        # Same phase: constructive initial overlap
        sim_s = Simulation(gravity_em(grid_size=N))
        sim_s.place_soliton(pos1, amplitude=AMP, sigma=SIGMA, phase=0.0)
        sim_s.place_soliton(pos2, amplitude=AMP, sigma=SIGMA, phase=0.0)
        psi_sq_same = _total_psi_sq(sim_s)

        # Opposite phase: destructive initial overlap
        sim_o = Simulation(gravity_em(grid_size=N))
        sim_o.place_soliton(pos1, amplitude=AMP, sigma=SIGMA, phase=0.0)
        sim_o.place_soliton(pos2, amplitude=AMP, sigma=SIGMA, phase=np.pi)
        psi_sq_opp = _total_psi_sq(sim_o)

        delta = psi_sq_same - psi_sq_opp
        results.append({"sep": sep, "same": psi_sq_same, "opp": psi_sq_opp, "delta": delta})

    # Check the initial norm difference, not an interaction energy
    all_positive = all(r["delta"] > 0 for r in results)
    # Check decreasing overlap
    decreasing = all(
        results[i]["delta"] >= results[i + 1]["delta"] for i in range(len(results) - 1)
    )
    # Closest separation should have significant effect
    max_effect = results[0]["delta"] / results[0]["same"]

    passed = all_positive and decreasing

    print("═" * 60)
    print("EXPERIMENT 03: Historical Phase-Overlap Control")
    print("═" * 60)
    print("Initial conditions only: no evolution or force measurement.\n")
    print("MEASUREMENTS:")
    print(f"  {'Sep':>4s}  {'|Ψ|²_same':>12s}  {'|Ψ|²_opp':>12s}  {'Δ(norm)':>10s}")
    print("  " + "─" * 44)
    for r in results:
        print(f"  {r['sep']:4d}  {r['same']:12.2f}  {r['opp']:12.2f}  {r['delta']:10.2f}")
    print()
    print("ANALYSIS:")
    print(f"  All Δ > 0 (same > opposite)?  {'YES' if all_positive else 'NO'}")
    print(f"  Δ decreasing with distance?   {'YES' if decreasing else 'NO'}")
    print(f"  Max norm difference: {max_effect:.2%} of total\n")
    print("SCOPE: Initial interference only; Coulomb and Maxwell NOT ESTABLISHED.")
    verdict = (
        "PASS — initial interference control only"
        if passed
        else "FAIL — initial overlap criterion"
    )
    print(f"VERDICT: {verdict}")
    print("═" * 60)

    return {
        "name": "Initial interference",
        "measured": f"Δ_max = {results[0]['delta']:.1f}",
        "expected": "Δ > 0, decreasing",
        "error": "OK" if passed else "WRONG",
        "passed": passed,  # Suite compatibility: control gate only.
        "scope": "initial_interference_control",
        "coulomb_status": "NOT_ESTABLISHED",
        "maxwell_status": "NOT_ESTABLISHED",
        "evolution_steps": 0,
        "measurements": results,
    }


if __name__ == "__main__":
    run()
