#!/usr/bin/env python
"""Experiment 02: Inverse-Square Gravitational Force
====================================================

CLAIM
-----
A massive soliton creates an emergent gravitational force that
falls off as 1/r² — the Newtonian inverse-square law.

DERIVATION
----------
GOV-02 quasi-static limit: ∇²δχ = κ(|Ψ|² − E₀²)
For a localised source, outside the source radius:
    δχ(r) ∝ 1/r  ⟹  F = −∇χ ∝ 1/r².
This IS Newton's law: F = −GM/r² in χ-space.

PROTOCOL
--------
1. Place single soliton at grid centre  (GOV-02 source)
2. Equilibrate χ via Poisson solve       (GOV-04 limit)
3. Measure |∇χ| along 6 cardinal rays   (emergent force)
4. Fit power law: |∇χ(r)| = A · r^n     (far-field, r > source)
5. Extract exponent n

PREDICTION: n ≈ −2.0 (inverse-square law), R² > 0.99

LFM-ONLY: No Newton's law injected.  Force EMERGES from GOV-02.
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.analysis.observables import fit_power_law
from lfm.config_presets import gravity_only

N = 96
R_MIN = 8.0  # well outside soliton core (sigma ≈ 3)
R_MAX = 25.0  # well inside frozen boundary (at ~5 cells from edge)
TOLERANCE_EXP = 0.15  # |n − (−2)| < 0.15


def _measure_radial_force(chi: np.ndarray, mid: int) -> tuple[np.ndarray, np.ndarray]:
    """Measure |∇χ| averaged over 6 cardinal directions."""
    radii: list[float] = []
    forces: list[float] = []
    max_r = min(mid - 2, N - mid - 2)  # stay inside grid
    for r in range(3, max_r):
        f_vals: list[float] = []
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            ix, iy, iz = mid + r * dx, mid + r * dy, mid + r * dz
            if 1 <= ix < N - 1 and 1 <= iy < N - 1 and 1 <= iz < N - 1:
                g = (chi[ix + dx, iy + dy, iz + dz] - chi[ix - dx, iy - dy, iz - dz]) / 2.0
                f_vals.append(abs(g))
        if f_vals:
            radii.append(float(r))
            forces.append(float(np.mean(f_vals)))
    return np.array(radii), np.array(forces)


def run() -> dict:
    sim = Simulation(gravity_only(grid_size=N))
    mid = N // 2
    sim.place_soliton((mid, mid, mid), amplitude=6.0, sigma=3.0)
    sim.equilibrate()

    chi = np.asarray(sim.chi)
    radii, forces = _measure_radial_force(chi, mid)

    exponent, r_sq = fit_power_law(radii, forces, r_min=R_MIN, r_max=R_MAX)

    error = abs(exponent - (-2.0))
    passed = error < TOLERANCE_EXP and r_sq > 0.99

    print("═" * 60)
    print("EXPERIMENT 02: Inverse-Square Gravitational Force")
    print("═" * 60)
    print("Equations: GOV-02 Poisson limit. No Newton injected.\n")
    print("MEASUREMENTS:")
    print("  Force measured along 6 cardinal rays, averaged")
    print(f"  Fit range: r ∈ [{R_MIN}, {R_MAX}] cells")
    print(f"  Power-law exponent n  = {exponent:.4f}")
    print(f"  R² (goodness of fit)  = {r_sq:.6f}\n")
    print("COMPARISON:")
    print(f"  Measured exponent:  {exponent:.4f}")
    print("  Newton prediction:  −2.0000 (F = −GM/r²)")
    print(f"  Error: {error:.4f}\n")
    # Show F*r² constancy check
    mask = (radii >= R_MIN) & (radii <= R_MAX)
    fr2 = forces[mask] * radii[mask] ** 2
    cv = float(np.std(fr2) / np.mean(fr2)) * 100
    print(f"  F·r² constancy (CV): {cv:.2f}%")
    print(f"  F·r² mean: {np.mean(fr2):.4f}  std: {np.std(fr2):.4f}\n")
    verdict = (
        "✅ PASS — Inverse-square law EMERGES from GOV-02"
        if passed
        else f"❌ FAIL — exponent {exponent:.3f}, R²={r_sq:.3f}"
    )
    print(f"VERDICT: {verdict}")
    print("═" * 60)

    return {
        "name": "Inverse-square force",
        "measured": f"n = {exponent:.3f}",
        "expected": "n = −2.000",
        "error": f"{error:.3f}",
        "passed": passed,
    }


if __name__ == "__main__":
    run()
