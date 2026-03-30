#!/usr/bin/env python
"""Experiment 01: Mass-Energy Equivalence (E = mc²)
====================================================

CLAIM
-----
A soliton's oscillation frequency ω equals the local χ field value.
Since E = ℏω and m = ℏχ/c², this gives E = mc² in natural units (ℏ = c = 1).

PROTOCOL
--------
1. Place soliton at grid center
2. Equilibrate χ via Poisson solve (GOV-04 quasi-static limit)
3. Run 400 steps, record Ψ_real at soliton center each step
4. FFT → extract dominant frequency ω
5. Compare ω to χ at soliton center (the local mass term)

PREDICTION: ω / χ_local ≈ 1.0 (within 5%)

LFM-ONLY: No E=mc² injected. Measured from GOV-01 + GOV-02 dynamics.
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.config_presets import gravity_only
from lfm.constants import CHI0

N = 32
STEPS = 400
TOLERANCE = 0.05


def run() -> dict:
    sim = Simulation(gravity_only(grid_size=N))
    mid = N // 2
    sim.place_soliton((mid, mid, mid), amplitude=6.0)
    sim.equilibrate()

    chi_local = float(np.asarray(sim.chi)[mid, mid, mid])

    # Record oscillation at soliton center
    signal = []
    for _ in range(STEPS):
        sim.run(1)
        signal.append(float(np.asarray(sim.psi_real)[mid, mid, mid]))

    # FFT → dominant angular frequency
    dt = sim.config.dt
    freqs = np.fft.rfftfreq(len(signal), d=dt)  # cycles per time unit
    power = np.abs(np.fft.rfft(signal)) ** 2
    f_peak = float(freqs[np.argmax(power[1:]) + 1])
    omega = 2.0 * np.pi * f_peak  # angular frequency

    ratio = omega / chi_local
    error = abs(ratio - 1.0)
    passed = error < TOLERANCE

    print("═" * 60)
    print("EXPERIMENT 01: Mass-Energy Equivalence (E = mc²)")
    print("═" * 60)
    print("Equations: GOV-01 + GOV-02 only. No E=mc² injected.\n")
    print("MEASUREMENTS:")
    print(f"  Oscillation frequency ω  = {omega:.4f}")
    print(f"  Local chi field χ_local  = {chi_local:.4f}")
    print(f"  Background χ₀           = {CHI0}\n")
    print("COMPARISON:")
    print(f"  ω / χ_local = {ratio:.6f}")
    print("  Prediction:   1.000000")
    print(f"  Error:        {error:.2%}\n")
    verdict = "✅ PASS — E = ℏω = ℏχ = mc² CONFIRMED" if passed else "❌ FAIL"
    print(f"VERDICT: {verdict}")
    print("═" * 60)

    return {
        "name": "E = mc²",
        "measured": f"ω/χ = {ratio:.4f}",
        "expected": "1.0000",
        "error": f"{error:.2%}",
        "passed": passed,
    }


if __name__ == "__main__":
    run()
