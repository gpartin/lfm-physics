#!/usr/bin/env python
"""Experiment 05: Gravitational Redshift / Time Dilation
========================================================

CLAIM
-----
Clocks tick slower in deeper gravitational wells.  In LFM, oscillation
frequency ω(r) = χ(r), so a wave deep in a χ-well oscillates slower
than one far away.  This IS gravitational time dilation:

    GR:  ω_obs / ω_emit = √(g₀₀(obs) / g₀₀(emit))
    LFM: g₀₀ = −(χ/χ₀)²  ⇒  ω ∝ χ(r)

PROTOCOL
--------
1. Place massive soliton at centre, equilibrate χ
2. Record χ at a near position (r=5) and a far position (r=15)
3. Set Ψ = small uniform perturbation everywhere (decouples cells)
4. Evolve 600 steps with frozen χ
5. FFT the signal at each position → extract ω_near, ω_far
6. Compare ω_near/ω_far to χ_near/χ_far

PREDICTION: |ω_ratio / χ_ratio − 1| < 5%

LFM-ONLY: No metric assumed.  Redshift emerges from GOV-01 in a χ-well.
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.config_presets import gravity_only

N = 48
STEPS = 600
R_NEAR = 5
R_FAR = 15
TOLERANCE = 0.05


def run() -> dict:
    # 1. Create gravitational well
    sim = Simulation(gravity_only(grid_size=N))
    mid = N // 2
    sim.place_soliton((mid, mid, mid), amplitude=6.0)
    sim.equilibrate()

    # 2. Record chi at two positions
    p_near = (mid + R_NEAR, mid, mid)
    p_far = (mid + R_FAR, mid, mid)
    chi_near = float(np.asarray(sim.chi)[p_near])
    chi_far = float(np.asarray(sim.chi)[p_far])
    chi_ratio = chi_near / chi_far

    # 3. Uniform perturbation (∇²Ψ = 0 → pure local oscillation at ω=χ)
    psi_uniform = np.full((N, N, N), 0.001, dtype=np.float32)
    sim.set_psi_real(psi_uniform)
    sim.set_psi_real_prev(psi_uniform.copy())

    # 4. Evolve with frozen chi
    sig_near: list[float] = []
    sig_far: list[float] = []
    for _ in range(STEPS):
        sim.run(1, evolve_chi=False)
        sig_near.append(float(np.asarray(sim.psi_real)[p_near]))
        sig_far.append(float(np.asarray(sim.psi_real)[p_far]))

    # 5. FFT → dominant frequencies
    dt = sim.config.dt
    freqs = np.fft.rfftfreq(len(sig_near), d=dt)

    pwr_near = np.abs(np.fft.rfft(sig_near)) ** 2
    f_near = float(freqs[np.argmax(pwr_near[1:]) + 1])
    omega_near = 2.0 * np.pi * f_near

    pwr_far = np.abs(np.fft.rfft(sig_far)) ** 2
    f_far = float(freqs[np.argmax(pwr_far[1:]) + 1])
    omega_far = 2.0 * np.pi * f_far

    omega_ratio = omega_near / omega_far
    error = abs(omega_ratio / chi_ratio - 1.0)
    passed = error < TOLERANCE

    # Relative redshift z = (ω_far − ω_near) / ω_near
    z_grav = (omega_far - omega_near) / omega_near

    print("═" * 60)
    print("EXPERIMENT 05: Gravitational Redshift / Time Dilation")
    print("═" * 60)
    print("Equations: GOV-01 in a frozen χ-well. No GR metric assumed.\n")
    print("MEASUREMENTS:")
    print(f"  χ at r={R_NEAR:2d} cells   = {chi_near:.4f}")
    print(f"  χ at r={R_FAR:2d} cells  = {chi_far:.4f}")
    print(f"  χ_near / χ_far        = {chi_ratio:.6f}")
    print(f"  ω at r={R_NEAR:2d} cells   = {omega_near:.4f}")
    print(f"  ω at r={R_FAR:2d} cells  = {omega_far:.4f}")
    print(f"  ω_near / ω_far        = {omega_ratio:.6f}\n")
    print("COMPARISON:")
    print(f"  ω_ratio / χ_ratio   = {omega_ratio / chi_ratio:.6f}")
    print("  Prediction: 1.000000  (ω ∝ χ)")
    print(f"  Error: {error:.2%}")
    print(f"  Gravitational redshift z = {z_grav:.6f}\n")
    print("PHYSICS:")
    print("  Deep well → lower χ → slower oscillation → clock slows")
    print("  Light escaping upward is blueshifted (higher ω far away)")
    verdict = (
        "✅ PASS — Gravitational redshift CONFIRMED (ω ∝ χ)"
        if passed
        else f"❌ FAIL — ratio off by {error:.2%}"
    )
    print(f"\nVERDICT: {verdict}")
    print("═" * 60)

    return {
        "name": "Grav. redshift",
        "measured": f"ω_ratio = {omega_ratio:.4f}",
        "expected": f"χ_ratio = {chi_ratio:.4f}",
        "error": f"{error:.2%}",
        "passed": passed,
    }


if __name__ == "__main__":
    run()
