#!/usr/bin/env python
"""Experiment 04: Klein-Gordon Dispersion Relation
==================================================

CLAIM
-----
GOV-01 produces the Klein-Gordon dispersion relation on the discrete lattice:
    ω² = c² · k_eff² + m²
where k_eff² = 4sin²(kΔx/2)/Δx² is the discrete Laplacian eigenvalue and
m = χ is the effective mass from the background chi field.

In the continuum limit (kΔx → 0), k_eff → k and this becomes the standard
relativistic dispersion ω² = c²k² + m².

PROTOCOL
--------
1. Set uniform χ = m_eff (low value so k² term is significant vs m²)
2. For each mode n: inject standing wave sin(2πnx/L) on periodic lattice
3. Evolve GOV-01 with frozen χ → pure wave dynamics
4. Extract Fourier mode coefficient vs time → FFT → ω
5. Fit ω² vs k_eff² → slope = c², intercept = m²

PREDICTIONS:
  - Slope c² ≈ 1.0 (within 10%)
  - Intercept ≈ m_eff² (within 5%)
  - R² > 0.99

LFM-ONLY: Dispersion measured from GOV-01 lattice dynamics.
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation, SimulationConfig
from lfm.config import BoundaryType, FieldLevel

N = 64
STEPS = 10000  # long run for fine FFT resolution
SAMPLE_EVERY = 5  # sample every 5 steps (Nyquist still >> ω≈3)
M_EFF = 3.0  # effective mass (lower than χ₀=19 so k² term is significant)
MODES = [4, 8, 12, 16, 20]  # wide k spread for strong dispersion signal


def run() -> dict:
    L = N
    dt = 0.02
    sample_dt = dt * SAMPLE_EVERY
    omegas: list[float] = []
    k_values: list[float] = []
    k_eff_sq: list[float] = []

    for n in MODES:
        k = 2.0 * np.pi * n / L
        k_values.append(k)
        # Discrete Laplacian eigenvalue for the 19-point stencil (axis-aligned mode)
        k_eff_sq.append(4.0 * np.sin(k / 2.0) ** 2)

        config = SimulationConfig(
            grid_size=N,
            field_level=FieldLevel.REAL,
            boundary_type=BoundaryType.PERIODIC,
            chi0=M_EFF,
            dt=dt,
        )
        sim = Simulation(config)
        # Override boundary mask for truly periodic evolution
        sim._evolver.boundary_mask[:] = 0

        # Inject standing wave: Ψ = A·sin(kx), start from rest
        x = np.arange(N, dtype=np.float32)
        wave_1d = 0.01 * np.sin(k * x)
        psi = np.zeros((N, N, N), dtype=np.float32)
        psi[:] = wave_1d[:, None, None]
        sim.set_psi_real(psi)
        sim.set_psi_real_prev(psi)  # ∂Ψ/∂t = 0 → cos(ωt) oscillation

        # Build sin basis for mode projection (avoids probe-on-node issues)
        basis = np.sin(k * x).astype(np.float32)

        # Evolve and extract Fourier sine coefficient vs time
        signal: list[float] = []
        for step in range(STEPS):
            sim.run(1, evolve_chi=False)
            if (step + 1) % SAMPLE_EVERY == 0:
                psi_1d = np.asarray(sim.psi_real)[:, N // 2, N // 2]
                coeff = float((2.0 / N) * np.sum(psi_1d * basis))
                signal.append(coeff)

        # FFT → dominant angular frequency
        freqs = np.fft.rfftfreq(len(signal), d=sample_dt)
        power = np.abs(np.fft.rfft(signal)) ** 2
        f_peak = float(freqs[np.argmax(power[1:]) + 1])
        omega = 2.0 * np.pi * f_peak
        omegas.append(omega)

    # Fit ω² = slope · k_eff² + intercept
    keff2 = np.array(k_eff_sq)
    w_arr = np.array(omegas)
    w2 = w_arr**2
    k_arr = np.array(k_values)

    A = np.column_stack([keff2, np.ones_like(keff2)])
    coeffs, _, _, _ = np.linalg.lstsq(A, w2, rcond=None)
    slope, intercept = coeffs

    pred = A @ coeffs
    ss_res = float(np.sum((w2 - pred) ** 2))
    ss_tot = float(np.sum((w2 - w2.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    c_sq_error = abs(slope - 1.0) / 1.0
    m_sq = M_EFF**2
    m_sq_error = abs(intercept - m_sq) / m_sq
    passed = c_sq_error < 0.10 and m_sq_error < 0.05 and r_sq > 0.99

    print("═" * 60)
    print("EXPERIMENT 04: Klein-Gordon Dispersion (ω² = c²k_eff² + m²)")
    print("═" * 60)
    print(f"Equations: GOV-01 with frozen χ = {M_EFF}. Pure wave dynamics.")
    print("k_eff² = 4sin²(kΔx/2) = discrete Laplacian eigenvalue.\n")
    print("MEASUREMENTS:")
    print(f"  {'n':>4s}  {'k':>7s}  {'k_eff²':>8s}  {'ω':>7s}  {'ω²':>8s}  {'predicted':>9s}")
    print("  " + "─" * 52)
    for i, n in enumerate(MODES):
        expected = keff2[i] + m_sq
        print(
            f"  {n:4d}  {k_arr[i]:7.4f}  {keff2[i]:8.4f}  "
            f"{w_arr[i]:7.4f}  {w2[i]:8.4f}  {expected:9.4f}"
        )

    print(f"\nLINEAR FIT: ω² = {slope:.4f}·k_eff² + {intercept:.4f}")
    print(f"  Slope  c²   = {slope:.4f}  (expected 1.0, error {c_sq_error:.2%})")
    print(f"  Intercept   = {intercept:.4f}  (expected m²={m_sq:.1f}, error {m_sq_error:.2%})")
    print(f"  R²          = {r_sq:.6f}\n")
    verdict = (
        "✅ PASS — Klein-Gordon dispersion EMERGES from GOV-01"
        if passed
        else f"❌ FAIL — c²={slope:.3f}, m²={intercept:.1f}, R²={r_sq:.3f}"
    )
    print(f"VERDICT: {verdict}")
    print("═" * 60)

    return {
        "name": "KG dispersion",
        "measured": f"c²={slope:.3f}, m²={intercept:.1f}",
        "expected": f"c²=1.000, m²={m_sq:.1f}",
        "error": f"c²:{c_sq_error:.2%}, m²:{m_sq_error:.2%}",
        "passed": passed,
    }


if __name__ == "__main__":
    run()


if __name__ == "__main__":
    run()
