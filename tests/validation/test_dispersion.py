"""Task 4.4 — Klein-Gordon Dispersion Relation.

Inject a standing-wave perturbation into a uniform χ=χ₀ background
and verify the oscillation frequency satisfies ω² = c²k² + χ₀².

Uses PERIODIC boundaries so the plane wave wraps cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.simulation import Simulation

from .conftest import BACKEND, GRID

_N = GRID
_CHI0 = 19.0
# k = 2π * n / L  for mode number n on a grid of size L = N*dx = N
_MODE_NUMBERS = [1, 2, 3]


def _make_standing_wave(n: int, steps: int = 400) -> tuple[float, float]:
    """Return (omega_measured, omega_expected) for mode number *n*.

    A standing wave Ψ = A cos(k·x) is initialised with dΨ/dt = 0.  It
    oscillates at ω = √(c²k² + χ₀²).  We track Ψ at the grid center
    over time, FFT the time series, and read off the peak frequency.
    """
    k = 2 * np.pi * n / _N
    omega_expected = np.sqrt(k**2 + _CHI0**2)

    cfg = SimulationConfig(
        grid_size=_N,
        field_level=FieldLevel.REAL,
        boundary_type=BoundaryType.PERIODIC,
        chi0=_CHI0,
        dt=0.02,
    )
    sim = Simulation(cfg, backend=BACKEND)

    # Uniform chi — no soliton, no equilibrate needed.
    # Inject standing wave: Ψ(x,y,z) = A * cos(k * x)
    amp = 0.01  # small perturbation
    xs = np.arange(_N, dtype=np.float32)
    wave_1d = amp * np.cos(k * xs)
    psi = np.broadcast_to(wave_1d[:, None, None], (_N, _N, _N)).copy()
    sim.set_psi_real(psi.astype(np.float32))

    # Evolve and record center-point value each step.
    mid = _N // 2
    dt = cfg.dt
    ts: list[float] = []
    for _ in range(steps):
        sim.run(steps=1, record_metrics=False, evolve_chi=False)
        val = float(sim.psi_real[mid, mid, mid])
        ts.append(val)

    # FFT to extract dominant frequency.
    signal = np.array(ts)
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    spectrum = np.abs(np.fft.rfft(signal))
    # Skip DC bin.
    spectrum[0] = 0.0
    peak_idx = int(np.argmax(spectrum))
    omega_measured = 2 * np.pi * freqs[peak_idx]

    return omega_measured, omega_expected


class TestKGDispersion:
    """Klein-Gordon dispersion: ω² = c²k² + χ₀²."""

    @pytest.mark.parametrize("n", _MODE_NUMBERS)
    def test_dispersion_mode(self, n: int) -> None:
        """Measured ω matches KG prediction within 5 %."""
        omega_m, omega_e = _make_standing_wave(n)
        rel_err = abs(omega_m - omega_e) / omega_e
        assert rel_err < 0.05, (
            f"Mode n={n}: ω_measured={omega_m:.3f}, ω_expected={omega_e:.3f}, rel_err={rel_err:.4f}"
        )

    def test_mass_gap(self) -> None:
        """At k=0 the frequency should equal χ₀ (the mass gap)."""
        # A uniform perturbation Ψ = const oscillates at ω = χ₀.
        cfg = SimulationConfig(
            grid_size=_N,
            field_level=FieldLevel.REAL,
            boundary_type=BoundaryType.PERIODIC,
            chi0=_CHI0,
            dt=0.02,
        )
        sim = Simulation(cfg, backend=BACKEND)
        amp = 0.01
        sim.set_psi_real(np.full((_N, _N, _N), amp, dtype=np.float32))

        mid = _N // 2
        dt = cfg.dt
        steps = 400
        ts: list[float] = []
        for _ in range(steps):
            sim.run(steps=1, record_metrics=False, evolve_chi=False)
            ts.append(float(sim.psi_real[mid, mid, mid]))

        signal = np.array(ts)
        freqs = np.fft.rfftfreq(len(signal), d=dt)
        spectrum = np.abs(np.fft.rfft(signal))
        spectrum[0] = 0.0
        peak_idx = int(np.argmax(spectrum))
        omega_measured = 2 * np.pi * freqs[peak_idx]

        rel_err = abs(omega_measured - _CHI0) / _CHI0
        assert rel_err < 0.05, (
            f"Mass gap: ω_measured={omega_measured:.3f}, χ₀={_CHI0}, rel_err={rel_err:.4f}"
        )
