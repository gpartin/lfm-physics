"""Task 4.6 — Gravitational Time Dilation.

In LFM, oscillation frequency ω ~ χ.  A probe in a deep χ-well (near a
massive body) oscillates at lower frequency than one in vacuum (χ ≈ χ₀).

Strategy:
1. Place a massive soliton, equilibrate χ → creates a χ-well.
2. Wipe wave content, inject two small Gaussians at *near* and *far* radii.
3. Evolve with ``evolve_chi=False`` (frozen geometry).
4. Measure oscillation frequencies at both probe centres.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm.config_presets import gravity_only
from lfm.simulation import Simulation

from .conftest import BACKEND, GRID_TWO

_N = GRID_TWO
_CHI0 = 19.0
_STEPS = 800
_SIGMA = 2.0


def _measure_frequency(signal: np.ndarray, dt: float) -> float:
    """Peak angular frequency from FFT, ignoring DC."""
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum[0] = 0.0
    peak_idx = int(np.argmax(spectrum))
    return 2 * np.pi * freqs[peak_idx]


def _build_probe(center: tuple[int, int, int], shape: tuple[int, ...]) -> np.ndarray:
    """Gaussian blob centred at *center*."""
    n = shape[-1] if len(shape) == 3 else shape[-1]
    xs = np.arange(n, dtype=np.float32)
    xx = xs.reshape(-1, 1, 1)
    yy = xs.reshape(1, -1, 1)
    zz = xs.reshape(1, 1, -1)
    g = np.exp(
        -((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2) / (2 * _SIGMA**2)
    ).astype(np.float32)
    if len(shape) == 4:
        g = g[np.newaxis]
    return g


class TestTimeDilation:
    """Gravitational time dilation: lower χ → lower oscillation frequency."""

    def _setup(self) -> tuple[Simulation, int, int]:
        """Create sim with frozen χ-well, return (sim, near_y, far_y)."""
        mid = _N // 2
        cfg = gravity_only(grid_size=_N)
        sim = Simulation(cfg, backend=BACKEND)

        # Carve a χ-well with a massive soliton.
        sim.place_soliton((mid, mid, mid), amplitude=8.0)
        sim.equilibrate()

        near_y = mid + 3  # inside χ-well
        far_y = mid + 14  # outside χ-well (far from mass)

        chi_near = float(sim.chi[mid, near_y, mid])
        chi_far = float(sim.chi[mid, far_y, mid])
        assert chi_near < chi_far, f"χ layout wrong: χ_near={chi_near:.2f} ≥ χ_far={chi_far:.2f}"

        # Wipe wave content, inject two isolated probes.
        shape = sim.psi_real.shape
        probe = _build_probe((mid, near_y, mid), shape) + _build_probe((mid, far_y, mid), shape)
        sim.set_psi_real(probe.copy())
        sim.set_psi_real_prev(probe.copy())  # start at rest
        if sim.psi_imag is not None:
            zeros = np.zeros(shape, dtype=np.float32)
            sim.set_psi_imag(zeros.copy())
            sim.set_psi_imag_prev(zeros.copy())

        return sim, near_y, far_y

    def test_frequency_lower_in_well(self) -> None:
        """Probe near mass oscillates at lower frequency than probe far away."""
        sim, near_y, far_y = self._setup()
        mid = _N // 2
        dt = sim.config.dt

        ts_near: list[float] = []
        ts_far: list[float] = []

        for _ in range(_STEPS):
            sim.run(steps=1, record_metrics=False, evolve_chi=False)
            ts_near.append(float(sim.psi_real[mid, near_y, mid]))
            ts_far.append(float(sim.psi_real[mid, far_y, mid]))

        sig_near = np.array(ts_near)
        sig_far = np.array(ts_far)

        if np.std(sig_near) < 1e-10 or np.std(sig_far) < 1e-10:
            pytest.skip("Probe signal too small to measure frequency.")

        omega_near = _measure_frequency(sig_near, dt)
        omega_far = _measure_frequency(sig_far, dt)

        assert omega_near < omega_far, f"ω_near={omega_near:.2f} should be < ω_far={omega_far:.2f}"

    def test_frequency_ratio_tracks_chi_ratio(self) -> None:
        """ω_near / ω_far ≈ χ_near / χ_far within 30%."""
        sim, near_y, far_y = self._setup()
        mid = _N // 2
        dt = sim.config.dt

        chi_near = float(sim.chi[mid, near_y, mid])
        chi_far = float(sim.chi[mid, far_y, mid])
        if chi_far < 1e-6:
            pytest.skip("χ_far too small")
        chi_ratio = chi_near / chi_far

        ts_near: list[float] = []
        ts_far: list[float] = []

        for _ in range(_STEPS):
            sim.run(steps=1, record_metrics=False, evolve_chi=False)
            ts_near.append(float(sim.psi_real[mid, near_y, mid]))
            ts_far.append(float(sim.psi_real[mid, far_y, mid]))

        sig_near = np.array(ts_near)
        sig_far = np.array(ts_far)

        if np.std(sig_near) < 1e-10 or np.std(sig_far) < 1e-10:
            pytest.skip("Probe signal too small.")

        omega_near = _measure_frequency(sig_near, dt)
        omega_far = _measure_frequency(sig_far, dt)

        if omega_far < 1e-6:
            pytest.skip("ω_far ~0")

        omega_ratio = omega_near / omega_far
        rel_err = abs(omega_ratio - chi_ratio) / max(abs(chi_ratio), 1e-10)
        assert rel_err < 0.30, (
            f"ω ratio {omega_ratio:.3f} vs χ ratio {chi_ratio:.3f}, rel_err={rel_err:.3f}"
        )
