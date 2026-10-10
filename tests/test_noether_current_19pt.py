"""Tests for the opt-in 19-point current-feedback observable."""

from __future__ import annotations

import numpy as np
import pytest

import lfm
from lfm.core.backends import gpu_available
from lfm.core.stencils import noether_current_19pt_raw


def _plane_wave(
    n: int,
    modes: tuple[int, int, int],
    amplitude: float,
    phase0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(n, dtype=np.float64)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    kx, ky, kz = (2.0 * np.pi * mode / n for mode in modes)
    phase = kx * x + ky * y + kz * z + phase0
    return amplitude * np.cos(phase), amplitude * np.sin(phase)


def _nyquist_strip(
    n: int,
    amplitude: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    psi = np.zeros((n, n, n), dtype=np.complex128)
    x0 = n // 2
    y0 = n // 2 - 1
    carrier = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    outer = amplitude * np.exp(-1j * delta) * carrier
    center = amplitude * carrier
    psi[x0, y0, :] = outer
    psi[x0, y0 + 1, :] = center
    psi[x0, y0 + 2, :] = outer
    return psi.real, psi.imag, (x0, y0, n // 2)


class TestNoetherCurrent19ptSymbol:
    def test_plane_wave_matches_stencil_symbol(self):
        n = 24
        modes = (2, 3, 5)
        amplitude = 1.7
        real, imag = _plane_wave(n, modes, amplitude, phase0=0.31)

        jx_raw, jy_raw, jz_raw = noether_current_19pt_raw(real, imag)

        kx, ky, kz = (2.0 * np.pi * mode / n for mode in modes)
        expected = (
            (2.0 * amplitude**2 / 3.0) * np.sin(kx) * (1.0 + np.cos(ky) + np.cos(kz)),
            (2.0 * amplitude**2 / 3.0) * np.sin(ky) * (1.0 + np.cos(kx) + np.cos(kz)),
            (2.0 * amplitude**2 / 3.0) * np.sin(kz) * (1.0 + np.cos(kx) + np.cos(ky)),
        )
        for actual, target in zip((jx_raw, jy_raw, jz_raw), expected, strict=True):
            np.testing.assert_allclose(actual, target, rtol=0.0, atol=2e-14)

    def test_global_phase_invariant_and_wave_reversal_changes_sign(self):
        n = 20
        modes = (2, 1, 3)
        real_a, imag_a = _plane_wave(n, modes, 0.8, phase0=0.0)
        real_b, imag_b = _plane_wave(n, modes, 0.8, phase0=1.23)
        real_r, imag_r = _plane_wave(n, tuple(-mode for mode in modes), 0.8)

        current_a = noether_current_19pt_raw(real_a, imag_a)
        current_b = noether_current_19pt_raw(real_b, imag_b)
        current_r = noether_current_19pt_raw(real_r, imag_r)
        for base, shifted, reversed_wave in zip(current_a, current_b, current_r, strict=True):
            np.testing.assert_allclose(shifted, base, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(reversed_wave, -base, rtol=0.0, atol=1e-14)

    def test_transverse_nyquist_strip_cancels_face_with_edges(self):
        real, imag, (x0, y0, z0) = _nyquist_strip(12, 1.4, 0.37)
        _, jy_raw, _ = noether_current_19pt_raw(real, imag)

        np.testing.assert_allclose(
            jy_raw[x0, y0 : y0 + 3, :],
            0.0,
            rtol=0.0,
            atol=1e-15,
        )

        d_real_y = np.roll(real, -1, axis=1) - np.roll(real, 1, axis=1)
        d_imag_y = np.roll(imag, -1, axis=1) - np.roll(imag, 1, axis=1)
        legacy_raw = real * d_imag_y - imag * d_real_y
        expected_top = 1.4**2 * np.sin(0.37)
        np.testing.assert_allclose(legacy_raw[x0, y0, z0], expected_top, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(legacy_raw[x0, y0 + 2, z0], -expected_top, rtol=0.0, atol=1e-14)


def test_cpu_complex_one_step_source_sign_and_normalization():
    n = 12
    amplitude = 0.4
    epsilon_w = 0.2
    modes = (1, 2, 3)
    real, imag = _plane_wave(n, modes, amplitude, phase0=0.17)
    cfg = lfm.SimulationConfig(
        grid_size=n,
        precision=lfm.Precision.FLOAT64,
        field_level=lfm.FieldLevel.COMPLEX,
        boundary_type=lfm.BoundaryType.PERIODIC,
        lambda_self=0.0,
        e0_sq=amplitude**2,
        epsilon_w=epsilon_w,
        use_stencil19_noether_current=True,
        report_interval=10**9,
    )
    sim = lfm.Simulation(cfg, backend="cpu")
    sim.set_psi_real(real)
    sim.set_psi_imag(imag)
    sim.run(1, record_metrics=False)

    raw_components = noether_current_19pt_raw(real, imag)
    physical_scalar = 0.5 * sum(raw_components)
    expected_chi = cfg.chi0 - cfg.dt**2 * cfg.kappa * epsilon_w * physical_scalar
    np.testing.assert_allclose(sim.get_chi(), expected_chi, rtol=0.0, atol=2e-14)


def _run_nyquist_color(
    epsilon_w: float,
    use_stencil19: bool,
    backend: str,
) -> np.ndarray:
    n = 12
    real, imag, _ = _nyquist_strip(n, 1.4, 0.37)
    real_color = np.zeros((3, n, n, n), dtype=np.float64)
    imag_color = np.zeros_like(real_color)
    real_color[0] = real
    imag_color[0] = imag
    cfg = lfm.SimulationConfig(
        grid_size=n,
        precision=lfm.Precision.FLOAT64,
        field_level=lfm.FieldLevel.COLOR,
        boundary_type=lfm.BoundaryType.PERIODIC,
        lambda_self=0.0,
        epsilon_w=epsilon_w,
        use_stencil19_noether_current=use_stencil19,
        report_interval=10**9,
    )
    sim = lfm.Simulation(cfg, backend=backend)
    sim.set_psi_real(real_color)
    sim.set_psi_imag(imag_color)
    sim.run(1, record_metrics=False)
    return sim.get_chi()


def test_cpu_color_default_retains_legacy_and_opt_in_removes_artifact():
    legacy_zero = _run_nyquist_color(0.0, False, "cpu")
    legacy_active = _run_nyquist_color(1.0, False, "cpu")
    stencil_zero = _run_nyquist_color(0.0, True, "cpu")
    stencil_active = _run_nyquist_color(1.0, True, "cpu")

    n = legacy_zero.shape[0]
    x0 = n // 2
    y0 = n // 2 - 1
    expected = -0.5 * (0.02**2) * lfm.KAPPA * 1.4**2 * np.sin(0.37)
    np.testing.assert_allclose(
        legacy_active[x0, y0, :] - legacy_zero[x0, y0, :],
        expected,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        legacy_active[x0, y0 + 2, :] - legacy_zero[x0, y0 + 2, :],
        -expected,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(stencil_active, stencil_zero)


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="CuPy GPU backend unavailable")
@pytest.mark.parametrize(
    "field_level",
    (lfm.FieldLevel.COMPLEX, lfm.FieldLevel.COLOR),
)
def test_float64_gpu_matches_cpu_with_stencil19_current(field_level):
    n = 8
    cfg = lfm.SimulationConfig(
        grid_size=n,
        precision=lfm.Precision.FLOAT64,
        field_level=field_level,
        boundary_type=lfm.BoundaryType.PERIODIC,
        lambda_self=0.0,
        epsilon_w=0.1,
        e0_sq=0.01,
        use_stencil19_noether_current=True,
        report_interval=10**9,
    )
    cpu = lfm.Simulation(cfg, backend="cpu")
    gpu = lfm.Simulation(cfg, backend="gpu")

    rng = np.random.default_rng(20260716)
    prefix = (3,) if field_level == lfm.FieldLevel.COLOR else ()
    shape = prefix + (n, n, n)
    real = rng.normal(0.0, 0.05, shape)
    imag = rng.normal(0.0, 0.05, shape)
    real_prev = real + rng.normal(0.0, 1e-4, shape)
    imag_prev = imag + rng.normal(0.0, 1e-4, shape)
    chi = 19.0 + rng.normal(0.0, 1e-3, (n, n, n))
    chi_prev = chi + rng.normal(0.0, 1e-5, (n, n, n))
    for sim in (cpu, gpu):
        sim.set_psi_real(real)
        sim.set_psi_imag(imag)
        sim.set_psi_real_prev(real_prev)
        sim.set_psi_imag_prev(imag_prev)
        sim.set_chi(chi)
        sim.set_chi_prev(chi_prev)
        sim.run(3, record_metrics=False)

    cpu_state = cpu.phase_space_snapshot()
    gpu_state = gpu.phase_space_snapshot()
    for key in (
        "psi_real",
        "psi_real_prev",
        "psi_imag",
        "psi_imag_prev",
        "chi",
        "chi_prev",
    ):
        np.testing.assert_allclose(
            gpu_state[key],
            cpu_state[key],
            rtol=5e-12,
            atol=5e-12,
        )
