"""Tests for local Noether phase-clock maps."""

import numpy as np
import pytest

from lfm.config import FieldLevel, SimulationConfig
from lfm.core.evolver import Evolver


def test_local_phase_clock_rotates_complex_state_and_prev_buffers():
    cfg = SimulationConfig(
        grid_size=8,
        field_level=FieldLevel.COMPLEX,
        e0_sq=0.0,
        lambda_self=0.0,
        kappa=0.0,
    )
    ev = Evolver(cfg, backend="cpu")
    real = np.ones((8, 8, 8), dtype=np.float64)
    imag = np.zeros((8, 8, 8), dtype=np.float64)
    dwell = np.zeros((8, 8, 8), dtype=np.float64)
    dwell[1, 2, 3] = 1.0
    dwell[2, 3, 4] = 2.0
    dwell[3, 4, 5] = 3.0

    ev.set_psi_real(real)
    ev.set_psi_imag(imag)
    ev.set_local_phase_clock_map(dwell, np.pi / 2.0)
    ev.apply_local_phase_clock_map()

    expected = real.astype(np.complex128) * np.exp(1j * dwell * np.pi / 2.0)
    np.testing.assert_allclose(ev.get_psi_real(), expected.real, atol=1e-12)
    np.testing.assert_allclose(ev.get_psi_imag(), expected.imag, atol=1e-12)
    np.testing.assert_allclose(ev.get_psi_real_prev(), expected.real, atol=1e-12)
    np.testing.assert_allclose(ev.get_psi_imag_prev(), expected.imag, atol=1e-12)


def test_local_phase_clock_can_address_color_components_separately():
    cfg = SimulationConfig(
        grid_size=8,
        field_level=FieldLevel.COLOR,
        n_colors=3,
        e0_sq=0.0,
        lambda_self=0.0,
        kappa=0.0,
    )
    ev = Evolver(cfg, backend="cpu")
    real = np.ones((3, 8, 8, 8), dtype=np.float64)
    imag = np.zeros((3, 8, 8, 8), dtype=np.float64)
    dwell = np.zeros((3, 8, 8, 8), dtype=np.float64)
    dwell[1, 1, 2, 3] = 1.0

    ev.set_psi_real(real)
    ev.set_psi_imag(imag)
    ev.set_local_phase_clock_map(dwell, np.pi)
    ev.apply_local_phase_clock_map()

    expected = real.astype(np.complex128)
    expected[1, 1, 2, 3] *= -1.0
    np.testing.assert_allclose(ev.get_psi_real(), expected.real, atol=1e-12)
    np.testing.assert_allclose(ev.get_psi_imag(), expected.imag, atol=1e-12)


def test_local_phase_clock_rejects_real_field_level():
    cfg = SimulationConfig(grid_size=8, field_level=FieldLevel.REAL)
    ev = Evolver(cfg, backend="cpu")

    with pytest.raises(ValueError, match="complex field level"):
        ev.set_local_phase_clock_map(np.zeros((8, 8, 8)), np.pi)


def test_local_phase_clock_map_must_be_declared_before_evolution():
    cfg = SimulationConfig(
        grid_size=8,
        field_level=FieldLevel.COMPLEX,
        e0_sq=0.0,
        lambda_self=0.0,
        kappa=0.0,
    )
    ev = Evolver(cfg, backend="cpu")
    ev.evolve(1)

    with pytest.raises(RuntimeError, match="before evolution"):
        ev.set_local_phase_clock_map(np.zeros((8, 8, 8)), np.pi)
