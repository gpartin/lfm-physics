"""Tests for external collective particle-kinematics readouts."""

from __future__ import annotations

import numpy as np

from lfm.analysis.particle_kinematics import (
    component_noether_charges,
    fit_offset_power_convergence,
    flat_octic_hamiltonian_19pt,
    time_centered_momentum_19pt,
)


def _packet(size: int = 12) -> tuple[np.ndarray, ...]:
    dx = 0.1
    dt = 0.002
    axis = (np.arange(size, dtype=np.float32) - 0.5 * (size - 1)) * dx
    x = axis[:, None, None]
    profile = np.exp(-(x * x) / np.float32(0.12)).astype(np.float32)
    profile = np.broadcast_to(profile, (size, size, size)).copy()
    components = np.zeros((3, size, size, size), dtype=np.float32)
    components[0] = profile
    real = components.copy()
    previous_real = real - np.float32(dt * 0.2) * np.roll(real, 1, axis=1) / np.float32(dx)
    imag = np.zeros_like(real)
    previous_imag = np.zeros_like(real)
    chi = np.full((size, size, size), 19.0, dtype=np.float32)
    return real, previous_real, imag, previous_imag, chi, chi.copy()


def test_time_centered_momentum_direction() -> None:
    state = _packet()
    momentum = time_centered_momentum_19pt(*state, dt=0.002, dx=0.1)
    assert np.all(np.isfinite(momentum))
    assert abs(momentum[0]) > 0.0
    transverse = float(np.linalg.norm(momentum[1:]))
    assert transverse / abs(float(momentum[0])) < 1.0e-5


def test_component_noether_charge_and_hamiltonian_are_finite() -> None:
    state = list(_packet())
    phase = np.float32(0.01)
    state[2][0] = phase * state[0][0]
    state[3][0] = state[2][0] - np.float32(0.002) * state[0][0]
    charges = component_noether_charges(*state[:4], dt=0.002, dx=0.1)
    energy = flat_octic_hamiltonian_19pt(*state, dt=0.002, dx=0.1)
    assert charges.shape == (3,)
    assert np.all(np.isfinite(charges))
    assert np.isfinite(energy["total"])
    assert energy["total"] > 0.0


def test_offset_power_convergence_recovers_synthetic_parameters() -> None:
    h = np.asarray((0.125, 0.1, 1.0 / 12.0, 0.0625))
    expected_offset = 7.0e-4
    expected_order = 1.5
    errors = expected_offset + 0.4 * h**expected_order
    fit = fit_offset_power_convergence(h, errors)
    assert fit["converged"]
    assert abs(fit["offset"] - expected_offset) < 1.0e-8
    assert abs(fit["order"] - expected_order) < 1.0e-6
    assert fit["r_squared"] > 0.999999
