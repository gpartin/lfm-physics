"""Tests for operational collective-substrate observables."""

from __future__ import annotations

import numpy as np
import pytest

from lfm.analysis.collective_geometry import (
    SOURCE_CASES,
    analytic_leapfrog_limit,
    apply_momentum_sponge,
    block_average,
    collective_initial_state,
    continuum_fit,
    dispersion_shell_metrics,
    energy_current_vector_and_tensor,
    periodic_weighted_moments,
    traceless,
)
from lfm.analysis.energy_current import BareLFMParameters


@pytest.mark.parametrize("case", SOURCE_CASES)
def test_registered_initial_states_use_only_six_plus_one_registers(case: str) -> None:
    state = collective_initial_state(12, 0.96, case)
    assert state.wave.shape == (6, 12, 12, 12)
    assert state.chi.shape == (12, 12, 12)
    assert np.all(state.chi == 19.0)
    assert np.any(state.wave_momentum[4] != 0.0)
    assert np.all(state.wave[2:4] == 0.0)


def test_moving_pair_reverses_internal_phase_gradient() -> None:
    plus = collective_initial_state(16, 0.96, "moving_plus")
    minus = collective_initial_state(16, 0.96, "moving_minus")
    assert np.allclose(plus.wave[0], minus.wave[0])
    assert np.allclose(plus.wave[1], -minus.wave[1])
    assert np.allclose(plus.wave_momentum[0], -minus.wave_momentum[0])
    assert np.allclose(plus.wave_momentum[1], minus.wave_momentum[1])


def test_diagonal_motion_preserves_wave_number_magnitude() -> None:
    state = collective_initial_state(
        16,
        0.96,
        "moving_plus",
        motion_direction=(1.0, 1.0, 1.0),
    )
    assert np.all(np.isfinite(state.wave))
    assert not np.allclose(state.wave[1], 0.0)


def test_sponge_leaves_interior_and_damps_boundary_momenta() -> None:
    state = collective_initial_state(16, 0.96, "static_sphere")
    state = type(state)(
        wave=state.wave,
        wave_momentum=np.ones_like(state.wave_momentum),
        chi=state.chi,
        chi_momentum=np.ones_like(state.chi_momentum),
    )
    damped = apply_momentum_sponge(state, 0.96, 0.01)
    assert damped.wave_momentum[0, 8, 8, 8] == pytest.approx(1.0)
    assert damped.wave_momentum[0, 0, 0, 0] < 1.0
    assert damped.chi_momentum[0, 0, 0] < 1.0


def test_periodic_weighted_moments_cross_boundary() -> None:
    weights = np.zeros((8, 8, 8))
    weights[0, 4, 4] = 1.0
    weights[-1, 4, 4] = 1.0
    moments = periodic_weighted_moments(weights, 0.8)
    assert moments.covariance[0, 0] < 0.02
    assert moments.covariance[1, 1] < 1.0e-14


def test_block_average_preserves_prefix_axes_and_mean() -> None:
    values = np.arange(2 * 8**3, dtype=float).reshape(2, 8, 8, 8)
    blocked = block_average(values, 2)
    assert blocked.shape == (2, 4, 4, 4)
    assert float(np.mean(blocked)) == pytest.approx(float(np.mean(values)))


def test_current_vector_and_tensor_have_operational_shapes() -> None:
    shape = (4, 4, 4)
    currents = {
        (1, 0, 0): np.ones(shape),
        (-1, 0, 0): -np.ones(shape),
    }
    vector, tensor = energy_current_vector_and_tensor(currents, 0.1)
    assert vector.shape == (3,) + shape
    assert tensor.shape == (3, 3) + shape
    assert np.allclose(vector[0], 0.1)
    assert np.allclose(tensor[0, 0], 1.0)
    assert np.allclose(traceless(np.eye(3)), np.zeros((3, 3)))


def test_continuum_fit_recovers_nonzero_intercept() -> None:
    spacing = np.asarray((0.04, 0.03, 0.02))
    values = 2.5 - 7.0 * spacing**2
    fit = continuum_fit(spacing, values)
    assert float(fit.intercept) == pytest.approx(2.5, abs=1.0e-12)
    assert float(fit.slope) == pytest.approx(-7.0, abs=1.0e-10)
    assert bool(fit.sign_consistent)


@pytest.mark.parametrize("stencil", ("19", "27"))
def test_dispersion_and_cfl_metrics_are_finite(stencil: str) -> None:
    dispersion = dispersion_shell_metrics(stencil, 0.03, 4.0 * np.pi / 0.96)
    assert dispersion["directional_anisotropy"] >= 0.0
    assert np.all(np.isfinite(list(dispersion.values())))
    limit = analytic_leapfrog_limit(
        BareLFMParameters(
            spacing=0.03,
            gov01_stencil=stencil,
            gov02_stencil=stencil,
        ),
        symbol_samples=17,
    )
    assert limit["maximum_dt"] > 0.0
    assert limit["maximum_courant"] > 0.0
