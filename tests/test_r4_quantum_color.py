"""Tests for the compact-link R4 quantum color diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from lfm.foundations.r3_link_frame_live import (
    R3LiveParameters,
    su3_generators,
)
from lfm.foundations.r4_quantum_color import (
    creutz_ratio_from_log_transfer,
    fundamental_flux_energy,
    local_su3_gauge_transform,
    minimum_link_distance,
    r4_magnetic_competition_bound,
    r4_quantum_color_coefficients,
    su3_fundamental_algebra_audit,
)
from lfm.foundations.r4_unified_live import R4Parameters, R4State, total_hamiltonian


def _parameters(stencil: str) -> R4Parameters:
    return R4Parameters(r3=R3LiveParameters(stencil=stencil))


def test_su3_fundamental_algebra_is_derived() -> None:
    audit = su3_fundamental_algebra_audit()
    assert audit["generator_count"] == 8
    assert audit["normalization_error"] < 1.0e-14
    assert audit["casimir_spread"] < 1.0e-14
    assert float(audit["casimir"]) == pytest.approx(4.0 / 3.0)


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_vacuum_coefficients_give_positive_flux_slope(stencil: str) -> None:
    coefficients = r4_quantum_color_coefficients(_parameters(stencil))
    assert coefficients.epsilon_vacuum == pytest.approx(1.0 / 63.0)
    assert coefficients.g_squared_from_electric == pytest.approx(63.0)
    assert coefficients.inverse_g_squared_from_magnetic == pytest.approx(1.0 / 63.0)
    assert coefficients.fundamental_flux_slope == pytest.approx(42.0)
    bound = r4_magnetic_competition_bound(_parameters(stencil))
    assert bound.magnetic_bound_per_link > 0.0
    assert bound.residual_positive_slope > 0.99 * 42.0


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_gauss_flux_energy_and_creutz_area_law(stencil: str) -> None:
    parameters = _parameters(stencil)
    for distance in range(1, 9):
        assert minimum_link_distance((distance, 0, 0), stencil) == distance
        assert fundamental_flux_energy(
            (distance, 0, 0),
            parameters,
        ) == pytest.approx(42.0 * distance)
    values = [
        creutz_ratio_from_log_transfer(
            distance,
            euclidean_time=0.05,
            time_increment=0.025,
            parameters=parameters,
        )
        for distance in range(1, 5)
    ]
    assert np.asarray(values) == pytest.approx(np.full(4, 42.0))


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_full_r4_hamiltonian_is_locally_su3_gauge_invariant(
    stencil: str,
) -> None:
    parameters = _parameters(stencil)
    state = R4State.vacuum(2, parameters)
    rng = np.random.default_rng(541)
    state.r3.matter = 0.02 * (
        rng.normal(size=state.r3.matter.shape) + 1.0j * rng.normal(size=state.r3.matter.shape)
    )
    state.r3.matter_momentum = 0.03 * (
        rng.normal(size=state.r3.matter.shape) + 1.0j * rng.normal(size=state.r3.matter.shape)
    )
    state.r3.color_electric = 0.01 * rng.normal(size=state.r3.color_electric.shape)
    generators = su3_generators()
    gauge = np.empty(state.r3.chi.shape + (3, 3), dtype=np.complex128)
    for site in np.ndindex(state.r3.chi.shape):
        algebra = np.einsum(
            "a,aij->ij",
            0.2 * rng.normal(size=8),
            generators,
        )
        gauge[site] = expm(1.0j * algebra)
    transformed = local_su3_gauge_transform(
        state,
        gauge,
        parameters,
    )
    before = total_hamiltonian(state, parameters)[0]
    after = total_hamiltonian(transformed, parameters)[0]
    assert after == pytest.approx(before, rel=1.0e-12, abs=1.0e-10)
