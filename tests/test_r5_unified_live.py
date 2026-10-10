"""Tests for the experiment-only R5 curvature completion."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from lfm.foundations.r3_link_frame_live import (
    R3LiveParameters,
    su3_generators,
)
from lfm.foundations.r4_quantum_color import (
    local_su3_gauge_transform,
)
from lfm.foundations.r4_unified_live import R4Parameters
from lfm.foundations.r5_unified_live import (
    R5Parameters,
    potential_energy_and_rates,
    r5_action_declaration,
    total_hamiltonian,
    vacuum_state,
)


def _parameters(stencil: str) -> R5Parameters:
    return R5Parameters(r4=R4Parameters(r3=R3LiveParameters(stencil=stencil)))


@pytest.mark.parametrize(
    ("stencil", "coefficient"),
    [("19", 5.0), ("27", 9.0)],
)
def test_r5_vacuum_and_derived_square_coefficient(
    stencil: str,
    coefficient: float,
) -> None:
    parameters = _parameters(stencil)
    state = vacuum_state(2, parameters)
    energy, parts = total_hamiltonian(state, parameters)
    assert energy == pytest.approx(0.0, abs=1.0e-12)
    assert parameters.square_coefficient == coefficient
    assert r5_action_declaration(parameters)["new_registers"] == []
    assert parts["phase_face_square"] == pytest.approx(0.0)
    assert parts["color_face_square"] == pytest.approx(0.0)
    assert parts["frame_face_square"] == pytest.approx(0.0)
    assert parts["weak_face_square"] == pytest.approx(0.0)


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r5_face_square_force_is_variational(stencil: str) -> None:
    parameters = _parameters(stencil)
    state = vacuum_state(3, parameters)
    rng = np.random.default_rng(719)
    state.r3.phase_links *= np.exp(1.0j * 0.02 * rng.normal(size=state.r3.phase_links.shape))
    _, rates, _ = potential_energy_and_rates(state, parameters)
    site = (1, 1, 1, 0)
    epsilon = 1.0e-6

    def energy(delta: float) -> float:
        varied = state.copy()
        varied.r3.phase_links[site] *= np.exp(1.0j * delta)
        return total_hamiltonian(varied, parameters)[0]

    derivative = (energy(epsilon) - energy(-epsilon)) / (2.0 * epsilon)
    assert derivative == pytest.approx(
        -rates.r3.phase_electric[site],
        rel=2.0e-6,
        abs=2.0e-7,
    )


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r5_full_hamiltonian_retains_local_su3_invariance(
    stencil: str,
) -> None:
    parameters = _parameters(stencil)
    state = vacuum_state(2, parameters)
    rng = np.random.default_rng(727)
    state.r3.matter = 0.02 * (
        rng.normal(size=state.r3.matter.shape) + 1.0j * rng.normal(size=state.r3.matter.shape)
    )
    gauge = np.empty(state.r3.chi.shape + (3, 3), dtype=np.complex128)
    generators = su3_generators()
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
        parameters.r4,
    )
    assert total_hamiltonian(
        transformed,
        parameters,
    )[0] == pytest.approx(
        total_hamiltonian(state, parameters)[0],
        rel=1.0e-12,
        abs=1.0e-10,
    )
