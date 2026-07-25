"""Tests for the experiment-only R6 weak normalization."""

from __future__ import annotations

import pytest

from lfm.foundations.r3_link_frame_live import R3LiveParameters
from lfm.foundations.r6_unified_live import (
    R6Parameters,
    R6R4Parameters,
    r6_action_declaration,
    total_hamiltonian,
    vacuum_state,
)


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r6_has_no_new_register_and_unit_weak_speed(stencil: str) -> None:
    parameters = R6Parameters(
        r4=R6R4Parameters(
            r3=R3LiveParameters(stencil=stencil)
        )
    )
    declaration = r6_action_declaration(parameters)
    assert parameters.r4.weak_stiffness == pytest.approx(10.0)
    assert parameters.r4.weak_inertia == pytest.approx(10.0)
    assert declaration["derived_parameters"]["weak_speed_squared"] == 1.0
    assert declaration["new_registers"] == []
    assert declaration["new_potential_terms"] == []
    energy = total_hamiltonian(
        vacuum_state(2, parameters),
        parameters,
    )[0]
    assert energy == pytest.approx(0.0, abs=1.0e-12)
