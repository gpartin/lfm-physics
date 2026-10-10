from __future__ import annotations

import numpy as np
import pytest

from lfm.foundations.parsimonious_four_force import (
    P4FParameters,
    flat_octic_minimality_ledger,
    inactive_frame_error,
    p4f_action_declaration,
    p4f_action_fingerprint,
    potential_energy_and_rates,
    step_p4f,
    total_hamiltonian,
    vacuum_state,
)
from lfm.foundations.r3_link_frame_live import (
    R3LiveParameters,
    R3LiveState,
)
from lfm.foundations.r3_link_frame_live import (
    potential_energy_and_rates as r3_potential_energy_and_rates,
)
from lfm.foundations.r6_unified_live import R6Parameters, R6R4Parameters


def test_p4f_freezes_required_action_choices() -> None:
    parameters = P4FParameters()
    r3 = parameters.r6.r4.r3
    assert r3.stencil == "19"
    assert r3.chi_potential == "flat_octic"
    assert not r3.frame_enabled


def test_p4f_rejects_frame_or_quartic_variants() -> None:
    with pytest.raises(ValueError):
        P4FParameters(
            r6=R6Parameters(
                r4=R6R4Parameters(
                    r3=R3LiveParameters(
                        frame_enabled=True,
                        chi_potential="flat_octic",
                    )
                )
            )
        )
    with pytest.raises(ValueError):
        P4FParameters(
            r6=R6Parameters(
                r4=R6R4Parameters(
                    r3=R3LiveParameters(
                        frame_enabled=False,
                        chi_potential="quartic",
                    )
                )
            )
        )


def test_flat_octic_force_matches_analytic_derivative() -> None:
    parameters = R3LiveParameters(
        frame_enabled=False,
        chi_potential="flat_octic",
    )
    state = R3LiveState.vacuum(3, parameters)
    state.chi[...] = parameters.chi0 + 0.125
    _, rates, parts = r3_potential_energy_and_rates(state, parameters)
    delta = state.chi**2 - parameters.chi0**2
    expected = (
        -8.0
        * parameters.frame_inertia
        * parameters.lambda_h
        * state.chi
        * delta**3
        / parameters.chi0**4
    )
    np.testing.assert_allclose(rates.chi, expected, rtol=1e-13, atol=1e-13)
    assert parts["shape_gradient"] == 0.0
    assert parts["frame_loop"] == 0.0


def test_inactive_frame_has_no_energy_or_rate() -> None:
    state = vacuum_state(3)
    base = state.r3
    base.shape_momentum[...] = 0.25
    base.frame_electric[...] = 0.5
    energy, rates, parts = potential_energy_and_rates(state)
    kinetic, kinetic_parts = total_hamiltonian(state)
    assert np.isfinite(energy)
    assert np.isfinite(kinetic)
    assert np.max(np.abs(rates.r3.shape)) == 0.0
    assert np.max(np.abs(rates.r3.frame_electric)) == 0.0
    assert parts["shape_gradient"] == 0.0
    assert parts["frame_loop"] == 0.0
    assert kinetic_parts["shape_kinetic"] == 0.0
    assert kinetic_parts["frame_electric"] == 0.0
    assert kinetic_parts["frame_electric_weighted"] == 0.0


def test_p4f_step_preserves_dormant_frame_exactly() -> None:
    state = vacuum_state(3)
    state.r3.matter[1, 1, 1, 0] = 1.0e-3
    before, _ = total_hamiltonian(state)
    assert inactive_frame_error(state) == 0.0
    for _ in range(4):
        step_p4f(state, 1.0e-4)
    after, _ = total_hamiltonian(state)
    assert inactive_frame_error(state) == 0.0
    assert np.isfinite(after)
    assert abs(after - before) / max(abs(before), 1.0e-30) < 1.0e-4


def test_action_declaration_and_fingerprint_are_stable() -> None:
    declaration = p4f_action_declaration()
    assert declaration["canonical_status"] == "EXPERIMENT_ONLY_UNPROMOTED"
    assert declaration["forbidden_mechanisms_used"] == []
    assert declaration["paper_45_update_authorized"] is False
    fingerprint = p4f_action_fingerprint()
    assert len(fingerprint) == 64
    assert fingerprint == p4f_action_fingerprint()


def test_flat_octic_is_minimal_only_in_declared_monomial_class() -> None:
    ledger = flat_octic_minimality_ledger()
    assert ledger["minimal_admissible_z_power"] == 4
    assert ledger["minimal_admissible_field_degree"] == 8
    by_power = {row["z_power"]: row for row in ledger["rows"]}
    assert not by_power[2]["vacuum_hessian_vanishes"]
    assert not by_power[3]["nonnegative_for_both_signs_of_z"]
    assert by_power[4]["admissible"]
    assert "monomial" in ledger["uniqueness_boundary"]
