"""Tests for the experimental same-action charge-coupling module."""

from __future__ import annotations

import numpy as np

from lfm.analysis.energy_current import (
    BareLFMParameters,
    BareLFMState,
    bare_hamilton_rates,
)
from lfm.experiment.charge_coupling import (
    ChargeCouplingParameters,
    canonical_charge_density,
    charge_coupled_hamiltonian,
    charge_coupled_rates,
    step_charge_coupled_lfm,
)


def sample_state(seed: int = 7) -> BareLFMState:
    rng = np.random.default_rng(seed)
    shape = (8, 8, 8)
    wave = 0.01 * rng.normal(size=(6,) + shape)
    wave_p = 0.01 * rng.normal(size=(6,) + shape)
    chi = 19.0 + 0.01 * rng.normal(size=shape)
    chi_p = 0.01 * rng.normal(size=shape)
    return BareLFMState(wave, wave_p, chi, chi_p)


def parameters(coupling: float) -> ChargeCouplingParameters:
    return ChargeCouplingParameters(
        bare=BareLFMParameters(
            chi_potential="flat_octic",
            spacing=0.8,
        ),
        coupling=coupling,
        midpoint_tolerance=1.0e-12,
        midpoint_max_iterations=30,
    )


def test_zero_coupling_rates_equal_bare_rates() -> None:
    state = sample_state()
    candidate = charge_coupled_rates(state, parameters(0.0))
    bare = bare_hamilton_rates(
        state.wave,
        state.wave_momentum,
        state.chi,
        state.chi_momentum,
        parameters(0.0).bare,
    )
    assert np.array_equal(candidate.wave, bare.wave)
    assert np.array_equal(candidate.wave_momentum, bare.wave_momentum)
    assert np.array_equal(candidate.chi, bare.chi)
    assert np.array_equal(candidate.chi_momentum, bare.chi_momentum)


def test_global_charge_rate_is_zero() -> None:
    state = sample_state()
    rates = charge_coupled_rates(state, parameters(0.1))
    density_rate = np.zeros(state.chi.shape, dtype=np.float64)
    for component in range(0, state.wave.shape[0], 2):
        u = state.wave[component]
        v = state.wave[component + 1]
        p_u = state.wave_momentum[component]
        p_v = state.wave_momentum[component + 1]
        u_rate = rates.wave[component]
        v_rate = rates.wave[component + 1]
        p_u_rate = rates.wave_momentum[component]
        p_v_rate = rates.wave_momentum[component + 1]
        density_rate += (
            u_rate * p_v
            + u * p_v_rate
            - v_rate * p_u
            - v * p_u_rate
        )
    assert abs(float(np.sum(density_rate))) < 1.0e-12


def test_signed_coupling_reverses_interaction_rates() -> None:
    state = sample_state()
    bare = charge_coupled_rates(state, parameters(0.0))
    positive = charge_coupled_rates(state, parameters(0.1))
    negative = charge_coupled_rates(state, parameters(-0.1))
    for zero_rate, positive_rate, negative_rate in zip(
        (
            bare.wave,
            bare.wave_momentum,
            bare.chi,
            bare.chi_momentum,
        ),
        (
            positive.wave,
            positive.wave_momentum,
            positive.chi,
            positive.chi_momentum,
        ),
        (
            negative.wave,
            negative.wave_momentum,
            negative.chi,
            negative.chi_momentum,
        ),
    ):
        assert np.max(
            np.abs((positive_rate - zero_rate) + (negative_rate - zero_rate))
        ) < 1.0e-14


def test_implicit_midpoint_is_time_reversible() -> None:
    state = sample_state()
    selected = parameters(0.1)
    forward = step_charge_coupled_lfm(state, 2.0e-4, selected)
    backward = step_charge_coupled_lfm(forward.state, -2.0e-4, selected)
    for recovered, expected in zip(
        (
            backward.state.wave,
            backward.state.wave_momentum,
            backward.state.chi,
            backward.state.chi_momentum,
        ),
        (
            state.wave,
            state.wave_momentum,
            state.chi,
            state.chi_momentum,
        ),
    ):
        assert np.max(np.abs(recovered - expected)) < 1.0e-11


def test_hamiltonian_and_charge_remain_finite() -> None:
    state = sample_state()
    selected = parameters(0.1)
    initial_energy = charge_coupled_hamiltonian(state, selected)
    initial_charge = float(np.sum(canonical_charge_density(state)))
    for _index in range(10):
        state = step_charge_coupled_lfm(state, 2.0e-4, selected).state
    final_energy = charge_coupled_hamiltonian(state, selected)
    final_charge = float(np.sum(canonical_charge_density(state)))
    assert np.isfinite(final_energy)
    assert abs(final_energy - initial_energy) / abs(initial_energy) < 1.0e-9
    assert abs(final_charge - initial_charge) < 1.0e-12
