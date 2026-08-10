from __future__ import annotations

import numpy as np

from lfm.analysis.clock_link_live import (
    LiveClockParameters,
    LiveClockState,
    clock_momentum_rate,
    make_traveling_packet,
    potential_momentum_rates,
    step_live_clock,
    total_hamiltonian,
    weighted_gradient_force,
)
from lfm.core.stencils import laplacian_19pt, laplacian_27pt


def test_weighted_force_recovers_laplacian() -> None:
    rng = np.random.default_rng(20260724)
    field = rng.normal(size=(7, 7, 7))
    q = np.ones_like(field)
    for stencil, laplacian in (
        ("19", laplacian_19pt),
        ("27", laplacian_27pt),
    ):
        force = weighted_gradient_force(
            field,
            q,
            coefficient=1.0,
            stencil=stencil,
        )
        np.testing.assert_allclose(force, laplacian(field), atol=2.0e-15)


def test_live_forces_are_hamiltonian_derivatives() -> None:
    rng = np.random.default_rng(13)
    shape = (5, 5, 5)
    parameters = LiveClockParameters()
    state = LiveClockState(
        field=0.02 * rng.normal(size=shape),
        field_momentum=0.03 * rng.normal(size=shape),
        chi=parameters.chi0 + 0.01 * rng.normal(size=shape),
        chi_momentum=0.02 * rng.normal(size=shape),
        varphi=0.005 * rng.normal(size=shape),
        clock_momentum=0.02 * rng.normal(size=shape),
    )
    field_rate, chi_rate, _ = potential_momentum_rates(state, parameters)
    full_clock_rate = clock_momentum_rate(state, parameters)
    index = (2, 1, 3)
    epsilon = 1.0e-6

    for name, expected in (
        ("field", field_rate[index]),
        ("chi", chi_rate[index]),
        ("varphi", full_clock_rate[index]),
    ):
        plus = state.copy()
        minus = state.copy()
        getattr(plus, name)[index] += epsilon
        getattr(minus, name)[index] -= epsilon
        derivative = (
            total_hamiltonian(plus, parameters) - total_hamiltonian(minus, parameters)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(-derivative, expected, rtol=2.0e-6, atol=2.0e-6)


def test_source_free_vacuum_is_fixed_point() -> None:
    parameters = LiveClockParameters()
    shape = (8, 8, 8)
    zeros = np.zeros(shape, dtype=np.float64)
    state = LiveClockState(
        field=zeros.copy(),
        field_momentum=zeros.copy(),
        chi=np.full(shape, parameters.chi0),
        chi_momentum=zeros.copy(),
        varphi=zeros.copy(),
        clock_momentum=zeros.copy(),
    )
    initial = state.copy()
    for _ in range(20):
        step_live_clock(state, 0.005, parameters)
    for name in (
        "field",
        "field_momentum",
        "chi",
        "chi_momentum",
        "varphi",
        "clock_momentum",
    ):
        np.testing.assert_array_equal(getattr(state, name), getattr(initial, name))


def test_symmetric_split_is_reversible() -> None:
    parameters = LiveClockParameters()
    state = make_traveling_packet(
        8,
        amplitude=0.01,
        width=1.5,
        carrier_index=1,
        parameters=parameters,
    )
    initial = state.copy()
    for _ in range(10):
        step_live_clock(state, 0.0025, parameters)
    state.field_momentum *= -1.0
    state.chi_momentum *= -1.0
    state.clock_momentum *= -1.0
    for _ in range(10):
        step_live_clock(state, 0.0025, parameters)
    state.field_momentum *= -1.0
    state.chi_momentum *= -1.0
    state.clock_momentum *= -1.0
    for name in (
        "field",
        "field_momentum",
        "chi",
        "chi_momentum",
        "varphi",
        "clock_momentum",
    ):
        np.testing.assert_allclose(
            getattr(state, name),
            getattr(initial, name),
            atol=2.0e-13,
            rtol=2.0e-13,
        )
