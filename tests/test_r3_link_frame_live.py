from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from lfm.foundations.r3_link_frame_live import (
    R3LiveParameters,
    R3LiveState,
    _tracefree_symmetric,
    group_constraint_errors,
    potential_energy_and_rates,
    reverse_momenta,
    so4_generators,
    state_distance,
    step_r3_live,
    su3_generators,
    total_hamiltonian,
    triangle_loops,
)


def _seeded_state(seed: int = 7) -> tuple[R3LiveState, R3LiveParameters]:
    rng = np.random.default_rng(seed)
    parameters = R3LiveParameters(stencil="19")
    state = R3LiveState.vacuum(2, parameters)
    state.matter = 0.01 * (
        rng.normal(size=state.matter.shape)
        + 1.0j * rng.normal(size=state.matter.shape)
    )
    state.matter_momentum = 0.01 * (
        rng.normal(size=state.matter.shape)
        + 1.0j * rng.normal(size=state.matter.shape)
    )
    state.chi += 1.0e-4 * rng.normal(size=state.chi.shape)
    state.chi_momentum = 0.01 * rng.normal(size=state.chi.shape)
    state.shape = _tracefree_symmetric(
        1.0e-4 * rng.normal(size=state.shape.shape)
    )
    state.shape_momentum = _tracefree_symmetric(
        0.01 * rng.normal(size=state.shape_momentum.shape)
    )
    state.phase_electric = 0.01 * rng.normal(
        size=state.phase_electric.shape
    )
    state.color_electric = 0.01 * rng.normal(
        size=state.color_electric.shape
    )
    state.frame_electric = 0.01 * rng.normal(
        size=state.frame_electric.shape
    )
    state.phase_links *= np.exp(
        1.0j * 1.0e-3 * rng.normal(size=state.phase_links.shape)
    )
    color_generators = su3_generators()
    frame_generators = so4_generators()
    for site in np.ndindex(state.chi.shape):
        for link_index in range(state.phase_links.shape[3]):
            color_coordinates = 1.0e-3 * rng.normal(size=8)
            state.color_links[site + (link_index,)] = expm(
                1.0j
                * np.einsum(
                    "a,aij->ij",
                    color_coordinates,
                    color_generators,
                )
            )
            frame_coordinates = 1.0e-3 * rng.normal(size=6)
            state.frame_links[site + (link_index,)] = expm(
                np.einsum(
                    "a,aij->ij",
                    frame_coordinates,
                    frame_generators,
                )
            )
    return state, parameters


def _potential(state: R3LiveState, parameters: R3LiveParameters) -> float:
    return potential_energy_and_rates(state, parameters)[0]


def test_live_loop_inventory_and_vacuum() -> None:
    assert len(triangle_loops("19")) == 10
    assert len(triangle_loops("27")) == 22
    parameters = R3LiveParameters()
    state = R3LiveState.vacuum(2, parameters)
    energy, parts = total_hamiltonian(state, parameters)
    assert energy == 0.0
    assert all(value == 0.0 for value in parts.values())
    step_r3_live(state, 1.0e-3, parameters)
    assert total_hamiltonian(state, parameters)[0] == 0.0


def test_live_rates_are_hamiltonian_gradients() -> None:
    state, parameters = _seeded_state()
    _, rates, _ = potential_energy_and_rates(state, parameters)
    epsilon = 1.0e-7
    site = (0, 0, 0)
    link = 0

    def directional(
        plus_update,
        minus_update,
    ) -> float:
        plus = state.copy()
        minus = state.copy()
        plus_update(plus)
        minus_update(minus)
        return (
            _potential(plus, parameters) - _potential(minus, parameters)
        ) / (2.0 * epsilon)

    matter_derivative = directional(
        lambda value: value.matter.__setitem__(
            site + (0,),
            value.matter[site + (0,)] + epsilon,
        ),
        lambda value: value.matter.__setitem__(
            site + (0,),
            value.matter[site + (0,)] - epsilon,
        ),
    )
    assert np.isclose(
        matter_derivative,
        -rates.matter[site + (0,)].real,
        rtol=2.0e-5,
        atol=2.0e-7,
    )

    chi_derivative = directional(
        lambda value: value.chi.__setitem__(
            site,
            value.chi[site] + epsilon,
        ),
        lambda value: value.chi.__setitem__(
            site,
            value.chi[site] - epsilon,
        ),
    )
    assert np.isclose(
        chi_derivative,
        -rates.chi[site],
        rtol=2.0e-5,
        atol=2.0e-7,
    )

    phase_derivative = directional(
        lambda value: value.phase_links.__setitem__(
            site + (link,),
            np.exp(1.0j * epsilon) * value.phase_links[site + (link,)],
        ),
        lambda value: value.phase_links.__setitem__(
            site + (link,),
            np.exp(-1.0j * epsilon) * value.phase_links[site + (link,)],
        ),
    )
    assert np.isclose(
        phase_derivative,
        -rates.phase_electric[site + (link,)],
        rtol=2.0e-5,
        atol=2.0e-7,
    )

    color_generator = su3_generators()[2]
    color_derivative = directional(
        lambda value: value.color_links.__setitem__(
            site + (link,),
            expm(1.0j * epsilon * color_generator)
            @ value.color_links[site + (link,)],
        ),
        lambda value: value.color_links.__setitem__(
            site + (link,),
            expm(-1.0j * epsilon * color_generator)
            @ value.color_links[site + (link,)],
        ),
    )
    assert np.isclose(
        color_derivative,
        -rates.color_electric[site + (link, 2)],
        rtol=2.0e-5,
        atol=2.0e-7,
    )

    frame_generator = so4_generators()[1]
    frame_derivative = directional(
        lambda value: value.frame_links.__setitem__(
            site + (link,),
            expm(epsilon * frame_generator)
            @ value.frame_links[site + (link,)],
        ),
        lambda value: value.frame_links.__setitem__(
            site + (link,),
            expm(-epsilon * frame_generator)
            @ value.frame_links[site + (link,)],
        ),
    )
    assert np.isclose(
        frame_derivative,
        -rates.frame_electric[site + (link, 1)],
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_live_step_preserves_groups_and_reverses() -> None:
    state, parameters = _seeded_state(11)
    initial = state.copy()
    for _ in range(4):
        step_r3_live(state, 2.0e-4, parameters)
    constraints = group_constraint_errors(state)
    assert max(constraints.values()) < 2.0e-13
    reverse_momenta(state)
    for _ in range(4):
        step_r3_live(state, 2.0e-4, parameters)
    reverse_momenta(state)
    assert state_distance(initial, state) < 2.0e-11


def test_live_energy_error_is_second_order_bounded() -> None:
    initial, parameters = _seeded_state(19)

    def run(dt: float, steps: int) -> tuple[R3LiveState, float]:
        state = initial.copy()
        energies = [total_hamiltonian(state, parameters)[0]]
        for _ in range(steps):
            step_r3_live(state, dt, parameters)
            energies.append(total_hamiltonian(state, parameters)[0])
        span = (max(energies) - min(energies)) / max(
            abs(energies[0]),
            1.0,
        )
        return state, span

    coarse, coarse_span = run(4.0e-4, 4)
    medium, medium_span = run(2.0e-4, 8)
    fine, fine_span = run(1.0e-4, 16)
    assert coarse_span < 2.0e-7
    assert medium_span < coarse_span
    assert fine_span < medium_span
    coarse_medium = state_distance(coarse, medium)
    medium_fine = state_distance(medium, fine)
    assert coarse_medium / medium_fine > 3.0
