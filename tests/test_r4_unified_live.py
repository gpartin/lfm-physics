from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from lfm.foundations.r3_link_frame_live import (
    R3LiveParameters,
    _neighbor,
    _temporal_shape_projector,
    _tracefree_symmetric,
    so4_generators,
    su3_generators,
)
from lfm.foundations.r4_unified_live import (
    R4Parameters,
    R4FrameScalarState,
    R4State,
    color_dielectric,
    group_constraint_errors,
    potential_energy_and_rates,
    reverse_momenta,
    r4_frame_scalar_energy,
    state_distance,
    step_r4,
    step_r4_frame_scalar,
    su2_generators,
    total_hamiltonian,
)


def _seeded_state(
    seed: int = 41,
    stencil: str = "19",
) -> tuple[R4State, R4Parameters]:
    rng = np.random.default_rng(seed)
    parameters = R4Parameters(r3=R3LiveParameters(stencil=stencil))
    state = R4State.vacuum(2, parameters)
    base = state.r3
    base.matter = 0.005 * (
        rng.normal(size=base.matter.shape)
        + 1.0j * rng.normal(size=base.matter.shape)
    )
    base.matter_momentum = 0.005 * (
        rng.normal(size=base.matter.shape)
        + 1.0j * rng.normal(size=base.matter.shape)
    )
    base.chi += 1.0e-4 * rng.normal(size=base.chi.shape)
    base.chi_momentum = 0.005 * rng.normal(size=base.chi.shape)
    base.shape = _tracefree_symmetric(
        1.0e-4 * rng.normal(size=base.shape.shape)
    )
    base.shape_momentum = _tracefree_symmetric(
        0.005 * rng.normal(size=base.shape.shape)
    )
    base.phase_links *= np.exp(
        1.0j * 1.0e-3 * rng.normal(size=base.phase_links.shape)
    )
    base.phase_electric = 0.005 * rng.normal(
        size=base.phase_electric.shape
    )
    base.color_electric = 0.005 * rng.normal(
        size=base.color_electric.shape
    )
    base.frame_electric = 0.005 * rng.normal(
        size=base.frame_electric.shape
    )
    state.weak_matter = 0.005 * (
        rng.normal(size=state.weak_matter.shape)
        + 1.0j * rng.normal(size=state.weak_matter.shape)
    )
    state.weak_momentum = 0.005 * (
        rng.normal(size=state.weak_momentum.shape)
        + 1.0j * rng.normal(size=state.weak_momentum.shape)
    )
    state.weak_electric = 0.005 * rng.normal(
        size=state.weak_electric.shape
    )
    state.higgs_electric = 0.005 * rng.normal(
        size=state.higgs_electric.shape
    )
    color_generators = su3_generators()
    frame_generators = so4_generators()
    weak_generators = su2_generators()
    for site in np.ndindex(base.chi.shape):
        state.higgs_orientation[site] = expm(
            1.0j
            * np.einsum(
                "a,aij->ij",
                1.0e-3 * rng.normal(size=3),
                weak_generators,
            )
        )
        for link in range(base.phase_links.shape[3]):
            base.color_links[site + (link,)] = expm(
                1.0j
                * np.einsum(
                    "a,aij->ij",
                    1.0e-3 * rng.normal(size=8),
                    color_generators,
                )
            )
            base.frame_links[site + (link,)] = expm(
                np.einsum(
                    "a,aij->ij",
                    1.0e-3 * rng.normal(size=6),
                    frame_generators,
                )
            )
            state.weak_links[site + (link,)] = expm(
                1.0j
                * np.einsum(
                    "a,aij->ij",
                    1.0e-3 * rng.normal(size=3),
                    weak_generators,
                )
            )
    return state, parameters


def _potential(state: R4State, parameters: R4Parameters) -> float:
    return potential_energy_and_rates(state, parameters)[0]


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r4_vacuum_dielectric_and_generated_weak_gap(
    stencil: str,
) -> None:
    parameters = R4Parameters(r3=R3LiveParameters(stencil=stencil))
    state = R4State.vacuum(2, parameters)
    epsilon, derivative = color_dielectric(state.r3.chi, parameters)
    assert np.max(np.abs(epsilon - parameters.r3.kappa)) < 1.0e-15
    assert np.max(np.abs(derivative)) == 0.0
    assert total_hamiltonian(state, parameters)[0] == 0.0

    generator = su2_generators()[0]
    amplitude = 1.0e-4
    for index, (offset, _) in enumerate(
        __import__(
            "lfm.analysis.energy_current",
            fromlist=["stencil_links"],
        ).stencil_links(parameters.stencil, oriented=False)
    ):
        state.weak_links[..., index, :, :] = expm(
            1.0j * amplitude * offset[0] * generator
        )
    energy = total_hamiltonian(state, parameters)[0]
    assert energy > 0.0


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r4_added_rates_are_hamiltonian_gradients(
    stencil: str,
) -> None:
    state, parameters = _seeded_state(stencil=stencil)
    _, rates, _ = potential_energy_and_rates(state, parameters)
    epsilon = 1.0e-7
    site = (0, 0, 0)
    link = 0

    def derivative(plus_update, minus_update) -> float:
        plus = state.copy()
        minus = state.copy()
        plus_update(plus)
        minus_update(minus)
        return (
            _potential(plus, parameters) - _potential(minus, parameters)
        ) / (2.0 * epsilon)

    weak_matter_derivative = derivative(
        lambda value: value.weak_matter.__setitem__(
            site + (0,),
            value.weak_matter[site + (0,)] + epsilon,
        ),
        lambda value: value.weak_matter.__setitem__(
            site + (0,),
            value.weak_matter[site + (0,)] - epsilon,
        ),
    )
    assert np.isclose(
        weak_matter_derivative,
        -rates.weak_matter[site + (0,)].real,
        rtol=3.0e-5,
        atol=3.0e-7,
    )

    weak_generator = su2_generators()[1]
    weak_link_derivative = derivative(
        lambda value: value.weak_links.__setitem__(
            site + (link,),
            expm(1.0j * epsilon * weak_generator)
            @ value.weak_links[site + (link,)],
        ),
        lambda value: value.weak_links.__setitem__(
            site + (link,),
            expm(-1.0j * epsilon * weak_generator)
            @ value.weak_links[site + (link,)],
        ),
    )
    assert np.isclose(
        weak_link_derivative,
        -rates.weak_electric[site + (link, 1)],
        rtol=4.0e-5,
        atol=4.0e-7,
    )

    higgs_derivative = derivative(
        lambda value: value.higgs_orientation.__setitem__(
            site,
            expm(1.0j * epsilon * weak_generator)
            @ value.higgs_orientation[site],
        ),
        lambda value: value.higgs_orientation.__setitem__(
            site,
            expm(-1.0j * epsilon * weak_generator)
            @ value.higgs_orientation[site],
        ),
    )
    assert np.isclose(
        higgs_derivative,
        -rates.higgs_electric[site + (1,)],
        rtol=4.0e-5,
        atol=4.0e-7,
    )

    chi_derivative = derivative(
        lambda value: value.r3.chi.__setitem__(
            site,
            value.r3.chi[site] + epsilon,
        ),
        lambda value: value.r3.chi.__setitem__(
            site,
            value.r3.chi[site] - epsilon,
        ),
    )
    assert np.isclose(
        chi_derivative,
        -rates.r3.chi[site],
        rtol=4.0e-5,
        atol=4.0e-7,
    )


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r4_weak_local_covariance_of_potential(
    stencil: str,
) -> None:
    state, parameters = _seeded_state(43, stencil)
    state.r3.matter_momentum.fill(0.0)
    state.weak_momentum.fill(0.0)
    state.weak_electric.fill(0.0)
    state.higgs_electric.fill(0.0)
    before = _potential(state, parameters)
    rng = np.random.default_rng(47)
    generators = su2_generators()
    transformations = np.empty(
        state.r3.chi.shape + (2, 2),
        dtype=np.complex128,
    )
    for site in np.ndindex(state.r3.chi.shape):
        transformations[site] = expm(
            1.0j
            * np.einsum(
                "a,aij->ij",
                0.2 * rng.normal(size=3),
                generators,
            )
        )
    state.weak_matter = np.einsum(
        "...ab,...b->...a",
        transformations,
        state.weak_matter,
    )
    state.higgs_orientation = (
        transformations @ state.higgs_orientation
    )
    unique = __import__(
        "lfm.analysis.energy_current",
        fromlist=["stencil_links"],
    ).stencil_links(parameters.stencil, oriented=False)
    for index, (offset, _) in enumerate(unique):
        neighbor_transformation = _neighbor(transformations, offset)
        state.weak_links[..., index, :, :] = (
            transformations
            @ state.weak_links[..., index, :, :]
            @ np.swapaxes(
                neighbor_transformation.conj(),
                -1,
                -2,
            )
        )
    after = _potential(state, parameters)
    assert abs(after - before) / max(abs(before), 1.0) < 2.0e-13


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r4_u1_su3_local_covariance_of_potential(
    stencil: str,
) -> None:
    state, parameters = _seeded_state(49, stencil)
    before = _potential(state, parameters)
    rng = np.random.default_rng(51)
    phases = 0.2 * rng.normal(size=state.r3.chi.shape)
    phase_transform = np.exp(1.0j * phases)
    generators = su3_generators()
    color_transform = np.empty(
        state.r3.chi.shape + (3, 3),
        dtype=np.complex128,
    )
    for site in np.ndindex(state.r3.chi.shape):
        color_transform[site] = expm(
            1.0j
            * np.einsum(
                "a,aij->ij",
                0.1 * rng.normal(size=8),
                generators,
            )
        )
    state.r3.matter = (
        phase_transform[..., np.newaxis]
        * np.einsum(
            "...ab,...b->...a",
            color_transform,
            state.r3.matter,
        )
    )
    state.weak_matter *= phase_transform[..., np.newaxis]
    unique = __import__(
        "lfm.analysis.energy_current",
        fromlist=["stencil_links"],
    ).stencil_links(parameters.stencil, oriented=False)
    for index, (offset, _) in enumerate(unique):
        neighbor_phase = _neighbor(phase_transform, offset)
        neighbor_color = _neighbor(color_transform, offset)
        state.r3.phase_links[..., index] *= (
            phase_transform * np.conj(neighbor_phase)
        )
        state.r3.color_links[..., index, :, :] = (
            color_transform
            @ state.r3.color_links[..., index, :, :]
            @ np.swapaxes(neighbor_color.conj(), -1, -2)
        )
    after = _potential(state, parameters)
    assert abs(after - before) / max(abs(before), 1.0) < 3.0e-13


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r4_live_reversal_groups_and_energy_order(
    stencil: str,
) -> None:
    initial, parameters = _seeded_state(53, stencil)

    reverse = initial.copy()
    for _ in range(3):
        step_r4(reverse, 1.0e-4, parameters)
    reverse_momenta(reverse)
    for _ in range(3):
        step_r4(reverse, 1.0e-4, parameters)
    reverse_momenta(reverse)
    assert state_distance(initial, reverse) < 5.0e-10
    assert max(group_constraint_errors(reverse).values()) < 2.0e-12

    def endpoint(dt: float, duration: float) -> tuple[R4State, float]:
        state = initial.copy()
        energies = [total_hamiltonian(state, parameters)[0]]
        for _ in range(int(round(duration / dt))):
            step_r4(state, dt, parameters)
            energies.append(total_hamiltonian(state, parameters)[0])
        span = (max(energies) - min(energies)) / max(
            abs(energies[0]),
            1.0,
        )
        return state, span

    coarse, coarse_span = endpoint(2.0e-4, 8.0e-4)
    medium, medium_span = endpoint(1.0e-4, 8.0e-4)
    fine, fine_span = endpoint(5.0e-5, 8.0e-4)
    assert coarse_span < 2.0e-6
    assert medium_span < coarse_span
    assert fine_span < medium_span
    assert state_distance(coarse, medium) / state_distance(medium, fine) > 3.0


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r4_scalar_frame_sector_matches_full_r4(stencil: str) -> None:
    parameters = R4Parameters(
        r3=R3LiveParameters(stencil=stencil)
    )
    rng = np.random.default_rng(71)
    scalar = R4FrameScalarState.vacuum(3)
    scalar.shape_amplitude = 1.0e-4 * rng.normal(
        size=scalar.shape_amplitude.shape
    )
    scalar.shape_momentum = 1.0e-4 * rng.normal(
        size=scalar.shape_momentum.shape
    )
    full = R4State.vacuum(3, parameters)
    projector = _temporal_shape_projector()
    full.r3.shape = (
        scalar.shape_amplitude[..., np.newaxis, np.newaxis]
        * projector
    )
    full.r3.shape_momentum = (
        scalar.shape_momentum[..., np.newaxis, np.newaxis]
        * projector
    )
    full_energy = total_hamiltonian(full, parameters)[0]
    scalar_energy = r4_frame_scalar_energy(
        scalar,
        parameters,
    )[0]
    assert abs(full_energy - scalar_energy) < 1.0e-13
    step = 1.0e-4
    step_r4(full, step, parameters)
    step_r4_frame_scalar(scalar, step, parameters)
    recovered_amplitude = (
        np.sum(full.r3.shape * projector, axis=(-2, -1))
        / np.sum(projector**2)
    )
    recovered_momentum = (
        np.sum(
            full.r3.shape_momentum * projector,
            axis=(-2, -1),
        )
        / np.sum(projector**2)
    )
    assert np.max(
        np.abs(recovered_amplitude - scalar.shape_amplitude)
    ) < 2.0e-15
    assert np.max(
        np.abs(recovered_momentum - scalar.shape_momentum)
    ) < 2.0e-14
    assert np.max(np.abs(full.r3.frame_electric)) < 2.0e-18


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_r4_scalar_frame_fixed_source_reverses(stencil: str) -> None:
    parameters = R4Parameters(
        r3=R3LiveParameters(stencil=stencil)
    )
    initial = R4FrameScalarState.vacuum(5)
    source = np.zeros(initial.shape_amplitude.shape)
    source[2, 2, 2] = 1.0
    state = initial.copy()
    for _ in range(10):
        step_r4_frame_scalar(
            state,
            1.0e-2,
            parameters,
            source_density=source,
        )
    state.shape_momentum *= -1.0
    for _ in range(10):
        step_r4_frame_scalar(
            state,
            1.0e-2,
            parameters,
            source_density=source,
        )
    state.shape_momentum *= -1.0
    assert np.max(np.abs(state.shape_amplitude)) < 2.0e-16
    assert np.max(np.abs(state.shape_momentum)) < 2.0e-16
