"""Tests for the experimental local domain-wall weak completion."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from lfm.foundations.parsimonious_domain_wall import (
    P4FDWParameters,
    apply_domain_wall,
    apply_domain_wall_adjoint,
    domain_wall_potential_and_rates,
    project_domain_wall_register,
    reverse_momenta,
    state_distance,
    step_p4f_domain_wall,
    total_hamiltonian,
    vacuum_state,
)
from lfm.foundations.r3_link_frame_live import (
    _dagger,
    _link_table,
    _neighbor,
)
from lfm.foundations.r4_unified_live import (
    group_constraint_errors,
    su2_generators,
)


def _parameters(depth: int = 4) -> P4FDWParameters:
    return P4FDWParameters(internal_depth=depth)


def _random_field(
    shape: tuple[int, ...],
    seed: int,
    scale: float = 0.02,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = scale * (rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape))
    return project_domain_wall_register(values)


def _random_site_gauge(
    size: int,
    seed: int,
    scale: float = 0.3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    generators = su2_generators()
    gauge = np.empty((size, size, size, 2, 2), dtype=np.complex128)
    for site in np.ndindex((size, size, size)):
        coefficients = scale * rng.standard_normal(3)
        gauge[site] = expm(
            1.0j
            * np.einsum(
                "a,aij->ij",
                coefficients,
                generators,
                optimize=True,
            )
        )
    return gauge


def _transform_field(
    gauge: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "...ij,...sajc->...saic",
        gauge,
        field,
        optimize=True,
    )


def _transform_links(
    gauge: np.ndarray,
    links: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(links)
    unique, _ = _link_table("19")
    for index, (offset, _) in enumerate(unique):
        transformed[..., index, :, :] = (
            gauge @ links[..., index, :, :] @ _dagger(_neighbor(gauge, offset))
        )
    return transformed


def test_domain_wall_operator_adjointness() -> None:
    parameters = _parameters()
    state = vacuum_state(2, parameters)
    state.domain_wall_field = _random_field(
        state.domain_wall_field.shape,
        seed=4101,
    )
    probe = _random_field(
        state.domain_wall_field.shape,
        seed=4102,
    )
    gauge = _random_site_gauge(2, seed=4103)
    state.base.weak_links = _transform_links(
        gauge,
        state.base.weak_links,
    )
    applied = apply_domain_wall(
        state.domain_wall_field,
        state.base.weak_links,
        parameters,
    )
    adjoint_applied = apply_domain_wall_adjoint(
        probe,
        state.base.weak_links,
        parameters,
    )
    left = np.vdot(applied, probe)
    right = np.vdot(state.domain_wall_field, adjoint_applied)
    scale = max(abs(left), abs(right), 1.0)
    assert abs(left - right) / scale < 1.0e-12


def test_domain_wall_operator_is_locally_su2_covariant() -> None:
    parameters = _parameters()
    state = vacuum_state(2, parameters)
    field = _random_field(
        state.domain_wall_field.shape,
        seed=4201,
    )
    gauge = _random_site_gauge(2, seed=4202)
    transformed_field = _transform_field(gauge, field)
    transformed_links = _transform_links(
        gauge,
        state.base.weak_links,
    )
    original_output = apply_domain_wall(
        field,
        state.base.weak_links,
        parameters,
    )
    transformed_output = apply_domain_wall(
        transformed_field,
        transformed_links,
        parameters,
    )
    expected = _transform_field(gauge, original_output)
    residual = np.linalg.norm(transformed_output - expected) / max(
        np.linalg.norm(expected),
        1.0,
    )
    assert residual < 1.0e-12


def test_domain_wall_weak_link_rate_is_action_gradient() -> None:
    parameters = _parameters()
    state = vacuum_state(2, parameters)
    state.domain_wall_field = _random_field(
        state.domain_wall_field.shape,
        seed=4301,
    )
    _, _, electric_rate = domain_wall_potential_and_rates(
        state,
        parameters,
    )
    site = (0, 0, 0)
    link_index = 0
    generator_index = 1
    generator = su2_generators()[generator_index]
    epsilon = 1.0e-7

    energies = []
    for sign in (-1.0, 1.0):
        varied = state.copy()
        varied.base.weak_links[site + (link_index,)] = (
            expm(1.0j * sign * epsilon * generator) @ varied.base.weak_links[site + (link_index,)]
        )
        energy, _, _ = domain_wall_potential_and_rates(
            varied,
            parameters,
        )
        energies.append(energy)
    numerical_derivative = (energies[1] - energies[0]) / (2.0 * epsilon)
    analytic_derivative = -electric_rate[site + (link_index, generator_index)]
    assert np.isclose(
        analytic_derivative,
        numerical_derivative,
        rtol=2.0e-6,
        atol=2.0e-9,
    )


def test_domain_wall_step_is_reversible_and_energy_bounded() -> None:
    parameters = _parameters()
    initial = vacuum_state(2, parameters)
    initial.domain_wall_field = _random_field(
        initial.domain_wall_field.shape,
        seed=4401,
        scale=0.002,
    )
    initial.domain_wall_momentum = _random_field(
        initial.domain_wall_momentum.shape,
        seed=4402,
        scale=0.001,
    )
    evolved = initial.copy()
    energy_initial, _ = total_hamiltonian(evolved, parameters)
    dt = 1.0e-4
    steps = 5
    for _ in range(steps):
        step_p4f_domain_wall(evolved, dt, parameters)
    energy_final, _ = total_hamiltonian(evolved, parameters)
    relative_drift = abs(energy_final - energy_initial) / max(
        abs(energy_initial),
        1.0,
    )
    assert relative_drift < 1.0e-8
    assert max(group_constraint_errors(evolved.base).values()) < 1.0e-12

    returned = reverse_momenta(evolved)
    for _ in range(steps):
        step_p4f_domain_wall(returned, dt, parameters)
    returned = reverse_momenta(returned)
    assert state_distance(initial, returned) < 1.0e-10


def test_single_wall_register_has_one_weyl_pair_and_no_mirror() -> None:
    parameters = _parameters()
    state = vacuum_state(2, parameters)
    reduced_shape = (2, 2, 2, parameters.internal_depth, 4)
    basis = [
        index
        for index in np.ndindex(reduced_shape)
        if not (index[3] == parameters.internal_depth - 1 and index[4] < 2)
    ]
    matrix = np.zeros(
        (int(np.prod(reduced_shape)), len(basis)),
        dtype=np.complex128,
    )
    for column, index in enumerate(basis):
        field = np.zeros_like(state.domain_wall_field)
        field[index + (0, 0)] = 1.0
        output = apply_domain_wall(
            field,
            state.base.weak_links,
            parameters,
        )
        matrix[:, column] = output[..., 0, 0].reshape(-1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert np.sum(singular_values < 1.0e-10) == 2
    assert singular_values[-3] > 0.9

    forbidden = state.copy()
    forbidden.domain_wall_field[..., -1, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="excluded right-wall mirror"):
        total_hamiltonian(forbidden, parameters)
