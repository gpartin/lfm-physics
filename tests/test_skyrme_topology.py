from __future__ import annotations

import numpy as np

from lfm.topology import (
    FRQuantization,
    hedgehog_degree,
    hedgehog_energy_gradient_hessian,
    make_hedgehog_profile,
    solve_skyrme_hedgehog,
)


def test_hedgehog_gradient_and_hessian_match_finite_differences() -> None:
    radius = 2.0
    dr = 0.1
    profile = make_hedgehog_profile(radius=radius, dr=dr)
    _, gradient, hessian, _ = hedgehog_energy_gradient_hessian(
        profile,
        radius=radius,
        dr=dr,
    )
    dense = hessian.toarray()
    assert np.allclose(dense, dense.T, rtol=0.0, atol=1.0e-12)

    epsilon = 1.0e-6
    for interior_index in (0, 4, 9, 17):
        profile_index = interior_index + 1
        plus = profile.copy()
        minus = profile.copy()
        plus[profile_index] += epsilon
        minus[profile_index] -= epsilon
        plus_energy, plus_gradient, _, _ = hedgehog_energy_gradient_hessian(
            plus,
            radius=radius,
            dr=dr,
        )
        minus_energy, minus_gradient, _, _ = hedgehog_energy_gradient_hessian(
            minus,
            radius=radius,
            dr=dr,
        )
        energy_difference = (
            plus_energy.total - minus_energy.total
        ) / (2.0 * epsilon)
        gradient_difference = (
            plus_gradient - minus_gradient
        ) / (2.0 * epsilon)
        assert np.isclose(
            gradient[interior_index],
            energy_difference,
            rtol=2.0e-7,
            atol=2.0e-7,
        )
        assert np.allclose(
            dense[:, interior_index],
            gradient_difference,
            rtol=2.0e-6,
            atol=2.0e-5,
        )


def test_hedgehog_solver_produces_stationary_unit_degree_state() -> None:
    solution = solve_skyrme_hedgehog(radius=6.0, dr=0.1)
    assert solution.stationary_relative_residual < 1.0e-9
    assert abs(solution.degree - 1.0) < 2.0e-3
    assert solution.unitarity_residual < 1.0e-14
    assert solution.derrick_relative_first_derivative < 1.0e-9
    assert solution.derrick_relative_second_derivative > 0.01
    assert solution.continuum_virial_mismatch < 0.03
    assert np.isclose(
        hedgehog_degree(
            solution.profile,
            radius=solution.radius,
            dr=solution.dr,
        ),
        solution.degree,
    )


def test_fr_nontrivial_character_gives_odd_degree_fermionic_signs() -> None:
    quantization = FRQuantization(degree=1, deck_character=-1)
    assert quantization.rotation_loop_class == 1
    assert quantization.exchange_loop_class == 1
    assert quantization.rotation_sign(1) == -1
    assert quantization.rotation_sign(2) == 1
    assert quantization.exchange_sign(1) == -1
    assert quantization.exchange_sign(2) == 1
    assert quantization.wavefunction_on_sheet(2.0 + 3.0j, 1) == -2.0 - 3.0j


def test_fr_even_degree_rotation_loop_is_contractible() -> None:
    quantization = FRQuantization(degree=2, deck_character=-1)
    assert quantization.rotation_loop_class == 0
    assert quantization.rotation_sign(1) == 1
    assert quantization.exchange_sign(1) == 1
