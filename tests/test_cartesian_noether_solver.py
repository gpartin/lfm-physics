from __future__ import annotations

import numpy as np

from lfm.constants import CHI0
from lfm.particles.noether import (
    cartesian_fixed_charge_energy_and_gradient,
    solve_cartesian_noether_soliton,
)


def test_cartesian_fixed_charge_gradient_matches_directional_difference() -> None:
    grid_size = 8
    dx = 0.25
    axis = (np.arange(grid_size, dtype=np.float64) - 0.5 * (grid_size - 1)) * dx
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    radius_sq = x * x + y * y + z * z
    phi = 1.7 * np.exp(-radius_sq / 0.45)
    chi = CHI0 - 0.8 * np.exp(-radius_sq / 0.65)
    variables = np.concatenate((phi.ravel(), chi.ravel()))
    rng = np.random.default_rng(19)
    direction = rng.normal(size=variables.size)
    direction /= np.linalg.norm(direction)

    ledger, gradient = cartesian_fixed_charge_energy_and_gradient(
        variables,
        target_charge=150.0,
        grid_size=grid_size,
        dx=dx,
    )
    epsilon = 1.0e-5
    plus, _ = cartesian_fixed_charge_energy_and_gradient(
        variables + epsilon * direction,
        target_charge=150.0,
        grid_size=grid_size,
        dx=dx,
    )
    minus, _ = cartesian_fixed_charge_energy_and_gradient(
        variables - epsilon * direction,
        target_charge=150.0,
        grid_size=grid_size,
        dx=dx,
    )
    finite_difference = (plus.total - minus.total) / (2.0 * epsilon)
    analytic = float(np.dot(gradient, direction))
    assert np.isfinite(ledger.total)
    assert np.isclose(finite_difference, analytic, rtol=2.0e-5, atol=2.0e-5)


def test_cartesian_solver_does_not_raise_fixed_charge_energy() -> None:
    grid_size = 8
    dx = 0.25
    axis = (np.arange(grid_size, dtype=np.float64) - 0.5 * (grid_size - 1)) * dx
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    radius_sq = x * x + y * y + z * z
    phi = 2.0 * np.exp(-radius_sq / 0.5)
    chi = CHI0 - 1.0 * np.exp(-radius_sq / 0.7)
    initial, _ = cartesian_fixed_charge_energy_and_gradient(
        np.concatenate((phi.ravel(), chi.ravel())),
        target_charge=200.0,
        grid_size=grid_size,
        dx=dx,
    )
    solution = solve_cartesian_noether_soliton(
        initial_phi=phi,
        initial_chi=chi,
        target_charge=200.0,
        dx=dx,
        max_iterations=8,
        history=3,
    )
    assert np.all(np.isfinite(solution.phi))
    assert np.all(np.isfinite(solution.chi))
    assert np.isclose(solution.charge, 200.0)
    assert solution.energy.total <= initial.total * (1.0 + 1.0e-10)
