from __future__ import annotations

import numpy as np

from lfm.foundations.r4_color_static import (
    _chi_energy_gradient,
    r4_color_static_energy,
    relax_r4_chi_at_fixed_color_electric,
    relax_r4_color_static,
    solve_color_gauss_minimum,
)
from lfm.foundations.r4_unified_live import R4Parameters


def _point_pair(size: int, separation: int, charge: float) -> np.ndarray:
    values = np.zeros((size, size, size), dtype=np.float64)
    center = size // 2
    left = center - separation // 2
    right = left + separation
    values[left, center, center] = charge
    values[right, center, center] = -charge
    return values


def test_r4_color_gauss_minimum_and_chi_gradient() -> None:
    parameters = R4Parameters()
    rng = np.random.default_rng(81)
    chi = parameters.r3.chi0 + 1.0e-3 * rng.normal(size=(5, 5, 5))
    charge = _point_pair(5, 2, 2.0)
    potential, electric, residual = solve_color_gauss_minimum(
        chi,
        charge,
        parameters,
        tolerance=1.0e-12,
    )
    assert residual < 1.0e-10
    gradient = _chi_energy_gradient(chi, electric, parameters)
    site = (2, 2, 2)
    epsilon = 1.0e-6

    def minimized_energy(delta: float) -> float:
        varied = chi.copy()
        varied[site] += delta
        _, varied_electric, varied_residual = solve_color_gauss_minimum(
            varied,
            charge,
            parameters,
            tolerance=1.0e-12,
            initial_potential=potential,
        )
        assert varied_residual < 1.0e-10
        return r4_color_static_energy(
            varied,
            varied_electric,
            parameters,
        )[0]

    derivative = (minimized_energy(epsilon) - minimized_energy(-epsilon)) / (2.0 * epsilon)
    assert np.isclose(
        derivative,
        gradient[site],
        rtol=2.0e-5,
        atol=2.0e-4,
    )


def test_r4_color_relaxation_preserves_gauss_and_is_finite() -> None:
    parameters = R4Parameters()
    charge = _point_pair(5, 2, 20.0)
    result = relax_r4_color_static(
        charge,
        parameters,
        seed=83,
        initial_chi_noise=1.0e-3,
        chi_iterations=20,
        chi_step=1.0e-6,
        gauss_tolerance=1.0e-11,
        gauss_block=5,
    )
    assert result.gauss_residual < 1.0e-9
    assert np.isfinite(result.energy)
    assert np.all(np.isfinite(result.chi))
    assert result.energy_parts["color_electric"] > 0.0


def test_fixed_divergence_free_flux_relaxation() -> None:
    parameters = R4Parameters()
    size = 7
    electric = np.zeros((size, size, size, 9), dtype=np.float64)
    electric[..., 0] = 1.0
    relaxed = relax_r4_chi_at_fixed_color_electric(
        electric,
        parameters,
        seed=11,
        initial_chi_noise=0.0,
        chi_iterations=10,
        chi_step=1.0e-6,
    )
    assert relaxed.gauss_residual < 1.0e-14
    assert np.isfinite(relaxed.energy)
    assert 1.0 <= relaxed.effective_g_squared <= 63.0
