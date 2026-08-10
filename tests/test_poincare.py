"""Tests for exact lattice Poincare-emergence diagnostics."""

from __future__ import annotations

import math

import numpy as np

from lfm.analysis.poincare import (
    STENCIL_19,
    STENCIL_27,
    cubic_rotation_residual,
    discrete_omega,
    fibonacci_sphere,
    gaussian_packet,
    group_velocity,
    max_spatial_eigenvalue,
    poincare_algebra_matrix_residual,
    stencil_symbol,
    symanzik_coefficients,
    time_composition_residual,
    translation_equivariance_residual,
)


def test_stencil_symbols_have_correct_axis_limit():
    k = np.asarray([0.01, 0.0, 0.0])
    for stencil in (STENCIL_19, STENCIL_27):
        symbol = float(stencil_symbol(k, stencil=stencil))
        assert math.isclose(symbol, -0.01**2, rel_tol=1.0e-5)


def test_exact_cubic_rotation_invariance():
    k = np.asarray([0.37, -0.51, 0.83])
    assert cubic_rotation_residual(k, stencil="19") < 1.0e-14
    assert cubic_rotation_residual(k, stencil="27") < 1.0e-14


def test_low_k_dispersion_and_group_velocity():
    directions = fibonacci_sphere(32)
    k = 1.0e-3 * directions
    for stencil in ("19", "27"):
        omega = discrete_omega(k, dt=1.0e-4, spacing=1.0, stencil=stencil)
        velocity = group_velocity(k, dt=1.0e-4, spacing=1.0, stencil=stencil)
        radial = np.sum(velocity * directions, axis=-1)
        assert np.max(np.abs(omega / 1.0e-3 - 1.0)) < 1.0e-6
        assert np.max(np.abs(radial - 1.0)) < 1.0e-6


def test_stability_extrema():
    assert math.isclose(max_spatial_eigenvalue(stencil="19"), 16.0 / 3.0)
    assert math.isclose(max_spatial_eigenvalue(stencil="27"), 52.0 / 9.0)


def test_symanzik_coefficients_share_isotropic_quartic_term():
    for stencil in ("19", "27"):
        coefficients = symanzik_coefficients(stencil)
        assert math.isclose(coefficients["quartic_pure"], 1.0 / 12.0)
        assert math.isclose(coefficients["quartic_mixed"], 1.0 / 6.0)


def test_continuum_poincare_matrix_algebra_closes():
    residuals = poincare_algebra_matrix_residual()
    assert residuals["maximum"] == 0.0


def test_integer_translation_and_time_composition():
    field = gaussian_packet(
        12,
        length=12.0,
        center=(3.0, 4.0, 5.0),
        direction=(1.0, 1.0, 0.5),
        k_magnitude=0.5,
        sigma=1.2,
    )
    translation = translation_equivariance_residual(
        field,
        shift=(2, -1, 3),
        time=0.4,
        length=12.0,
        dt=0.05,
        stencil="19",
    )
    composition = time_composition_residual(
        field,
        time_1=0.2,
        time_2=0.3,
        length=12.0,
        dt=0.05,
        stencil="19",
    )
    assert translation < 1.0e-12
    assert composition < 1.0e-12
