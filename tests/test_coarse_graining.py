"""Tests for finite local Fourier blocking diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from lfm.analysis.coarse_graining import (
    block_window_magnitude_sq,
    blocked_static_propagator,
    inverse_response_intercept,
    response_log_slope,
)


@pytest.mark.parametrize("block_factor", [1, 2, 4, 8])
def test_alias_windows_form_partition(block_factor: int) -> None:
    coarse_k = np.asarray([0.0, 0.17, 1.1])
    weight_sum = np.zeros_like(coarse_k)
    for alias in range(block_factor):
        fine_k = (coarse_k + 2.0 * np.pi * alias) / block_factor
        weight_sum += block_window_magnitude_sq(fine_k, block_factor)
    np.testing.assert_allclose(weight_sum, 1.0, atol=2.0e-14)


@pytest.mark.parametrize("stencil", ["19", "27"])
@pytest.mark.parametrize("block_factor", [1, 2, 4, 8])
def test_gapped_blocked_response_retains_intercept(
    stencil: str,
    block_factor: int,
) -> None:
    wave_number = np.geomspace(1.0e-6, 1.0e-2, 32)
    response = blocked_static_propagator(
        wave_number,
        0.0,
        0.0,
        block_factor=block_factor,
        mass_sq=372.64516129032256,
        stencil=stencil,
    )
    slope = response_log_slope(wave_number / block_factor, response, count=12)
    intercept = inverse_response_intercept(
        wave_number / block_factor,
        response,
        count=12,
    )
    assert abs(slope) < 1.0e-4
    assert intercept > 300.0


@pytest.mark.parametrize("stencil", ["19", "27"])
@pytest.mark.parametrize("block_factor", [1, 2, 4, 8])
def test_massless_control_retains_inverse_square_pole(
    stencil: str,
    block_factor: int,
) -> None:
    wave_number = np.geomspace(1.0e-5, 1.0e-2, 32)
    response = blocked_static_propagator(
        wave_number,
        0.0,
        0.0,
        block_factor=block_factor,
        mass_sq=0.0,
        stencil=stencil,
    )
    slope = response_log_slope(wave_number / block_factor, response, count=12)
    assert -2.01 < slope < -1.99


def test_invalid_mass_and_zero_mass_uniform_mode_rejected() -> None:
    with pytest.raises(ValueError):
        blocked_static_propagator(
            0.1,
            0.0,
            0.0,
            block_factor=2,
            mass_sq=-1.0,
        )
    with pytest.raises(ValueError):
        blocked_static_propagator(
            0.0,
            0.0,
            0.0,
            block_factor=2,
            mass_sq=0.0,
        )
