"""Tests for periodic spatial and leapfrog temporal mode observables."""

from __future__ import annotations

import numpy as np
import pytest

import lfm


def test_periodic_mode_coefficient_handles_a_bank_of_lines() -> None:
    n = 17
    z = np.arange(n, dtype=np.float64)
    amplitudes = np.asarray([1.0 + 0.5j, -0.2 + 0.3j])
    field = amplitudes[:, None] * np.exp(2j * np.pi * 5 * z / n)
    observed = lfm.periodic_mode_coefficient(field, 5, axis=1)
    assert np.allclose(observed, amplitudes, rtol=0.0, atol=1.0e-15)


def test_periodic_mode_coefficient_subtracts_background() -> None:
    n = 16
    z = np.arange(n, dtype=np.float64)
    field = 19.0 + 0.4 * np.exp(2j * np.pi * 3 * z / n)
    observed = lfm.periodic_mode_coefficient(field, 3, background=19.0)
    assert np.isclose(observed, 0.4, rtol=0.0, atol=1.0e-15)


def test_leapfrog_branch_projection_recovers_both_branches() -> None:
    theta = 0.37
    forward = np.asarray([0.7 - 0.2j, -0.1 + 0.4j])
    backward = np.asarray([-0.03 + 0.04j, 0.2 - 0.1j])
    current = forward + backward
    previous = forward * np.exp(1j * theta) + backward * np.exp(-1j * theta)
    observed_forward, observed_backward = lfm.leapfrog_branch_projection(current, previous, theta)
    assert np.allclose(observed_forward, forward, rtol=0.0, atol=1.0e-15)
    assert np.allclose(observed_backward, backward, rtol=0.0, atol=1.0e-15)


def test_leapfrog_branch_projection_rejects_degenerate_frequency() -> None:
    with pytest.raises(ValueError):
        lfm.leapfrog_branch_projection(1.0, 1.0, 0.0)


def test_project_leapfrog_mode_locks_spatial_and_buffer_conventions() -> None:
    n = 20
    z = np.arange(n, dtype=np.float64)
    theta = 0.41
    forward = 0.8 - 0.3j
    backward = -0.04 + 0.06j
    spatial = np.exp(2j * np.pi * 7 * z / n)
    current = (forward + backward) * spatial
    previous = (forward * np.exp(1j * theta) + backward * np.exp(-1j * theta)) * spatial
    observed_forward, observed_backward = lfm.project_leapfrog_mode(current, previous, 7, theta)
    assert abs(observed_forward - forward) <= 1.0e-15
    assert abs(observed_backward - backward) <= 1.0e-15
