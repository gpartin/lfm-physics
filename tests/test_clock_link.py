"""Tests for the unpromoted LFM clock-link candidate diagnostics."""

from __future__ import annotations

import numpy as np

from lfm.analysis.clock_link import (
    ClockLinkParameters,
    clock_link_frequency_sq,
    clock_link_green_residue,
    clock_link_static_response,
    matter_clock_sensitivity,
    matter_frequency_sq,
    solve_static_clock_link,
    static_clock_link_residual,
)
from lfm.constants import CHI0, KAPPA


def test_default_inertia_reuses_substrate_normalization() -> None:
    parameters = ClockLinkParameters()
    assert parameters.inertia == CHI0 / KAPPA
    assert clock_link_green_residue(parameters) == -KAPPA / CHI0


def test_clock_link_branch_is_gapless() -> None:
    stiffness = np.asarray([0.0, 0.1, 1.0])
    frequency_sq = clock_link_frequency_sq(stiffness)
    assert frequency_sq[0] == 0.0
    assert np.all(frequency_sq[1:] > 0.0)


def test_static_response_has_nonzero_one_over_k_residue() -> None:
    stiffness = np.asarray([1.0e-2, 1.0e-4, 1.0e-6])
    response = clock_link_static_response(stiffness)
    expected = clock_link_green_residue()
    np.testing.assert_allclose(stiffness * response, expected)


def test_matter_frequency_reads_clock_factor() -> None:
    base = matter_frequency_sq(0.2, CHI0, 0.0)
    shifted = matter_frequency_sq(0.2, CHI0, 0.1)
    assert shifted > base
    np.testing.assert_allclose(
        matter_clock_sensitivity(0.2, CHI0),
        2.0 * base,
    )


def test_periodic_static_solver_closes_for_both_stencils() -> None:
    source = np.zeros((20, 20, 20), dtype=np.float64)
    source[10, 10, 10] = 1.0
    for stencil in ("19", "27"):
        field = solve_static_clock_link(source, stencil=stencil)
        residual = static_clock_link_residual(
            field,
            source,
            stencil=stencil,
        )
        assert float(np.max(np.abs(residual))) < 1.0e-12
        assert abs(float(np.mean(field))) < 1.0e-15

