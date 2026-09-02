"""Spectral tests for the R4 compact-link loop complex."""

from __future__ import annotations

import numpy as np
import pytest

from lfm.foundations.r4_gauge_spectrum import (
    directional_link_inertia,
    gauge_link_spectrum,
    transverse_mode_speeds,
)


@pytest.mark.parametrize(
    ("stencil", "inertia"),
    [("19", 5.0), ("27", 9.0)],
)
def test_directional_link_inertia_is_geometric(
    stencil: str,
    inertia: float,
) -> None:
    assert directional_link_inertia(stencil) == inertia


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_triangle_only_r4_has_two_spurious_flat_modes(stencil: str) -> None:
    momentum = 2.0 * np.pi / 64.0
    eigenvalues = gauge_link_spectrum(
        stencil,
        (momentum, 0.0, 0.0),
        include_face_squares=False,
    )
    assert np.count_nonzero(np.abs(eigenvalues) < 1.0e-12) == 3


@pytest.mark.parametrize("stencil", ["19", "27"])
def test_geometry_normalized_squares_restore_two_unit_speed_modes(
    stencil: str,
) -> None:
    momentum = 2.0 * np.pi / 256.0
    eigenvalues = gauge_link_spectrum(
        stencil,
        (momentum, 0.0, 0.0),
        include_face_squares=True,
    )
    assert np.count_nonzero(np.abs(eigenvalues) < 1.0e-12) == 1
    speeds = transverse_mode_speeds(
        stencil,
        (momentum, 0.0, 0.0),
        include_face_squares=True,
    )
    assert speeds[0] == pytest.approx(1.0, abs=2.0e-3)
    assert speeds[1] == pytest.approx(1.0, abs=2.0e-3)
