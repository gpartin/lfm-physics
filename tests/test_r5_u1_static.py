"""Tests for Gauss-constrained R5 U(1) probes."""

from __future__ import annotations

import numpy as np
import pytest

from lfm.foundations.r3_link_frame_live import R3LiveParameters
from lfm.foundations.r4_unified_live import R4Parameters
from lfm.foundations.r5_u1_static import (
    periodic_point_pair_charge,
    solve_u1_gauss_minimum,
)
from lfm.foundations.r5_unified_live import R5Parameters


@pytest.mark.parametrize("stencil", ["19", "27"])
@pytest.mark.parametrize("relative_sign", [-1, 1])
def test_periodic_u1_point_pair_satisfies_gauss(
    stencil: str,
    relative_sign: int,
) -> None:
    parameters = R5Parameters(
        r4=R4Parameters(
            r3=R3LiveParameters(stencil=stencil)
        )
    )
    charge = periodic_point_pair_charge(9, 2, relative_sign)
    state = solve_u1_gauss_minimum(
        charge,
        parameters,
        tolerance=1.0e-12,
    )
    assert abs(float(np.sum(charge))) < 1.0e-12
    assert state.gauss_residual < 1.0e-10
    assert state.electric_energy > 0.0
