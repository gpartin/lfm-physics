from __future__ import annotations

import numpy as np

from lfm.particles.stationary import solve_stationary_branch_point


def test_stationary_branch_point_satisfies_norm_and_boundaries() -> None:
    point = solve_stationary_branch_point(
        10,
        1000.0,
        max_cycles=40,
        tolerance=1.0e-5,
        mixing=0.5,
    )
    assert point.converged
    assert np.isclose(np.sum(point.phi * point.phi), 1000.0, rtol=1.0e-10)
    assert point.chi_min > 0.0
    assert np.all(point.phi[0] == 0.0)
    assert np.all(point.chi[0] == 19.0)
    assert point.phi_residual < 1.0e-5
    assert point.chi_residual_rms < 1.0e-5
