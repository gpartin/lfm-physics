from __future__ import annotations

import numpy as np

from lfm.particles.stationary import continue_support_removal


C7_OFFSETS = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def test_support_removal_preserves_norm_and_responds_to_dynamic_source() -> None:
    grid = 9
    source_total = 1_847_183.1602203925
    fixed_source = np.zeros((grid, grid, grid), dtype=np.float64)
    center = grid // 2
    for dx, dy, dz in C7_OFFSETS:
        fixed_source[center + dx, center + dy, center + dz] = source_total / 7.0

    fixed, partly_dynamic = continue_support_removal(
        fixed_source,
        [0.0, 0.01],
        max_cycles=80,
        tolerance=1.0e-5,
        mixing=0.4,
    )

    assert fixed.converged
    assert partly_dynamic.converged
    assert np.isclose(np.sum(fixed.source_density), source_total)
    assert np.isclose(np.sum(partly_dynamic.source_density), source_total)
    assert np.allclose(fixed.source_density, fixed_source)
    assert partly_dynamic.effective_sites < fixed.effective_sites
    assert partly_dynamic.chi_min < fixed.chi_min
