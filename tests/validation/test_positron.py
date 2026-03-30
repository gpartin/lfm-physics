"""Task 1.2 — Positron stability & charge sign.

H₀: Positron behaves differently from electron (different mass or stability).
H₁: Positron = electron with opposite phase (same stability, same χ well).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only
  ✓ Phase π encodes positive charge — no external EM field
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm.config_presets import gravity_em

from .conftest import measure_particle


@pytest.fixture(scope="class")
def pair():
    e = measure_particle("electron", gravity_em)
    p = measure_particle("positron", gravity_em)
    return e, p


class TestPositronStability:
    """Positron should match the electron in mass/stability."""

    def test_positron_stable(self, pair) -> None:
        _, p = pair
        assert p.retention > 0.50, f"Positron dissolved: {p.retention:.1%} retained"

    def test_same_well_depth(self, pair) -> None:
        """χ-well depths should be nearly identical (charge ≠ mass)."""
        e, p = pair
        assert np.isclose(e.chi_min, p.chi_min, rtol=0.05), (
            f"Different well depths: e={e.chi_min:.3f}, p={p.chi_min:.3f}"
        )
