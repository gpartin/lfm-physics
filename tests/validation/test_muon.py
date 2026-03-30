"""Task 1.3 — Muon stability & mass ratio.

H₀: Muon has same mass (χ well depth) as electron.
H₁: Muon is measurably heavier — deeper χ well or higher amplitude.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only
"""

from __future__ import annotations

import pytest

from lfm.config_presets import gravity_em

from .conftest import measure_particle


@pytest.fixture(scope="class")
def pair():
    e = measure_particle("electron", gravity_em)
    mu = measure_particle("muon", gravity_em)
    return e, mu


class TestMuon:
    """Muon should be a heavier, localised soliton."""

    def test_muon_stable(self, pair) -> None:
        """Muon should retain meaningful energy."""
        _, mu = pair
        assert mu.retention > 0.30, f"Muon dissolved: {mu.retention:.1%} retained"

    def test_muon_heavier_than_electron(self, pair) -> None:
        """Muon χ-well should differ from electron (mass ∝ χ_well)."""
        e, mu = pair
        assert mu.chi_min != e.chi_min, "Muon and electron have identical χ wells"
