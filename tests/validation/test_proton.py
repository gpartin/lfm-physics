"""Task 1.5 — Proton stability.

H₀: Proton soliton disperses (energy < 50 % retained).
H₁: Proton is extremely stable (energy > 50 % retained, deep χ well).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only
"""

from __future__ import annotations

import pytest

from lfm.config_presets import gravity_em

from .conftest import measure_particle


@pytest.fixture(scope="class")
def metrics():
    return measure_particle("proton", gravity_em)


class TestProtonStability:
    """Proton should be a highly stable, localised soliton."""

    def test_chi_well_forms(self, metrics) -> None:
        assert metrics.chi_min < metrics.chi0 - 0.01

    def test_energy_retained(self, metrics) -> None:
        assert metrics.retention > 0.50, f"Proton dissolved: {metrics.retention:.1%} retained"

    def test_localisation_persists(self, metrics) -> None:
        assert metrics.peak > 1e-4
