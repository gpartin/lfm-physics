"""Task 1.1 — Electron stability & mass.

H₀: Electron soliton disperses (energy < 50 % retained).
H₁: Electron is a stable bound state (energy > 50 % retained,
     localised χ well persists after evolution).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only (via lfm.Simulation)
  ✓ No external potentials, Coulomb, or QED injected
"""

from __future__ import annotations

import pytest

from lfm.config_presets import gravity_em

from .conftest import measure_particle


@pytest.fixture(scope="class")
def metrics():
    return measure_particle("electron", gravity_em)


class TestElectronStability:
    """Electron should survive as a localised soliton."""

    def test_chi_well_forms(self, metrics) -> None:
        """After equilibration, χ must be below χ₀ at the soliton centre."""
        assert metrics.chi_min < metrics.chi0 - 0.01

    def test_energy_retained(self, metrics) -> None:
        """Reject H₀ if > 50 % of initial energy survives."""
        assert metrics.retention > 0.50, f"Electron dissolved: {metrics.retention:.1%} retained"

    def test_localisation_persists(self, metrics) -> None:
        """Peak |Ψ|² should remain well above noise after evolution."""
        assert metrics.peak > 1e-4, "No localised energy peak found"
