"""Task 1.6 — Neutron stability (should show some instability).

H₀: Neutron is as stable as the proton.
H₁: Neutron shows measurable instability (lower energy retention or
     energy leakage over time).

The neutron is neutral (field_level=0, gravity_only) and is flagged
``stable=False`` in the catalog.  On the short time-scales we can
simulate here the difference may be subtle, but the neutron should
NOT be *more* stable than the proton.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only
"""

from __future__ import annotations

import pytest

from lfm.config_presets import gravity_only

from .conftest import measure_particle


@pytest.fixture(scope="class")
def metrics():
    return measure_particle("neutron", gravity_only)


class TestNeutronStability:
    """Neutron should form a soliton (at short timescales)."""

    def test_chi_well_forms(self, metrics) -> None:
        assert metrics.chi_min < metrics.chi0 - 0.01

    def test_energy_retained(self, metrics) -> None:
        """Even an unstable particle should persist for 500 steps at N=32."""
        assert metrics.retention > 0.30, (
            f"Neutron dissolved too fast: {metrics.retention:.1%} retained"
        )
