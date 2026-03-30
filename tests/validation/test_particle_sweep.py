"""Task 1.7 — Automated sweep over all catalog particles.

For every particle in the catalog:
  1. Create with the appropriate config preset
  2. Equilibrate + evolve for a modest number of steps
  3. Measure: χ-well depth, energy retention
  4. Report pass/fail per particle

Each particle gets ONE simulation (via ``measure_particle``) and ALL
assertions are checked on the resulting metrics — no redundant sims.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only (via config presets)
"""

from __future__ import annotations

import pytest

from lfm.config_presets import full_physics, gravity_em, gravity_only
from lfm.particles.catalog import PARTICLES

from .conftest import measure_particle

# Build a list of names for parametrisation.
_ALL_NAMES = [p.name for p in PARTICLES.values()]

_PRESET_MAP = {0: gravity_only, 1: gravity_em, 2: full_physics}


def _preset_for(field_level: int):  # noqa: ANN202
    return _PRESET_MAP.get(field_level, full_physics)


# ---------------------------------------------------------------------------
# Quick sweep — first 10 particles (CI-fast).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_NAMES[:10], ids=lambda n: n)
def test_particle_quick(name: str) -> None:
    """Single test per particle: χ-well + energy survival."""
    p = PARTICLES[name]
    m = measure_particle(name, _preset_for(p.field_level))

    if p.mass_ratio > 0.01:
        assert m.chi_min < m.chi0, f"{name}: no χ-well (chi_min={m.chi_min:.2f})"
    if p.stable:
        assert m.retention > 0.20, (
            f"{name}: stable particle lost too much energy ({m.retention:.1%})"
        )


# ---------------------------------------------------------------------------
# Full 69-particle sweep (slow — run with ``pytest -m slow``).
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", _ALL_NAMES, ids=lambda n: n)
def test_particle_full(name: str) -> None:
    """Full catalog: χ-well + energy survival."""
    p = PARTICLES[name]
    m = measure_particle(name, _preset_for(p.field_level))

    if p.mass_ratio > 0.01:
        assert m.chi_min < m.chi0
    if p.stable:
        assert m.retention > 0.20, f"{name}: {m.retention:.1%} retained"
