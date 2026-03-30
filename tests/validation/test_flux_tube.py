"""Task 3.4 — Quark-antiquark flux tube (meson analog).

H₀: No χ depression exists between separated colored quarks.
H₁: χ between two differently-colored quarks is depressed below χ₀.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 with full_physics preset (v17 Helmholtz SCV)
  ✓ Two solitons in *different* color channels
  ✓ No Wilson loops or external QCD injected
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.config_presets import full_physics

from .conftest import BACKEND, GRID_TWO, STEPS


def _tube_chi(sim: Simulation, sep: int, axis: int = 0) -> float:
    """Mean χ along the segment between the two quarks.

    Takes the centre-line between (half-sep/2) and (half+sep/2) along
    *axis*, excluding the 3 cells closest to each quark.
    """
    chi = np.asarray(sim.chi)
    half = sim.config.grid_size // 2
    lo = half - sep // 2 + 3  # skip near-quark region
    hi = half + sep // 2 - 3
    if lo >= hi:
        return float(chi.min())  # quarks too close for a tube segment
    idx = [half] * 3
    vals = []
    for x in range(lo, hi + 1):
        idx[axis] = x
        vals.append(float(chi[tuple(idx)]))
    return float(np.mean(vals))


class TestFluxTube:
    """Two quarks of different color should show a χ-depression tube."""

    def _make_pair(self, sep: int = 20) -> Simulation:
        cfg = full_physics(grid_size=GRID_TWO)
        sim = Simulation(cfg, backend=BACKEND)
        half = GRID_TWO // 2
        sim.place_solitons(
            [(half - sep // 2, half, half), (half + sep // 2, half, half)],
            amplitude=6.0,
            sigma=3.0,
            colors=[0, 1],
        )
        sim.equilibrate()
        return sim

    def test_tube_below_vacuum(self) -> None:
        """Mean χ between the quarks should be below χ₀."""
        sim = self._make_pair(sep=20)
        sim.run(steps=STEPS)
        chi0 = sim.config.chi0
        tube_chi = _tube_chi(sim, sep=20, axis=0)
        assert tube_chi < chi0 - 0.01, (
            f"Tube χ = {tube_chi:.4f}, χ₀ = {chi0} — expected χ depression between quarks."
        )

    def test_wells_at_quarks(self) -> None:
        """Each quark should sit in its own χ-well."""
        sim = self._make_pair(sep=20)
        sim.run(steps=STEPS)
        chi_min = float(sim.chi.min())
        assert chi_min < sim.config.chi0 - 0.5, (
            f"chi_min={chi_min:.2f} — expected deep wells at quark sites."
        )
