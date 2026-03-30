"""Task 3.5 — String tension measurement at multiple separations.

H₀: Tube energy does not grow with quark separation (σ = 0).
H₁: Tube energy grows with separation (σ > 0), indicating confinement.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 with full_physics preset (v17 Helmholtz SCV)
  ✓ Two solitons in different color channels at varying separations
  ✓ No Wilson loops or external QCD injected
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.config_presets import full_physics

from .conftest import BACKEND, GRID_TWO, STEPS

# Separations to probe.  Must fit inside GRID_TWO with room for the solitons.
_SEPS = [10, 14, 18, 22]


def _tube_energy(sim: Simulation, sep: int, axis: int = 0) -> float:
    """Integrated χ-depression below χ₀ along the midline between quarks.

    Sums (χ₀ − χ) for cells in the tube segment (excluding 3 cells near each
    quark).  Higher values mean a deeper / wider tube.
    """
    xp = np
    chi = xp.asarray(sim.chi)
    chi0 = sim.config.chi0
    half = sim.config.grid_size // 2
    lo = half - sep // 2 + 3
    hi = half + sep // 2 - 3
    if lo >= hi:
        return 0.0

    idx = [half] * 3
    total = 0.0
    for x in range(lo, hi + 1):
        idx[axis] = x
        val = float(chi[tuple(idx)])
        if val < chi0:
            total += chi0 - val
    return total


def _make_pair(sep: int) -> Simulation:
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


class TestStringTension:
    """Tube energy between colored quarks should grow with separation."""

    def test_tube_energy_increases(self) -> None:
        """Tube integral should be monotonically increasing with separation."""
        energies: list[float] = []
        for sep in _SEPS:
            sim = _make_pair(sep)
            sim.run(steps=STEPS)
            energies.append(_tube_energy(sim, sep))

        # Every increase should be strictly positive.
        for i in range(1, len(energies)):
            assert energies[i] > energies[i - 1], (
                f"Tube energy did not increase: sep={_SEPS[i]} gave "
                f"E_tube={energies[i]:.4f} ≤ {energies[i - 1]:.4f} at "
                f"sep={_SEPS[i - 1]}.  Full: {list(zip(_SEPS, energies, strict=True))}"
            )

    def test_positive_string_tension(self) -> None:
        """Linear fit σ = dE_tube/d(sep) should be positive."""
        energies: list[float] = []
        for sep in _SEPS:
            sim = _make_pair(sep)
            sim.run(steps=STEPS)
            energies.append(_tube_energy(sim, sep))

        seps_arr = np.array(_SEPS, dtype=float)
        e_arr = np.array(energies, dtype=float)
        slope, _intercept = np.polyfit(seps_arr, e_arr, 1)
        assert slope > 0, (
            f"σ = {slope:.6f} — expected positive string tension.  "
            f"Data: {list(zip(_SEPS, energies, strict=True))}"
        )
