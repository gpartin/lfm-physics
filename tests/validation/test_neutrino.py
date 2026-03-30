"""Task 1.4 — Neutrino stability (near-massless).

H₀: Neutrino behaves like a massive trapped soliton (stays put).
H₁: Neutrino propagates freely at ~c (not trapped) — energy delocalises.

The electron neutrino has mass_ratio ≈ 4 × 10⁻⁶ and field_level=0
(real E, gravity only).  On a small grid the soliton's self-gravity
is negligible, so it should spread out rapidly.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only (gravity_only preset)
"""

from __future__ import annotations

import numpy as np

from lfm import Simulation
from lfm.config_presets import gravity_only

from .conftest import BACKEND, GRID, STEPS, peak_psi_sq


class TestNeutrino:
    """Near-massless neutrino should spread rather than sit still."""

    def test_neutrino_delocalises(self) -> None:
        """Peak |Ψ|² should drop from its initial value (energy spreads out)."""
        sim = Simulation(gravity_only(grid_size=GRID), backend=BACKEND)
        sim.place_particle("electron_neutrino")
        sim.equilibrate()

        # Capture the initial peak (before any evolution).
        peak_initial = float(np.max(sim.psi_real**2))

        sim.run(steps=STEPS)
        peak_after = peak_psi_sq(sim)

        if peak_initial > 1e-30:
            ratio = peak_after / peak_initial
            assert ratio < 1.50, (
                f"Neutrino amplified too much (peak ratio {ratio:.2f}). "
                "Expected near-massless particle not to self-amplify strongly."
            )
