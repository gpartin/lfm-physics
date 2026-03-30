"""Task 4.7 — Dark Matter Halo (χ Memory).

In LFM, when matter evacuates a region the χ-well persists because χ is a
wave field with its own momentum (GOV-02 is second-order in time).  The
reduced-χ region acts as a gravitational phantom — "dark matter."

Test: place soliton → evolve → remove all Ψ → evolve more → check χ-well
still partially present.
"""

from __future__ import annotations

import numpy as np

from lfm.config_presets import gravity_only
from lfm.simulation import Simulation

from .conftest import BACKEND, GRID, STEPS


class TestDarkMatterHalo:
    """χ-well persists after matter is removed — dark matter analog."""

    def test_chi_well_persists(self) -> None:
        """After zeroing Ψ, the χ-well does not immediately vanish."""
        cfg = gravity_only(grid_size=GRID)
        sim = Simulation(cfg, backend=BACKEND)

        mid = GRID // 2
        sim.place_soliton((mid, mid, mid), amplitude=6.0)
        sim.equilibrate()

        # Evolve so χ dynamics are active (not just the static Poisson soln).
        sim.run(steps=STEPS)
        chi_before_removal = float(sim.chi.min())

        # Confirm a well exists.
        assert chi_before_removal < cfg.chi0 - 0.1, (
            f"No χ-well formed: chi_min={chi_before_removal:.3f}"
        )

        # Remove all matter.
        sim.set_psi_real(np.zeros((GRID, GRID, GRID), dtype=np.float32))

        # Evolve more — χ should still show a residual well.
        sim.run(steps=STEPS // 2)
        chi_after = float(sim.chi.min())

        # χ should still be below vacuum (memory persists).
        assert chi_after < cfg.chi0 - 0.05, (
            f"χ-well vanished too fast: chi_min={chi_after:.3f}, χ₀={cfg.chi0:.1f}"
        )

    def test_chi_recovery_direction(self) -> None:
        """After matter removal, χ_min should be moving upward (recovering)."""
        cfg = gravity_only(grid_size=GRID)
        sim = Simulation(cfg, backend=BACKEND)

        mid = GRID // 2
        sim.place_soliton((mid, mid, mid), amplitude=6.0)
        sim.equilibrate()
        sim.run(steps=STEPS)

        chi_with_matter = float(sim.chi.min())

        # Remove matter.
        sim.set_psi_real(np.zeros((GRID, GRID, GRID), dtype=np.float32))

        # Evolve a short time — well persists.
        sim.run(steps=STEPS // 4)

        # Evolve longer — well starts recovering.
        sim.run(steps=STEPS)
        chi_long = float(sim.chi.min())

        # After enough time, χ_min should be rising toward χ₀.
        # (It may overshoot and oscillate, so check it's not deeper.)
        assert chi_long > chi_with_matter, (
            f"χ not recovering: chi_with_matter={chi_with_matter:.3f}, chi_long={chi_long:.3f}"
        )
