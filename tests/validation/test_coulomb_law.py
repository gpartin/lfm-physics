"""Historical overlap/chi-depth controls, not Coulomb validation.

Neither chi_min nor the initial norm is a force or total interaction energy.
The filename is retained for compatibility; see docs/coulomb_audit.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from lfm import create_two_particles


class TestHistoricalPhaseControls:
    """Preparation-dependent field diagnostics only."""

    @pytest.mark.slow
    def test_closer_pair_has_deeper_chi(self) -> None:
        """Closer prepared pairs have deeper chi; no energy inference."""
        separations = [8, 12, 18, 24]
        chi_mins: list[float] = []

        for sep in separations:
            pa, pb = create_two_particles(
                "electron",
                "electron",
                separation=sep,
                N=48,
            )
            sim = pa.sim
            sim.equilibrate()
            sim.run(steps=2)
            chi_mins.append(float(np.min(np.asarray(sim.chi))))

        # Historical depth criterion only; not a potential or force.
        assert chi_mins[0] < chi_mins[-1], (
            f"Expected deeper chi for closer same-phase pair. "
            f"chi_mins={[f'{c:.4f}' for c in chi_mins]}, "
            f"seps={separations}"
        )

    def test_opposite_phase_has_lower_initial_norm(self) -> None:
        """Opposite-phase pair should have less total |Ψ|² (destructive).

        Uses explicit sigma=3.0 for significant overlap at sep=6.
        """
        from lfm import Simulation
        from lfm.config_presets import gravity_em

        N = 48
        c = N // 2
        sep = 6
        cfg = gravity_em(grid_size=N)

        # Same-phase (constructive → more E²)
        sim_s = Simulation(cfg)
        sim_s.place_soliton((c - sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=0.0)
        sim_s.place_soliton((c + sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=0.0)
        psi_sq = sim_s.psi_real**2
        if sim_s.psi_imag is not None:
            psi_sq = psi_sq + sim_s.psi_imag**2
        total_same = float(np.sum(np.asarray(psi_sq)))

        # Opposite-phase initial destructive overlap
        sim_o = Simulation(cfg)
        sim_o.place_soliton((c - sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=0.0)
        sim_o.place_soliton((c + sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=np.pi)
        psi_sq = sim_o.psi_real**2
        if sim_o.psi_imag is not None:
            psi_sq = psi_sq + sim_o.psi_imag**2
        total_opp = float(np.sum(np.asarray(psi_sq)))

        assert total_same > total_opp * 1.01, (
            f"Same-phase pair should have more total |Ψ|² than opposite. "
            f"same={total_same:.2f}, opposite={total_opp:.2f}"
        )
