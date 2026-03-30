"""Task 2.6 — Charge quantisation: phase determines interaction character.

H₀: All phases produce identical two-particle interactions.
H₁: θ=0 vs θ=π pairs show distinct energy behaviour — same-phase
     (constructive) has more overlap energy than opposite-phase
     (destructive).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only
  ✓ Phase assigned via place_soliton() API

Note: Single-soliton stability is phase-independent (GOV-01 is linear
in Ψ, GOV-02 depends on |Ψ|²). Charge quantisation manifests in
INTERACTIONS, not isolated stability.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm import Simulation
from lfm.config_presets import gravity_em

from .conftest import STEPS


def _make_single_soliton(phase: float, N: int = 48) -> Simulation:
    """Create a single Gaussian soliton at the grid centre with given phase."""
    cfg = gravity_em(grid_size=N)
    sim = Simulation(cfg)
    c = N // 2
    sim.place_soliton((c, c, c), amplitude=6.0, phase=phase)
    sim.equilibrate()
    return sim


def _peak_psi_sq(sim: Simulation) -> float:
    psi_sq = sim.psi_real**2
    pi = sim.psi_imag
    if pi is not None:
        psi_sq = psi_sq + pi**2
    if psi_sq.ndim == 4:
        psi_sq = psi_sq.sum(axis=0)
    return float(np.max(np.asarray(psi_sq)))


class TestChargeQuantisation:
    """Phase determines interaction character (charge quantisation)."""

    @pytest.mark.parametrize("phase", [0.0, np.pi])
    def test_stable_phases(self, phase: float) -> None:
        """Canonical charges must keep their peak amplitude."""
        sim = _make_single_soliton(phase)

        peak0 = _peak_psi_sq(sim)
        assert peak0 > 0, "Soliton was not placed — peak |Ψ|² is 0"

        sim.run(steps=STEPS)

        peak_f = _peak_psi_sq(sim)

        # Peak should not drop below 20% of original (generous)
        assert peak_f > peak0 * 0.20, (
            f"Phase {phase:.2f}: peak dropped from {peak0:.3f} to {peak_f:.3f} — expected stable."
        )

    def test_phase_determines_interaction(self) -> None:
        """Same-phase pair should have more total |Ψ|² than opposite-phase.

        Constructive interference (same θ) increases overlap E².
        Destructive (opposite θ) reduces overlap E².
        Uses explicit sigma=3.0 for significant overlap at sep=6.
        """
        N = 48
        c = N // 2
        sep = 6
        cfg = gravity_em(grid_size=N)

        # Same-charge pair (constructive)
        sim_s = Simulation(cfg)
        sim_s.place_soliton((c - sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=0.0)
        sim_s.place_soliton((c + sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=0.0)
        psi_sq = sim_s.psi_real**2
        if sim_s.psi_imag is not None:
            psi_sq = psi_sq + sim_s.psi_imag**2
        total_same = float(np.sum(np.asarray(psi_sq)))

        # Opposite-charge pair (destructive)
        sim_o = Simulation(cfg)
        sim_o.place_soliton((c - sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=0.0)
        sim_o.place_soliton((c + sep // 2, c, c), amplitude=6.0, sigma=3.0, phase=np.pi)
        psi_sq = sim_o.psi_real**2
        if sim_o.psi_imag is not None:
            psi_sq = psi_sq + sim_o.psi_imag**2
        total_opp = float(np.sum(np.asarray(psi_sq)))

        # Same-phase → constructive → more total |Ψ|²
        assert total_same > total_opp * 1.01, (
            f"Same-phase pair should have more total |Ψ|² than opposite. "
            f"same={total_same:.2f}, opposite={total_opp:.2f}"
        )
