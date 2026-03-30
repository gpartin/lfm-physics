"""Task 4.3 — Pair Creation via Parametric Resonance.

When χ oscillates at Ω = 2χ₀, GOV-01 becomes a Mathieu equation and tiny Ψ
perturbations grow exponentially.  This tests the LFM matter-creation mechanism.

We compare |Ψ|² growth under resonant driving (Ω = 2χ₀) versus a static-χ
control.  The resonant case should show orders-of-magnitude more growth.
"""

from __future__ import annotations

import numpy as np

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.simulation import Simulation

from .conftest import BACKEND, GRID

_N = GRID
_CHI0 = 19.0
_DRIVE_AMP = 2.0  # amplitude of χ oscillation
_STEPS = 1000


def _psi_norm(sim: Simulation) -> float:
    """Sum of |Ψ|² over the grid."""
    psi_sq: np.ndarray = sim.psi_real**2
    pi = sim.psi_imag
    if pi is not None:
        psi_sq = psi_sq + pi**2
    return float(psi_sq.sum())


def _run_driven_experiment(omega: float) -> float:
    """Drive χ at angular frequency *omega* and return final |Ψ|² norm."""
    cfg = SimulationConfig(
        grid_size=_N,
        field_level=FieldLevel.REAL,
        boundary_type=BoundaryType.PERIODIC,
        chi0=_CHI0,
        dt=0.02,
        random_seed=42,
    )
    sim = Simulation(cfg, backend=BACKEND)

    # Seed tiny random noise (machine-epsilon level is too small for
    # short CI runs; use 1e-6 so growth is observable in 1000 steps).
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1e-6, (_N, _N, _N)).astype(np.float32)
    sim.set_psi_real(noise)

    def chi_forcing(t: float) -> np.float32:
        return np.float32(_CHI0 + _DRIVE_AMP * np.sin(omega * t))

    sim.run_driven(_STEPS, chi_forcing=chi_forcing)
    return _psi_norm(sim)


class TestPairCreation:
    """Parametric resonance matter creation from vacuum noise."""

    def test_resonant_growth(self) -> None:
        """Resonant Ω = 2χ₀ drives |Ψ|² to much higher values than control."""
        omega_res = 2 * _CHI0  # = 38
        psi_sq_res = _run_driven_experiment(omega_res)

        # Static control: no oscillation (ω = 0 is just constant chi)
        # Equivalent to normal evolution with uniform χ₀
        psi_sq_static = _run_driven_experiment(0.0)

        # Resonant growth should be at least 10× the static case.
        ratio = psi_sq_res / max(psi_sq_static, 1e-30)
        assert ratio > 10, (
            f"Resonant |Ψ|²={psi_sq_res:.4e}, Static |Ψ|²={psi_sq_static:.4e}, "
            f"ratio={ratio:.1f} — expected > 10"
        )

    def test_off_resonance_weaker(self) -> None:
        """Driving at Ω = χ₀ (off-resonance) produces less growth than 2χ₀."""
        omega_res = 2 * _CHI0
        omega_off = _CHI0  # half the resonant frequency

        psi_sq_res = _run_driven_experiment(omega_res)
        psi_sq_off = _run_driven_experiment(omega_off)

        assert psi_sq_res > psi_sq_off, (
            f"Resonant |Ψ|²={psi_sq_res:.4e} should exceed off-resonance |Ψ|²={psi_sq_off:.4e}"
        )
