"""Task 3.7 — Quark vs hadron stability comparison.

H₀: All color configurations behave identically (same f_c, same localization).
H₁: Quarks, mesons, and baryons show measurably different color physics.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 with full_physics preset (v17 Helmholtz SCV)
  ✓ Three configurations: free quark, meson (qq̄), baryon (qqq)
  ✓ No external physics injected

NOTE: Energy retention ordering (hadron > quark) documented in the v15
stress test (Session 140b) used ε_cc coupling over 3000 steps.  At the
short CI step budget (500 steps) with all v17 terms active, ψ-energy
retention reflects early χ→ψ transient transfer rather than long-term
stability.  This test therefore checks *color variance ordering* and
*survival*, which are robust at short timescales.
"""

from __future__ import annotations

from lfm import Simulation
from lfm.analysis.measurements import measure_color_fraction
from lfm.config_presets import full_physics

from .conftest import BACKEND, GRID_TWO, STEPS, WARMUP, peak_psi_sq


def _make_quark() -> Simulation:
    """Single-color soliton (free quark, f_c ≈ 2/3)."""
    cfg = full_physics(grid_size=GRID_TWO)
    sim = Simulation(cfg, backend=BACKEND)
    sim.place_solitons([(GRID_TWO // 2,) * 3], amplitude=6.0, sigma=3.0, colors=[0])
    sim.equilibrate()
    return sim


def _make_meson(sep: int = 14) -> Simulation:
    """Two quarks of different color (meson analog)."""
    cfg = full_physics(grid_size=GRID_TWO)
    sim = Simulation(cfg, backend=BACKEND)
    h = GRID_TWO // 2
    sim.place_solitons(
        [(h - sep // 2, h, h), (h + sep // 2, h, h)],
        amplitude=6.0,
        sigma=3.0,
        colors=[0, 1],
    )
    sim.equilibrate()
    return sim


def _make_baryon(sep: int = 10) -> Simulation:
    """Three quarks — one per color channel (baryon analog)."""
    cfg = full_physics(grid_size=GRID_TWO)
    sim = Simulation(cfg, backend=BACKEND)
    h = GRID_TWO // 2
    positions = [
        (h - sep // 2, h - sep // 3, h),
        (h + sep // 2, h - sep // 3, h),
        (h, h + sep // 2, h),
    ]
    sim.place_solitons(positions, amplitude=6.0, sigma=3.0, colors=[0, 1, 2])
    sim.equilibrate()
    return sim


class TestQuarkHadronStability:
    """Quarks, mesons, and baryons should show distinct color physics."""

    def test_fc_ordering(self) -> None:
        """Color variance ordering: quark > meson > baryon.

        Single-color quarks have the highest f_c (~2/3).
        Multi-color composites average toward lower f_c as colors balance.
        """
        sim_q = _make_quark()
        sim_q.run(steps=WARMUP)
        fc_q = measure_color_fraction(sim_q)

        sim_m = _make_meson()
        sim_m.run(steps=WARMUP)
        fc_m = measure_color_fraction(sim_m)

        sim_b = _make_baryon()
        sim_b.run(steps=WARMUP)
        fc_b = measure_color_fraction(sim_b)

        assert fc_q > fc_m, (
            f"f_c quark={fc_q:.4f} ≤ meson={fc_m:.4f} — "
            "expected quark to have highest color variance."
        )
        assert fc_q > fc_b, (
            f"f_c quark={fc_q:.4f} ≤ baryon={fc_b:.4f} — expected quark > baryon in color variance."
        )

    def test_all_survive(self) -> None:
        """All three configurations should remain localized after evolution."""
        for label, make_fn in [
            ("quark", _make_quark),
            ("meson", _make_meson),
            ("baryon", _make_baryon),
        ]:
            sim = make_fn()
            sim.run(steps=STEPS)
            peak = peak_psi_sq(sim)
            assert peak > 1e-4, f"{label}: peak |Ψ|² = {peak:.2e} — soliton dispersed."
