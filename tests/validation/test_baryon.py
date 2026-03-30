"""Task 3.6 — Three-quark bound state (baryon analog).

H₀: Three differently-colored quarks fly apart (no binding).
H₁: Three quarks form a bound cluster with f_c → 0 (color singlet).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 with full_physics preset (v17 Helmholtz SCV)
  ✓ Three solitons — one per color channel [0, 1, 2]
  ✓ No external QCD or confinement force injected
"""

from __future__ import annotations

from lfm import Simulation
from lfm.analysis.measurements import measure_color_fraction
from lfm.config_presets import full_physics

from .conftest import BACKEND, GRID_TWO, STEPS, WARMUP, peak_psi_sq


def _make_baryon(sep: int = 10) -> Simulation:
    """Place three quarks in a triangle — one of each color."""
    cfg = full_physics(grid_size=GRID_TWO)
    sim = Simulation(cfg, backend=BACKEND)
    h = GRID_TWO // 2
    # Equilateral-ish triangle in the xy-plane, centred on the grid.
    positions = [
        (h - sep // 2, h - sep // 3, h),
        (h + sep // 2, h - sep // 3, h),
        (h, h + sep // 2, h),
    ]
    sim.place_solitons(positions, amplitude=6.0, sigma=3.0, colors=[0, 1, 2])
    sim.equilibrate()
    return sim


class TestBaryon:
    """Three quarks (r/g/b) should bind into a color-singlet cluster."""

    def test_baryon_survives(self) -> None:
        """Combined energy should remain localized after evolution."""
        sim = _make_baryon()
        sim.run(steps=STEPS)
        peak = peak_psi_sq(sim)
        assert peak > 1e-4, f"peak |Ψ|² = {peak:.2e} — baryon soliton dispersed."

    def test_fc_lower_than_quark(self) -> None:
        """Global f_c for the baryon should be lower than for a single quark.

        A co-located singlet has f_c = 0.  With quarks at separate sites
        f_c reflects the local single-color dominance at each vertex, so
        the reduction vs a lone quark (f_c ≈ 2/3) is modest but real.
        """
        sim = _make_baryon()
        sim.run(steps=WARMUP)
        fc = measure_color_fraction(sim)
        # Three equal-energy, separate-color solitons give a global f_c
        # that is measurably below 2/3 due to overlapping tails.
        assert fc < 0.65, (
            f"f_c = {fc:.4f} — expected < 0.65 for a three-color baryon (single quark gives ~0.67)."
        )

    def test_chi_wells_form(self) -> None:
        """Each quark vertex should create a χ-well."""
        sim = _make_baryon()
        sim.run(steps=WARMUP)
        chi_min = float(sim.chi.min())
        assert chi_min < sim.config.chi0 - 0.3, (
            f"chi_min={chi_min:.2f} — expected wells at quark vertices."
        )
