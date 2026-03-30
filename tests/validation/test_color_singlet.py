"""Task 3.3 — Color singlet (three equal colors).

H₀: Singlet has same f_c as a single-color quark.
H₁: Singlet has f_c ≈ 0 (balanced colors cancel color variance).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 with full_physics preset
  ✓ All 3 color channels excited equally
  ✓ No external QCD injected
"""

from __future__ import annotations

from lfm import Simulation
from lfm.analysis.measurements import measure_color_fraction
from lfm.config_presets import full_physics

from .conftest import BACKEND, GRID, WARMUP


class TestColorSinglet:
    """A balanced 3-color soliton should have f_c ≈ 0."""

    def _make_singlet(self) -> Simulation:
        cfg = full_physics(grid_size=GRID)
        sim = Simulation(cfg, backend=BACKEND)
        center = (GRID // 2,) * 3
        # Place same-amplitude Gaussian in all 3 colors at the same spot
        sim.place_solitons(
            [center, center, center],
            amplitude=6.0,
            sigma=3.0,
            colors=[0, 1, 2],
        )
        sim.equilibrate()
        return sim

    def test_fc_near_zero(self) -> None:
        """f_c should be near-zero for a balanced-color state."""
        sim = self._make_singlet()
        sim.run(steps=WARMUP)
        fc = measure_color_fraction(sim)
        assert abs(fc) < 0.05, f"f_c = {fc:.4f} — expected ≈ 0 for balanced singlet."

    def test_chi_well_from_gravity(self) -> None:
        """Singlet should still form a χ-well (from gravity/energy density)."""
        sim = self._make_singlet()
        sim.run(steps=WARMUP)
        chi_min = float(sim.chi.min())
        assert chi_min < sim.config.chi0, (
            f"chi_min={chi_min:.2f} ≥ chi0={sim.config.chi0} — "
            "singlet energy should still source a χ-well."
        )

    def test_singlet_vs_quark_fc(self) -> None:
        """Singlet f_c should be much smaller than single-color f_c."""
        cfg = full_physics(grid_size=GRID)
        center = (GRID // 2,) * 3

        # Quark (single color)
        sim_q = Simulation(cfg, backend=BACKEND)
        sim_q.place_solitons([center], amplitude=6.0, sigma=3.0, colors=[0])
        sim_q.equilibrate()
        sim_q.run(steps=WARMUP)
        fc_q = measure_color_fraction(sim_q)

        # Singlet (balanced)
        sim_s = Simulation(cfg, backend=BACKEND)
        sim_s.place_solitons(
            [center, center, center],
            amplitude=6.0,
            sigma=3.0,
            colors=[0, 1, 2],
        )
        sim_s.equilibrate()
        sim_s.run(steps=WARMUP)
        fc_s = measure_color_fraction(sim_s)

        assert fc_q > fc_s + 0.1, (
            f"Quark f_c={fc_q:.4f}, Singlet f_c={fc_s:.4f} — "
            "quark should have substantially higher color variance."
        )
