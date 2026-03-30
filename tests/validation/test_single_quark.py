"""Task 3.2 — Single quark in isolation (color physics).

H₀: Quark behaves identically to a colorless soliton (f_c ≈ 0).
H₁: Quark has measurably non-zero f_c and forms a χ-well.

LFM-ONLY:
  ✓ GOV-01 + GOV-02 with full_physics preset (all four forces)
  ✓ Single color channel excited (Ψ_0 only)
  ✓ No external QCD injected
"""

from __future__ import annotations

from lfm import Simulation
from lfm.analysis.measurements import measure_color_fraction
from lfm.config_presets import full_physics

from .conftest import BACKEND, GRID, STEPS, WARMUP, peak_psi_sq


class TestSingleQuark:
    """A single-color soliton should show non-zero color variance."""

    def _make_quark(self) -> Simulation:
        cfg = full_physics(grid_size=GRID)
        sim = Simulation(cfg, backend=BACKEND)
        center = (GRID // 2,) * 3
        sim.place_solitons([center], amplitude=6.0, sigma=3.0, colors=[0])
        sim.equilibrate()
        return sim

    def test_initial_color_fraction(self) -> None:
        """f_c should be near 2/3 right after placement (single color)."""
        sim = self._make_quark()
        sim.run(steps=WARMUP)
        fc = measure_color_fraction(sim)
        assert fc > 0.3, (
            f"f_c = {fc:.4f} — expected > 0.3 for a single-color soliton shortly after placement."
        )

    def test_chi_well_forms(self) -> None:
        """χ should dip below χ₀ where the quark sits."""
        sim = self._make_quark()
        sim.equilibrate()
        sim.run(steps=WARMUP)
        chi_min = float(sim.chi.min())
        assert chi_min < sim.config.chi0, (
            f"chi_min={chi_min:.2f} ≥ chi0={sim.config.chi0} — "
            "quark should create a gravitational well."
        )

    def test_quark_survives(self) -> None:
        """Quark soliton should remain localized after evolution."""
        sim = self._make_quark()
        sim.equilibrate()
        sim.run(steps=STEPS)
        peak = peak_psi_sq(sim)
        assert peak > 1e-4, f"peak |Ψ|² = {peak:.2e} — quark dispersed."
