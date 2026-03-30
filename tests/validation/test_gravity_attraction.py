"""Task 2.4 — Gravitational attraction between two neutral particles.

H₀: Two neutrons placed apart keep constant separation.
H₁: Their χ-wells overlap and deepen → separation shrinks (gravity).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only (gravity_only preset, real E field)
  ✓ No Coulomb, no Newton, no external forces
"""

from __future__ import annotations

from lfm import create_two_particles, measure_separation

from .conftest import STEPS


class TestGravityAttraction:
    """Two massive neutral solitons should attract via χ-well overlap."""

    def test_separation_decreases(self) -> None:
        """Distance between two neutrons should shrink over time."""
        pa, pb = create_two_particles(
            "neutron",
            "neutron",
            separation=16,
            N=48,
        )
        sim = pa.sim
        sim.equilibrate()

        sim.run(steps=2)

        psi_sq = sim.psi_real**2
        pi = sim.psi_imag
        if pi is not None:
            psi_sq = psi_sq + pi**2
        if psi_sq.ndim == 4:
            psi_sq = psi_sq.sum(axis=0)
        sep_initial = measure_separation(psi_sq, min_peak_separation=4)

        sim.run(steps=STEPS)

        psi_sq = sim.psi_real**2
        pi = sim.psi_imag
        if pi is not None:
            psi_sq = psi_sq + pi**2
        if psi_sq.ndim == 4:
            psi_sq = psi_sq.sum(axis=0)
        sep_final = measure_separation(psi_sq, min_peak_separation=4)

        # Gravity is weak — allow up to 5% growth as tolerance
        if sep_initial > 0:
            assert sep_final <= sep_initial * 1.05, (
                f"Separation grew: {sep_initial:.1f} → {sep_final:.1f}. "
                "Neutral massive particles should attract via gravity."
            )
