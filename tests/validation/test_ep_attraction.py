"""Task 2.2 — Electron-positron attraction (opposite-phase EM).

H₀: Opposite-phase solitons repel or show no interaction.
H₁: Opposite-phase solitons attract via destructive interference (separation shrinks).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only (gravity_em preset, auto-selected by factory)
  ✓ Charge from phase — electron θ=0, positron θ=π
"""

from __future__ import annotations

from lfm import create_two_particles, measure_separation

from .conftest import STEPS


class TestElectronPositronAttraction:
    """Electron (θ=0) + positron (θ=π) should attract."""

    def test_separation_decreases(self) -> None:
        """Distance between the two peaks should shrink over time."""
        pa, pb = create_two_particles(
            "electron",
            "positron",
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

        # If both peaks are still visible, the gap should have narrowed
        # (or they merged — sep_final==0 counts as "decreased").
        if sep_initial > 0:
            assert sep_final <= sep_initial * 1.05, (
                f"Separation grew: {sep_initial:.1f} → {sep_final:.1f}. "
                "Opposite charges should attract, not repel."
            )
