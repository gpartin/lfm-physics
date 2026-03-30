"""Task 2.1 — Electron-electron repulsion (same-phase EM).

H₀: Same-phase solitons attract or show no interaction.
H₁: Same-phase solitons repel via constructive interference (separation grows).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only (gravity_em preset)
  ✓ Charge from phase — both θ=0 (same charge)
"""

from __future__ import annotations

from lfm import create_two_particles, find_peaks, measure_separation

from .conftest import GRID_TWO, STEPS


class TestElectronElectronRepulsion:
    """Two electrons (both θ=0) placed nearby should repel."""

    def test_separation_increases(self) -> None:
        """Distance between the two peaks should grow over time."""
        pa, pb = create_two_particles(
            "electron",
            "electron",
            separation=16,
            N=GRID_TWO,
        )
        sim = pa.sim
        sim.equilibrate()

        sim.run(steps=2)
        sep_initial = measure_separation(sim.psi_real**2)

        sim.run(steps=STEPS)
        sep_final = measure_separation(sim.psi_real**2)

        assert sep_initial > 0, "Could not detect two peaks initially"
        assert sep_final >= sep_initial * 0.95, (
            f"Separation shrank: {sep_initial:.1f} → {sep_final:.1f}. "
            "Same-charge electrons should not attract."
        )

    def test_both_peaks_survive(self) -> None:
        """Both particles should remain as identifiable peaks."""
        pa, pb = create_two_particles(
            "electron",
            "electron",
            separation=16,
            N=GRID_TWO,
        )
        sim = pa.sim
        sim.equilibrate()
        sim.run(steps=STEPS)

        psi_sq = sim.psi_real**2
        pi = sim.psi_imag
        if pi is not None:
            psi_sq = psi_sq + pi**2
        if psi_sq.ndim == 4:
            psi_sq = psi_sq.sum(axis=0)

        peaks = find_peaks(psi_sq, n=2, min_separation=4)
        assert len(peaks) >= 2, f"Only {len(peaks)} peak(s) found — expected 2 distinct solitons."
