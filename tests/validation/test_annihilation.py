"""Task 2.3 — Electron-positron annihilation (close approach).

H₀: Closely placed e⁻ + e⁺ keep two distinct peaks.
H₁: They merge and produce outgoing radiation (peaks vanish, energy
     transitions from localised to diffuse).

LFM-ONLY:
  ✓ GOV-01 + GOV-02 only
  ✓ Phase cancellation (θ=0 + θ=π → destructive)
"""

from __future__ import annotations

import numpy as np

from lfm import create_two_particles, measure_separation

from .conftest import STEPS


class TestAnnihilation:
    """Close e⁻ + e⁺ pair should merge and radiate."""

    def test_peaks_approach(self) -> None:
        """Opposite-charge solitons at close range should converge."""
        pa, pb = create_two_particles(
            "electron",
            "positron",
            separation=8,
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
        sep_initial = measure_separation(psi_sq, min_peak_separation=3)

        sim.run(steps=STEPS)
        psi_sq = sim.psi_real**2
        pi = sim.psi_imag
        if pi is not None:
            psi_sq = psi_sq + pi**2
        if psi_sq.ndim == 4:
            psi_sq = psi_sq.sum(axis=0)
        sep_final = measure_separation(psi_sq, min_peak_separation=3)

        # Either peaks merged (sep_final==0) or they got closer.
        if sep_initial > 0:
            assert sep_final <= sep_initial * 1.05, (
                f"Separation grew: {sep_initial:.1f} → {sep_final:.1f}. "
                "Close opposite charges should attract, not repel."
            )

    def test_energy_spreads(self) -> None:
        """Localised energy should drop — radiation carries energy outward."""
        pa, pb = create_two_particles(
            "electron",
            "positron",
            separation=8,
            N=48,
        )
        sim = pa.sim
        sim.equilibrate()

        psi_sq = sim.psi_real**2
        pi = sim.psi_imag
        if pi is not None:
            psi_sq = psi_sq + pi**2
        if psi_sq.ndim == 4:
            psi_sq = psi_sq.sum(axis=0)
        # Energy in the central region (within 10 cells of centre)
        N = psi_sq.shape[0]
        c = N // 2
        r = 10
        sl = slice(max(c - r, 0), min(c + r + 1, N))
        centre_energy_0 = float(np.sum(np.asarray(psi_sq)[sl, sl, sl]))

        sim.run(steps=STEPS)

        psi_sq = sim.psi_real**2
        pi = sim.psi_imag
        if pi is not None:
            psi_sq = psi_sq + pi**2
        if psi_sq.ndim == 4:
            psi_sq = psi_sq.sum(axis=0)
        centre_energy_f = float(np.sum(np.asarray(psi_sq)[sl, sl, sl]))

        assert centre_energy_f < centre_energy_0 * 0.90, (
            f"Central energy did not drop enough: {centre_energy_0:.2f} → "
            f"{centre_energy_f:.2f}. Expected ≥10% outward radiation."
        )
