"""Task 4.1 — Hydrogen Atom (Electron Bound to Proton).

Place a proton (phase = π, positive charge) at the center and an electron
(phase = 0, negative charge) nearby.  Opposite phases create destructive
interference → lower energy → attraction (Coulomb from phase).

After evolution the electron should remain bound (localized near the proton)
rather than dispersing to the boundary.
"""

from __future__ import annotations

import numpy as np

from lfm.config_presets import gravity_em
from lfm.simulation import Simulation

from .conftest import BACKEND, GRID_TWO, STEPS

_N = GRID_TWO  # 48


def _peak_radius(sim: Simulation, center: tuple[int, int, int]) -> float:
    """Distance from *center* to the peak |Ψ|² location."""
    psi_sq: np.ndarray = sim.psi_real**2
    pi = sim.psi_imag
    if pi is not None:
        psi_sq = psi_sq + pi**2
    if psi_sq.ndim == 4:
        psi_sq = psi_sq.sum(axis=0)

    peak_idx = np.unravel_index(int(np.argmax(psi_sq)), psi_sq.shape)
    dist = np.sqrt(sum((p - c) ** 2 for p, c in zip(peak_idx, center, strict=True)))
    return float(dist)


def _energy_within_radius(sim: Simulation, center: tuple[int, int, int], r: float) -> float:
    """Fraction of total |Ψ|² within radius *r* of *center*."""
    psi_sq: np.ndarray = sim.psi_real**2
    pi = sim.psi_imag
    if pi is not None:
        psi_sq = psi_sq + pi**2
    if psi_sq.ndim == 4:
        psi_sq = psi_sq.sum(axis=0)

    xs = np.arange(_N, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    mask = dist <= r
    total = float(psi_sq.sum())
    if total < 1e-30:
        return 0.0
    return float(psi_sq[mask].sum()) / total


class TestHydrogenAtom:
    """Electron binds to proton via phase-interference attraction."""

    def test_electron_stays_bound(self) -> None:
        """After evolution, most |Ψ|² is still near the proton."""
        cfg = gravity_em(grid_size=_N)
        sim = Simulation(cfg, backend=BACKEND)

        mid = _N // 2
        proton_pos = (mid, mid, mid)
        electron_pos = (mid, mid, mid + 5)

        # Proton: heavy, positive charge (phase = π).
        sim.place_soliton(proton_pos, amplitude=6.0, phase=np.pi)
        # Electron: lighter, negative charge (phase = 0).
        sim.place_soliton(electron_pos, amplitude=2.0, phase=0.0)
        sim.equilibrate()

        # Measure initial energy concentration near proton.
        frac_initial = _energy_within_radius(sim, proton_pos, r=10.0)

        sim.run(steps=STEPS)

        frac_final = _energy_within_radius(sim, proton_pos, r=10.0)

        # Energy should NOT have dispersed to infinity.
        # At minimum, ≥30% should remain within 10 cells of the proton.
        assert frac_final > 0.30, (
            f"Electron dispersed: {frac_final:.1%} within r=10 (initial: {frac_initial:.1%})"
        )

    def test_binding_vs_free_electron(self) -> None:
        """An electron near a proton stays more concentrated than a free one."""
        mid = _N // 2
        proton_pos = (mid, mid, mid)
        electron_pos = (mid, mid, mid + 5)

        # --- Bound system ---
        cfg_b = gravity_em(grid_size=_N)
        sim_b = Simulation(cfg_b, backend=BACKEND)
        sim_b.place_soliton(proton_pos, amplitude=6.0, phase=np.pi)
        sim_b.place_soliton(electron_pos, amplitude=2.0, phase=0.0)
        sim_b.equilibrate()
        sim_b.run(steps=STEPS)
        frac_bound = _energy_within_radius(sim_b, proton_pos, r=10.0)

        # --- Free electron (no proton) ---
        cfg_f = gravity_em(grid_size=_N)
        sim_f = Simulation(cfg_f, backend=BACKEND)
        sim_f.place_soliton(electron_pos, amplitude=2.0, phase=0.0)
        sim_f.equilibrate()
        sim_f.run(steps=STEPS)
        frac_free = _energy_within_radius(sim_f, proton_pos, r=10.0)

        # Bound electron should be more concentrated than free.
        assert frac_bound > frac_free, (
            f"Bound ({frac_bound:.1%}) should be more concentrated than free ({frac_free:.1%})"
        )
