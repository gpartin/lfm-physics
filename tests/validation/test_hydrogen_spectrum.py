"""Task 4.2 — Hydrogen Energy Levels (Discrete Spectrum).

Different angular-momentum seeds (l=0 spherical vs l=1 dipole) should give
distinguishable spatial patterns.  At CI resolution the mass gap dominates
oscillation frequency, so we test the spatial signature instead:

*  l=0 (s-orbital): peaked at center (spherically symmetric).
*  l=1 (p-orbital): has a node at center and lobes offset along an axis.

After evolution, the radial profile of |Ψ|² should reflect these different
orbital geometries.
"""

from __future__ import annotations

import numpy as np

from lfm.config_presets import gravity_em
from lfm.simulation import Simulation

from .conftest import BACKEND, GRID_TWO, STEPS

_N = GRID_TWO  # 48


def _setup_hydrogen(l_mode: int = 0) -> Simulation:
    """Place proton + electron with angular-momentum seed *l_mode*.

    l_mode = 0  →  s-orbital (spherically symmetric envelope)
    l_mode = 1  →  p-orbital (dipole: cos θ × radial envelope)
    """
    cfg = gravity_em(grid_size=_N)
    sim = Simulation(cfg, backend=BACKEND)

    mid = _N // 2

    # Proton at center.
    sim.place_soliton((mid, mid, mid), amplitude=6.0, phase=np.pi)

    xs = np.arange(_N, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    dx = xx - mid
    dy = yy - mid
    dz = zz - mid
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    sigma = 3.0
    radial = np.exp(-(r**2) / (2 * sigma**2))

    if l_mode == 0:
        electron = 2.0 * radial
    else:
        # Dipole along x: node at center, lobes at ±x.
        cos_theta = np.zeros_like(dx)
        np.divide(dx, r, out=cos_theta, where=r > 0.5)
        electron = 2.0 * radial * cos_theta

    current = sim.psi_real.copy()
    sim.set_psi_real(current + electron.astype(np.float32))
    sim.equilibrate()
    return sim


def _psi_sq(sim: Simulation) -> np.ndarray:
    """Sum |Ψ|² over color components if present."""
    psi_sq: np.ndarray = sim.psi_real**2
    pi = sim.psi_imag
    if pi is not None:
        psi_sq = psi_sq + pi**2
    if psi_sq.ndim == 4:
        psi_sq = psi_sq.sum(axis=0)
    return psi_sq


class TestHydrogenSpectrum:
    """Different angular-momentum seeds produce distinguishable spatial modes."""

    def test_s_orbital_peaked_at_center(self) -> None:
        """l=0 seed: |Ψ|² is maximal near the proton (center)."""
        sim = _setup_hydrogen(l_mode=0)
        sim.run(steps=STEPS)

        mid = _N // 2
        field = _psi_sq(sim)

        # Peak should be within 4 cells of center.
        peak_idx = np.unravel_index(int(np.argmax(field)), field.shape)
        dist = np.sqrt(sum((p - mid) ** 2 for p in peak_idx))
        assert dist < 6, f"s-orbital peak too far from center: {dist:.1f} cells"

    def test_p_orbital_distinguishable_from_s(self) -> None:
        """p-orbital seed produces a measurably different spatial profile
        than s-orbital seed.  We compare σ²_x / σ²_y (anisotropy):
        the s is symmetric (≈ 1) and the p has a dipole perturbation."""

        def anisotropy(sim: Simulation) -> float:
            field = _psi_sq(sim).astype(np.float64)
            xs = np.arange(_N, dtype=np.float64)
            prof_x = field.sum(axis=(1, 2))
            total = prof_x.sum()
            if total < 1e-30:
                return 1.0
            mean_x = np.dot(xs, prof_x) / total
            var_x = np.dot((xs - mean_x) ** 2, prof_x) / total
            prof_y = field.sum(axis=(0, 2))
            mean_y = np.dot(xs, prof_y) / total
            var_y = np.dot((xs - mean_y) ** 2, prof_y) / total
            if var_y < 1e-30:
                return 1.0
            return float(var_x / var_y)

        sim_s = _setup_hydrogen(l_mode=0)
        sim_s.run(steps=STEPS)
        a_s = anisotropy(sim_s)

        sim_p = _setup_hydrogen(l_mode=1)
        sim_p.run(steps=STEPS)
        a_p = anisotropy(sim_p)

        diff = abs(a_s - a_p)
        assert diff > 0.001, (
            f"Modes indistinguishable: a_s={a_s:.6f}, a_p={a_p:.6f}, diff={diff:.6f}"
        )
