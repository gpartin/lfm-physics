"""Task 4.5 — Gravitational Lensing.

A wave packet in a frozen χ-gradient (created by a massive soliton) should
drift toward the χ-well.  We create the landscape via a mass + equilibration,
then wipe the wave content, inject an isolated probe, and evolve with
``evolve_chi=False`` so ONLY wave refraction in the fixed geometry is tested.
"""

from __future__ import annotations

import numpy as np

from lfm.config_presets import gravity_only
from lfm.simulation import Simulation

from .conftest import BACKEND, GRID_TWO, STEPS

_N = GRID_TWO
_SIGMA_PROBE = 2.5


def _y_centroid(field: np.ndarray) -> float:
    """Energy-weighted centroid along y-axis (axis 1)."""
    if field.ndim == 4:
        field = field.sum(axis=0)
    profile = field.sum(axis=(0, 2)).astype(np.float64)
    total = profile.sum()
    if total < 1e-30:
        return _N / 2.0
    coords = np.arange(_N, dtype=np.float64)
    return float(np.dot(coords, profile) / total)


class TestGravitationalLensing:
    """Wave energy refracts toward a χ-well in a frozen landscape."""

    def test_deflection_toward_mass(self) -> None:
        """Probe's y-centroid drifts toward the χ-well."""
        mid = _N // 2
        mass_y = mid + 10  # well location along y

        # --- Build χ landscape via mass + equilibration ---
        cfg = gravity_only(grid_size=_N)
        sim = Simulation(cfg, backend=BACKEND)
        sim.place_soliton((mid, mass_y, mid), amplitude=10.0)
        sim.equilibrate()

        # Confirm χ is depressed near the mass.
        chi_at_mass = float(sim.chi[mid, mass_y, mid])
        chi_at_probe = float(sim.chi[mid, mid - 2, mid])
        assert chi_at_mass < chi_at_probe, "No χ-well found"

        # --- Wipe wave content, inject isolated probe ---
        shape = sim.psi_real.shape
        zeros = np.zeros(shape, dtype=np.float32)
        sim.set_psi_real(zeros.copy())
        sim.set_psi_real_prev(zeros.copy())
        if sim.psi_imag is not None:
            sim.set_psi_imag(zeros.copy())
            sim.set_psi_imag_prev(zeros.copy())

        # Gaussian probe at (mid, mid-2, mid) — well is at (mid, mid+10, mid).
        xs = np.arange(_N, dtype=np.float32)
        yy = xs.reshape(1, _N, 1)
        xx = xs.reshape(_N, 1, 1)
        zz = xs.reshape(1, 1, _N)
        probe = np.exp(
            -((xx - mid) ** 2 + (yy - (mid - 2)) ** 2 + (zz - mid) ** 2) / (2 * _SIGMA_PROBE**2)
        ).astype(np.float32)
        if len(shape) == 4:
            probe = probe[np.newaxis]
        sim.set_psi_real(probe.copy())
        sim.set_psi_real_prev(probe.copy())  # start at rest

        # --- Evolve with frozen χ ---
        y0 = _y_centroid(sim.psi_real**2)
        sim.run(steps=STEPS, evolve_chi=False)
        y1 = _y_centroid(sim.psi_real**2)

        drift = y1 - y0  # positive = toward mass
        assert drift > 0, f"No deflection toward χ-well: drift={drift:.4f}"
