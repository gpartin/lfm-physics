"""Physical units mapping for LFM simulations.

The lattice operates in natural units (Δx = c = 1).  This module
provides lightweight helpers to convert between lattice quantities
and SI / cosmological units when interpreting results.

Usage::

    from lfm.units import CosmicScale

    scale = CosmicScale(box_mpc=100.0)
    print(scale.step_to_gyr(541_000))   # ~13.8 Gyr
    print(scale.cell_to_mpc())          # ~0.39 Mpc (for N=256)
"""

from __future__ import annotations

from dataclasses import dataclass

from lfm.constants import CHI0, DT_DEFAULT


@dataclass(frozen=True)
class CosmicScale:
    """Map lattice units to cosmological scales.

    Parameters
    ----------
    box_mpc : float
        Physical size of the simulation box in Mpc.
    grid_size : int
        Number of cells per side (e.g. 256).
    dt : float
        Simulation timestep in lattice units (default 0.02).
    chi0 : float
        Background chi value (default 19.0).
    """

    box_mpc: float = 100.0
    grid_size: int = 256
    dt: float = DT_DEFAULT
    chi0: float = CHI0

    def cell_to_mpc(self) -> float:
        """Physical size of one cell in Mpc."""
        return self.box_mpc / self.grid_size

    def step_to_gyr(self, step: int) -> float:
        """Convert simulation step to Giga-years.

        Uses the canonical calibration: 1.2M steps ≈ 30.6 Gyr
        (from the 256³ cosmic history simulation).
        The conversion factor is dt × box_mpc / grid_size,
        scaled so that step 541_000 ≈ 13.8 Gyr (present epoch).
        """
        # Calibration: 541_000 steps = 13.8 Gyr in canonical 256³ sim
        # → gyr_per_step = 13.8 / 541_000 ≈ 2.55e-5
        # Scale by box size relative to canonical 100 Mpc
        gyr_per_step_canonical = 13.8 / 541_000
        box_factor = self.box_mpc / 100.0
        dt_factor = self.dt / 0.02
        return step * gyr_per_step_canonical * box_factor * dt_factor

    def gyr_to_step(self, gyr: float) -> int:
        """Convert Giga-years to simulation step."""
        gyr_per_step_canonical = 13.8 / 541_000
        box_factor = self.box_mpc / 100.0
        dt_factor = self.dt / 0.02
        return int(gyr / (gyr_per_step_canonical * box_factor * dt_factor))

    def format_cosmic_time(self, step: int) -> str:
        """Human-readable cosmic time string."""
        gyr = self.step_to_gyr(step)
        if gyr < 0.001:
            return f"{gyr * 1000:.1f} Myr"
        return f"{gyr:.2f} Gyr"
