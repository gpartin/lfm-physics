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

from lfm.constants import CHI0, DT_DEFAULT, OBSERVABLE_RADIUS_PLANCK, PLANCK_TIME_SEC


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


@dataclass(frozen=True)
class PlanckScale:
    """Map LFM grid cells to Planck units.

    The LFM lattice has c = 1 cell per Planck tick. This class answers:
    *how many Planck lengths does one simulation grid cell represent?*

    By default the simulation box is taken to cover the observable universe
    (diameter = 2 × OBSERVABLE_RADIUS_PLANCK Planck cells). Pass a custom
    ``box_planck_radius`` to represent a different physical scale.

    Parameters
    ----------
    grid_size : int
        Number of cells per side (e.g. 256).
    box_planck_radius : float
        Physical radius of the simulation box in Planck cells.
        Default: OBSERVABLE_RADIUS_PLANCK ≈ 8.07×10⁶⁰.

    .. note::
        ``PlanckScale`` is for simulations whose box physically spans
        ``box_planck_radius`` Planck cells — e.g. a true universe-scale grid.
        At the default (observable-universe) scale a 256-cell sim has
        ~6.3×10⁵⁸ Planck lengths per cell and ~0.11 Gyr per step, so
        ``step_to_gyr(541_000) ≈ 58_000 Gyr`` (not 13.8 Gyr).

        For the canonical 256³ 100 Mpc cosmological simulation use
        :class:`CosmicScale` instead — it is empirically calibrated so
        that step 541_000 ≅ 13.8 Gyr.

    Examples
    --------
    ::

        from lfm.units import PlanckScale

        ps = PlanckScale(grid_size=256)
        print(ps)
        # PlanckScale(N=256): 1 cell = 6.31e+58 Planck lengths, 1 step = 1.08e-01 Gyr

        # How many steps for the universe to reach present age at this scale?
        print(ps.gyr_to_step(13.8))   # ≈ 128 steps
    """

    grid_size: int = 256
    box_planck_radius: float = OBSERVABLE_RADIUS_PLANCK

    @property
    def cells_per_planck(self) -> float:
        """Number of Planck lengths per grid cell (diameter / N)."""
        return (2.0 * self.box_planck_radius) / self.grid_size

    @property
    def steps_per_planck_tick(self) -> float:
        """Planck ticks per simulation step (= cells_per_planck since c=1)."""
        return self.cells_per_planck

    @property
    def gyr_per_step(self) -> float:
        """Giga-years of cosmic time per simulation step."""
        ticks_per_gyr = (1e9 * 365.25 * 24 * 3600) / PLANCK_TIME_SEC
        return self.steps_per_planck_tick / ticks_per_gyr

    def step_to_gyr(self, step: int) -> float:
        """Convert a simulation step count to Giga-years."""
        return step * self.gyr_per_step

    def gyr_to_step(self, gyr: float) -> int:
        """Convert Giga-years to nearest simulation step."""
        return round(gyr / self.gyr_per_step)

    def __str__(self) -> str:
        cppg = self.cells_per_planck
        gps = self.gyr_per_step
        return (
            f"PlanckScale(N={self.grid_size}): "
            f"1 cell = {cppg:.2e} Planck lengths, "
            f"1 step = {gps:.3e} Gyr"
        )
