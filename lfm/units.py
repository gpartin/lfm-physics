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

from lfm.constants import (
    CHI0,
    DT_DEFAULT,
    OBSERVABLE_RADIUS_PLANCK,
    PLANCK_LENGTH_M,
    PLANCK_TIME_SEC,
)


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

    Parameters
    ----------
    dt : float
        Simulation timestep in lattice units (default 0.02).  The
        physical time per step scales as ``dt × Δx / c``, so this must
        match the value used in the kernel.

    .. note::
        ``PlanckScale`` derives time from first principles:
        ``gyr_per_step = cells_per_planck × dt / ticks_per_gyr``.
        At the default observable-universe scale with N=256 and dt=0.02:

        * 1 cell ≈ 6.31×10⁵⁸ Planck lengths ≈ 33 Mpc
        * 1 step ≈ 2.16×10⁻³ Gyr (≈ 2.16 Myr)
        * Present epoch (13.8 Gyr) ≈ step 6,401
        * Total box diameter ≈ 8,462 Mpc (full observable universe)

        ``CosmicScale`` is the empirically calibrated alternative
        (step 541,000 ≅ 13.8 Gyr for a 100 Mpc box).  Use
        ``CosmicScale`` for the compact canonical sim;
        use ``PlanckScale`` when you specifically want the
        first-principles Planck calibration.

    Examples
    --------
    ::

        from lfm.units import PlanckScale

        # Observable-universe scale (default)
        ps = PlanckScale(grid_size=256)
        print(ps)
        # PlanckScale(N=256): 1 cell = 6.31e+58 Planck lengths (~33 Mpc), 1 step = 2.16e-03 Gyr
        print(ps.box_size_mpc)          # ≈ 8,462 Mpc (observable universe)
        print(ps.gyr_to_step(13.8))     # 6401  (present epoch)
        print(ps.step_to_gyr(6401))     # ≈ 13.80 Gyr

        # Planck-resolution scale: 1 cell = 1 Planck length
        pp = PlanckScale.at_planck_resolution(grid_size=256)
        print(pp)
        # PlanckScale(N=256, Planck-resolution):
        #   1 cell = 1 Planck length (1.616e-35 m),
        #   1 step = 0.02 Planck ticks (1.08e-45 s)
        print(pp.cells_per_planck)      # 1.0  (exact)
        print(pp.is_planck_resolution)  # True
        print(pp.cell_size_m)           # 1.616e-35 m (= Planck length)
        print(pp.step_to_planck_ticks(1000))  # 20.0
    """

    grid_size: int = 256
    box_planck_radius: float = OBSERVABLE_RADIUS_PLANCK
    dt: float = DT_DEFAULT

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def at_planck_resolution(cls, grid_size: int = 256, dt: float = DT_DEFAULT) -> "PlanckScale":
        """Return a PlanckScale where **1 cell = 1 Planck length** exactly.

        Sets ``box_planck_radius = grid_size / 2`` so that::

            cells_per_planck = (2 * grid_size/2) / grid_size = 1.0

        This is the highest-resolution physical mode: the lattice spacing
        equals the Planck length.  For N=256 the box is 256 Planck lengths
        (~4.1×10⁻³³ m) across — a quantum-gravity domain, not cosmological.

        Parameters
        ----------
        grid_size : int
            Number of cells per side.
        dt : float
            Timestep in lattice units (default 0.02).

        Examples
        --------
        ::

            from lfm.units import PlanckScale

            ps = PlanckScale.at_planck_resolution(grid_size=256)
            print(ps.cells_per_planck)      # 1.0 exactly
            print(ps.planck_ticks_per_step) # 0.02  (= dt)
            print(ps)
            # PlanckScale(N=256, Planck-resolution):
            #   1 cell = 1 Planck length,
            #   1 step = 0.02 Planck ticks (1.08e-45 s)
        """
        return cls(grid_size=grid_size, box_planck_radius=grid_size / 2.0, dt=dt)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cells_per_planck(self) -> float:
        """Number of Planck lengths per grid cell (= box_diameter / N).

        Equal to 1.0 when constructed via :meth:`at_planck_resolution`.
        """
        return (2.0 * self.box_planck_radius) / self.grid_size

    @property
    def is_planck_resolution(self) -> bool:
        """True when this instance has exactly 1 Planck length per cell."""
        return abs(self.cells_per_planck - 1.0) < 1e-9

    @property
    def cell_size_m(self) -> float:
        """Physical size of one cell in metres."""
        return self.cells_per_planck * PLANCK_LENGTH_M

    @property
    def planck_ticks_per_step(self) -> float:
        """Physical Planck-time ticks that elapse each simulation step.

        With c = 1 Planck-length / Planck-tick and the leapfrog timestep
        ``dt`` (in lattice units), one step advances physical time by
        ``dt × Δx_physical / c = dt × cells_per_planck`` Planck ticks.
        """
        return self.cells_per_planck * self.dt

    @property
    def gyr_per_step(self) -> float:
        """Giga-years of cosmic time per simulation step (first-principles)."""
        ticks_per_gyr = (1e9 * 365.25 * 24 * 3600) / PLANCK_TIME_SEC
        return self.planck_ticks_per_step / ticks_per_gyr

    @property
    def box_size_mpc(self) -> float:
        """Physical box diameter in Mpc (derived from Planck lengths)."""
        MPC_IN_METERS: float = 3.0857e22
        diameter_m = 2.0 * self.box_planck_radius * PLANCK_LENGTH_M
        return diameter_m / MPC_IN_METERS

    def step_to_gyr(self, step: int) -> float:
        """Convert a simulation step count to Giga-years."""
        return step * self.gyr_per_step

    def gyr_to_step(self, gyr: float) -> int:
        """Convert Giga-years to nearest simulation step."""
        return round(gyr / self.gyr_per_step)

    def step_to_planck_ticks(self, step: int) -> float:
        """Convert a simulation step count to Planck-time ticks."""
        return step * self.planck_ticks_per_step

    def planck_ticks_to_step(self, ticks: float) -> int:
        """Convert Planck-time ticks to the nearest simulation step."""
        return round(ticks / self.planck_ticks_per_step)

    def step_to_seconds(self, step: int) -> float:
        """Convert a simulation step count to seconds (SI)."""
        return self.step_to_planck_ticks(step) * PLANCK_TIME_SEC

    def __str__(self) -> str:
        cpp = self.cells_per_planck
        tps = self.planck_ticks_per_step
        if self.is_planck_resolution:
            step_sec = tps * PLANCK_TIME_SEC
            return (
                f"PlanckScale(N={self.grid_size}, Planck-resolution): "
                f"1 cell = 1 Planck length ({PLANCK_LENGTH_M:.3e} m), "
                f"1 step = {tps:.4g} Planck ticks ({step_sec:.3e} s)"
            )
        gps = self.gyr_per_step
        mpc = self.box_size_mpc
        return (
            f"PlanckScale(N={self.grid_size}): "
            f"1 cell = {cpp:.2e} Planck lengths (~{mpc / self.grid_size:.1f} Mpc), "
            f"1 step = {gps:.2e} Gyr"
        )
