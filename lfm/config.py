"""
Simulation Configuration
========================

SimulationConfig dataclass with validation.
Extracted from production code in universe_simulator and primordial_soup.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfm.units import PlanckScale

from lfm.constants import (
    BOUNDARY_FRACTION_DEFAULT,
    C_DEFAULT,
    CHI0,
    DT_DEFAULT,
    E0_SQ_DEFAULT,
    E_AMPLITUDE_BY_GRID,
    EPSILON_W,
    KAPPA,
    KAPPA_C,
    LAMBDA_H,
    N_COLORS,
    OBSERVABLE_RADIUS_PLANCK,
    SA_D,
    SA_GAMMA,
)


class FieldLevel(enum.IntEnum):
    """Field representation level.

    Level 0: Real E — gravity only (cosmology, dark matter)
    Level 1: Complex Ψ — gravity + EM (charged particles)
    Level 2: Complex Ψₐ (3-color) — all four forces
    """

    REAL = 0
    COMPLEX = 1
    COLOR = 2


class ChiMode(enum.Enum):
    """Which GOV-02 update rule to use for the χ field.

    WAVE
        Full second-order PDE (default). Both ∂²χ/∂t² and ∇²χ active.
        Required for solitons, orbits, quantum, nuclear, and solar-scale.
    MEMORY
        GOV-03 quasi-static approximation: χ² = χ₀² − g⟨|Ψ|²⟩_τ.
        χ tracks a running average of energy over τ steps (dark matter halo).
        Appropriate for galactic- and cosmic-scale dark matter simulations.
    STATIC
        GOV-04 Poisson limit: ∇²χ = (κ/c²)(|Ψ|² − E₀²).
        χ solved once from instantaneous energy density.
        Fastest; valid when χ responds much faster than Ψ dynamics.
    """

    WAVE = "wave"
    MEMORY = "memory"
    STATIC = "static"


class PhysicsScale(enum.Enum):
    """Named physical scale of the simulation.

    Setting ``physical_scale`` on :class:`SimulationConfig` (or using
    :meth:`SimulationConfig.for_scale`) auto-selects the appropriate
    :class:`FieldLevel`, :class:`ChiMode`, and coupling constants.

    The table below lists the *minimum* physical size of one grid cell
    for each scale, the forces active, and which GOV equations are used.

    +------------+----------------+------------+-----------+-------------------+
    | Scale      | Δx / cell      | FieldLevel | ChiMode   | Active physics    |
    +============+================+============+===========+===================+
    | PLANCK     | 1.6 × 10⁻³⁵ m | COLOR      | WAVE      | All 4 forces      |
    | NUCLEAR    | ~ 1 fm         | COLOR      | WAVE      | QCD + confinement |
    | ATOMIC     | ~ 0.1 Å        | COMPLEX    | WAVE      | EM + orbitals     |
    | MOLECULAR  | ~ 1 nm         | COMPLEX    | WAVE      | Chemistry / bonds |
    | CLASSICAL  | > 1 μm         | REAL       | WAVE      | Gravity only      |
    | SOLAR      | ~ 10⁹ m        | REAL       | WAVE      | Orbital mechanics |
    | GALACTIC   | ~ 1 parsec     | REAL       | MEMORY    | Dark matter halo  |
    | COSMIC     | ~ 1 Mpc        | REAL       | MEMORY    | Large-scale struct|
    +------------+----------------+------------+-----------+-------------------+

    The minimum resolvable particle at each scale is one with Compton
    wavelength ≥ one grid cell (σ_min ≈ 1.0 cell).
    """

    PLANCK = "planck"
    NUCLEAR = "nuclear"
    ATOMIC = "atomic"
    MOLECULAR = "molecular"
    CLASSICAL = "classical"
    SOLAR = "solar"
    GALACTIC = "galactic"
    COSMIC = "cosmic"


# Physical size of one grid cell at each named scale (metres).
_SCALE_CELL_SIZE_M: dict[PhysicsScale, float] = {
    PhysicsScale.PLANCK:    1.616e-35,   # 1 Planck length
    PhysicsScale.NUCLEAR:   1.0e-15,     # 1 fm
    PhysicsScale.ATOMIC:    1.0e-11,     # 0.1 Angstrom
    PhysicsScale.MOLECULAR: 1.0e-9,      # 1 nm
    PhysicsScale.CLASSICAL: 1.0e-4,      # 0.1 mm
    PhysicsScale.SOLAR:     1.5e11,      # 1 AU
    PhysicsScale.GALACTIC:  3.086e16,    # 1 parsec
    PhysicsScale.COSMIC:    3.086e22,    # 1 Mpc
}

# Regime default: (FieldLevel, ChiMode, lambda_self, kappa_c)
_SCALE_DEFAULTS: dict[PhysicsScale, tuple[FieldLevel, ChiMode, float, float]] = {
    PhysicsScale.PLANCK:    (FieldLevel.COLOR,   ChiMode.WAVE,   LAMBDA_H, KAPPA_C),
    PhysicsScale.NUCLEAR:   (FieldLevel.COLOR,   ChiMode.WAVE,   LAMBDA_H, KAPPA_C),
    PhysicsScale.ATOMIC:    (FieldLevel.COMPLEX, ChiMode.WAVE,   0.0,      0.0),
    PhysicsScale.MOLECULAR: (FieldLevel.COMPLEX, ChiMode.WAVE,   0.0,      0.0),
    PhysicsScale.CLASSICAL: (FieldLevel.REAL,    ChiMode.WAVE,   0.0,      0.0),
    PhysicsScale.SOLAR:     (FieldLevel.REAL,    ChiMode.WAVE,   0.0,      0.0),
    PhysicsScale.GALACTIC:  (FieldLevel.REAL,    ChiMode.MEMORY, 0.0,      0.0),
    PhysicsScale.COSMIC:    (FieldLevel.REAL,    ChiMode.MEMORY, 0.0,      0.0),
}


class BoundaryType(enum.Enum):
    """Boundary condition type."""

    FROZEN = "frozen"
    """χ frozen at χ₀ in outer shell, Ψ = 0. Production default."""

    PERIODIC = "periodic"
    """Wraparound. Use for spectral analysis, NOT structure formation."""

    ABSORBING = "absorbing"
    """Sponge layer. Reduces reflections for scattering experiments."""


@dataclass
class SimulationConfig:
    """Complete configuration for an LFM simulation.

    All parameters have validated defaults matching production code.
    """

    # Grid
    grid_size: int = 128
    """Number of lattice points per axis (N). Grid is N×N×N."""

    # Physics
    chi0: float = CHI0
    """Background χ value. Default 19.0 (DO NOT change unless you know why)."""

    kappa: float = KAPPA
    """Gravity coupling = 1/63."""

    lambda_self: float = 0.0
    """Mexican hat self-coupling. 0.0 = gravity-only (cosmological default).
    Set to LAMBDA_H (4/31) for Higgs physics, BH interiors, vacuum stability."""

    epsilon_w: float = EPSILON_W
    """Weak/helicity coupling = 0.1. Only matters when j(x,t) is computed."""

    kappa_c: float = 0.0
    """Color variance coupling (v14). 0.0 = colorblind (v13 default).
    Set to KAPPA_C (1/189) for color-aware χ deepening of non-singlet states.
    Only active at FieldLevel.COLOR."""

    epsilon_cc: float = 0.0
    """Cross-color coupling (v15). 0.0 = independent colors (v14 default).
    Set to EPSILON_CC (2/17) for dynamic f_c evolution.
    Adds −ε_cc·χ²·(Ψₐ − Ψ̄) to GOV-01. Only active at FieldLevel.COLOR."""

    kappa_string: float = 0.0
    """Color current variance (CCV) coupling (v15 GOV-02). Default 0.0 = off.
    Set to KAPPA_STRING (= KAPPA_C = 1/189) for v15-style GOV-02 CCV term.
    Adds −κ_string·CCV to GOV-02. Only active at FieldLevel.COLOR."""

    kappa_tube: float = 0.0
    """Smoothed color variance (SCV) coupling (v16). Default 0.0 = off.
    Set to KAPPA_TUBE (30/63) for full confinement with λ_self=LAMBDA_H,
    or to 10*KAPPA for stable experiments without Mexican hat.
    Adds −κ_tube·SCV to GOV-02. Only active at FieldLevel.COLOR."""

    sa_gamma: float = SA_GAMMA
    """S_a decay rate γ = 0.1. Only used when kappa_tube > 0."""

    sa_d: float = SA_D
    """S_a diffusion coefficient D = γ·L² = 4.9. Only used when kappa_tube > 0."""

    e0_sq: float = E0_SQ_DEFAULT
    """Background energy density. 0 = vacuum."""

    # Numerics
    dt: float = DT_DEFAULT
    """Timestep. Default 0.02. Must satisfy CFL condition."""

    c: float = C_DEFAULT
    """Wave speed. 1.0 in natural lattice units."""

    # Field type
    field_level: FieldLevel = FieldLevel.REAL
    """Which field representation to use."""

    n_colors: int = N_COLORS
    """Number of color components (only used at FieldLevel.COLOR)."""

    # Boundary
    boundary_type: BoundaryType = BoundaryType.FROZEN
    """Boundary condition type. FROZEN is production default."""

    boundary_fraction: float = BOUNDARY_FRACTION_DEFAULT
    """Fraction of grid radius for frozen boundary shell."""

    # Initialization
    e_amplitude: float = 0.0
    """Soliton amplitude. If 0, looked up from E_AMPLITUDE_BY_GRID."""

    blob_sigma_factor: float = 12.0
    """Sigma = grid_size / blob_sigma_factor for Gaussian solitons."""

    # Parametric resonance (matter creation)
    phase1_steps: int = 10_000
    """Steps of χ oscillation for parametric resonance seeding."""

    phase1_omega: float = 38.0
    """Oscillation frequency for resonance = 2χ₀."""

    phase1_amplitude: float = 0.3
    """Oscillation amplitude for resonance seeding."""

    # Runtime
    random_seed: int = 42
    """Random seed for reproducibility."""

    report_interval: int = 5000
    """Print metrics every N steps."""

    # Cosmological scale (metadata — does not affect physics kernels)
    box_planck_radius: float = OBSERVABLE_RADIUS_PLANCK
    """Physical radius of the simulation box in Planck cells.

    Default = OBSERVABLE_RADIUS_PLANCK ≈ 8.07×10⁶⁰ (observable universe).
    Every grid cell then represents ~(2×radius/N) Planck lengths.
    Override to simulate a sub-universe region at finer Planck resolution.
    This value is metadata only — it does not change the physics kernels.
    """

    # Physical regime
    physical_scale: PhysicsScale | None = None
    """Named physical scale.  When set, the appropriate :class:`FieldLevel`,
    :class:`ChiMode`, and coupling constants (``lambda_self``, ``kappa_c``)
    are applied automatically in ``__post_init__``.

    The specific couplings chosen are those in ``_SCALE_DEFAULTS``.
    Any coupling you set *before* this auto-selection will be overridden.
    To take manual control, leave ``physical_scale=None`` and set
    ``field_level`` / ``chi_mode`` explicitly.

    Use :meth:`SimulationConfig.for_scale` as a convenience factory::

        cfg = SimulationConfig.for_scale(PhysicsScale.SOLAR, grid_size=256)
    """

    chi_mode: ChiMode = ChiMode.WAVE
    """Which GOV-02 update rule to use.  See :class:`ChiMode`.

    ``WAVE`` (default): full second-order PDE — required for solitons, orbits,
    and quantum-scale simulations.
    ``MEMORY``: GOV-03 τ-averaging (dark matter halos).
    ``STATIC``: GOV-04 Poisson solve (quasi-static Newtonian limit).

    Set automatically when ``physical_scale`` is provided.
    """

    # Derived (computed in __post_init__)
    dx: float = field(init=False, default=1.0)
    """Grid spacing. Always 1.0 in natural units."""

    sigma: float = field(init=False, default=0.0)
    """Gaussian soliton width = grid_size / blob_sigma_factor."""

    def __post_init__(self) -> None:
        # Apply regime defaults BEFORE validation so derived fields see them.
        if self.physical_scale is not None:
            fl, cm, ls, kc = _SCALE_DEFAULTS[self.physical_scale]
            self.field_level = fl
            self.chi_mode = cm
            self.lambda_self = ls
            self.kappa_c = kc
        self._validate()
        self.dx = 1.0
        self.sigma = self.grid_size / self.blob_sigma_factor
        if self.e_amplitude == 0.0:
            self.e_amplitude = E_AMPLITUDE_BY_GRID.get(
                self.grid_size,
                3.6,  # default for unlisted grid sizes
            )

    def _validate(self) -> None:
        """Validate configuration parameters."""
        import warnings
        if self.grid_size < 8:
            raise ValueError(f"grid_size must be >= 8, got {self.grid_size}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        # CFL limit depends on chi0: dt < 1/sqrt(16/3 + chi0^2)
        # For chi0=0 (massless/EM) the limit is the wave-only CFL: 1/sqrt(16/3)
        import math
        cfl_limit = 1.0 / math.sqrt(16.0 / 3.0 + self.chi0 ** 2)
        if self.dt > cfl_limit:
            raise ValueError(
                f"dt={self.dt} exceeds CFL limit {cfl_limit:.4f} "
                f"for 19-point stencil with \u03c7\u2080={self.chi0}"
            )
        if self.chi0 <= 0:
            raise ValueError(f"chi0 must be > 0, got {self.chi0}")
        if self.kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa}")
        if self.lambda_self < 0:
            raise ValueError(f"lambda_self must be >= 0, got {self.lambda_self}")
        if not 0 < self.boundary_fraction < 1:
            raise ValueError(f"boundary_fraction must be in (0, 1), got {self.boundary_fraction}")
        if self.field_level == FieldLevel.COLOR and self.n_colors < 1:
            raise ValueError(f"n_colors must be >= 1, got {self.n_colors}")
        # Cross-check: warn when field_level disagrees with physical_scale.
        if self.physical_scale is None and self.field_level != FieldLevel.REAL:
            # User set field_level manually — no warning needed.
            pass
        elif self.physical_scale is not None:
            expected_fl = _SCALE_DEFAULTS[self.physical_scale][0]
            if self.field_level != expected_fl:
                warnings.warn(
                    f"physical_scale={self.physical_scale.value!r} expects "
                    f"FieldLevel.{expected_fl.name} but field_level="
                    f"FieldLevel.{self.field_level.name} is set. "
                    "The physical_scale auto-selection has been overridden.",
                    stacklevel=3,
                )

    @property
    def sa_enabled(self) -> bool:
        """True when S_a auxiliary fields are active (kappa_tube > 0)."""
        return self.kappa_tube > 0.0

    @property
    def cell_size_m(self) -> float | None:
        """Physical size of one grid cell in metres, or *None* if not anchored.

        Returns the cell size for the :attr:`physical_scale` if one is set,
        otherwise ``None`` (the user has not provided a physical anchor).

        Example::

            cfg = SimulationConfig.for_scale(PhysicsScale.SOLAR, grid_size=256)
            print(cfg.cell_size_m)   # 1.5e+11  (1 AU per cell)
        """
        if self.physical_scale is None:
            return None
        return _SCALE_CELL_SIZE_M[self.physical_scale]

    @property
    def minimum_particle_size_m(self) -> float | None:
        """Smallest resolvable particle diameter in metres, or *None*.

        Defined as ``cell_size_m`` (i.e. σ_min = 1.0 grid cells).
        A particle with Compton wavelength smaller than this cannot be
        represented on the current grid — use a finer :class:`PhysicsScale`.
        """
        cs = self.cell_size_m
        return cs  # σ_min = 1.0 cell

    @classmethod
    def for_scale(
        cls,
        scale: PhysicsScale,
        grid_size: int = 128,
        **kwargs,
    ) -> SimulationConfig:
        """Create a :class:`SimulationConfig` with regime defaults for *scale*.

        This is the recommended way to configure scale-based simulations::

            # Nuclear-scale: COLOR field, WAVE chi, Mexican hat on
            cfg = SimulationConfig.for_scale(PhysicsScale.NUCLEAR, grid_size=64)

            # Solar-scale: REAL field, WAVE chi, gravity only
            cfg = SimulationConfig.for_scale(PhysicsScale.SOLAR, grid_size=256)

            # Galactic-scale: REAL field, MEMORY chi (dark matter)
            cfg = SimulationConfig.for_scale(PhysicsScale.GALACTIC, grid_size=128)

        Any keyword argument overrides the regime default::

            cfg = SimulationConfig.for_scale(
                PhysicsScale.ATOMIC, grid_size=64, dt=0.01
            )

        Parameters
        ----------
        scale:
            Named physical scale (see :class:`PhysicsScale`).
        grid_size:
            Grid side length N.
        **kwargs:
            Additional overrides passed to :class:`SimulationConfig`.

        Returns
        -------
        SimulationConfig
        """
        return cls(grid_size=grid_size, physical_scale=scale, **kwargs)

    @property
    def planck_scale(self) -> PlanckScale:
        """Planck-unit scale for this configuration.

        Returns a :class:`~lfm.units.PlanckScale` providing the conversion
        between simulation steps/cells and Planck ticks/lengths.

        Example::

            cfg = SimulationConfig(grid_size=256)
            print(cfg.planck_scale)
            # PlanckScale(N=256): 1 cell = 6.30e+58 Planck lengths, ...
        """
        from lfm.units import PlanckScale

        return PlanckScale(grid_size=self.grid_size, box_planck_radius=self.box_planck_radius)
