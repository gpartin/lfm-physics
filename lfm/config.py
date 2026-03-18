"""
Simulation Configuration
========================

SimulationConfig dataclass with validation.
Extracted from production code in universe_simulator and primordial_soup.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from lfm.constants import (
    BOUNDARY_FRACTION_DEFAULT,
    C_DEFAULT,
    CFL_19PT_MASSIVE,
    CHI0,
    DT_DEFAULT,
    E0_SQ_DEFAULT,
    E_AMPLITUDE_BY_GRID,
    EPSILON_W,
    KAPPA,
    N_COLORS,
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

    # Derived (computed in __post_init__)
    dx: float = field(init=False, default=1.0)
    """Grid spacing. Always 1.0 in natural units."""

    sigma: float = field(init=False, default=0.0)
    """Gaussian soliton width = grid_size / blob_sigma_factor."""

    def __post_init__(self) -> None:
        self._validate()
        self.dx = 1.0
        self.sigma = self.grid_size / self.blob_sigma_factor
        if self.e_amplitude == 0.0:
            self.e_amplitude = E_AMPLITUDE_BY_GRID.get(
                self.grid_size, 3.6  # default for unlisted grid sizes
            )

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.grid_size < 8:
            raise ValueError(f"grid_size must be >= 8, got {self.grid_size}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.dt > CFL_19PT_MASSIVE:
            raise ValueError(
                f"dt={self.dt} exceeds CFL limit {CFL_19PT_MASSIVE:.4f} "
                f"for 19-point stencil with χ₀={self.chi0}"
            )
        if self.chi0 <= 0:
            raise ValueError(f"chi0 must be positive, got {self.chi0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.lambda_self < 0:
            raise ValueError(f"lambda_self must be >= 0, got {self.lambda_self}")
        if not 0 < self.boundary_fraction < 1:
            raise ValueError(
                f"boundary_fraction must be in (0, 1), got {self.boundary_fraction}"
            )
        if self.field_level == FieldLevel.COLOR and self.n_colors < 1:
            raise ValueError(f"n_colors must be >= 1, got {self.n_colors}")
