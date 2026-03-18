"""
LFM — Lattice Field Medium Physics Library
===========================================

Two governing equations. One integer (χ₀ = 19). All of physics.

Quick start::

    import lfm

    print(lfm.CHI0)        # 19.0
    print(lfm.KAPPA)       # 0.015873...
    print(lfm.ALPHA_EM)    # 0.007299... ≈ 1/137.088

    config = lfm.SimulationConfig(grid_size=64)
    sim = lfm.Simulation(config)
"""

__version__ = "0.1.0"

from lfm.constants import (
    ALPHA_EM,
    ALPHA_S,
    CHI0,
    D,
    D_ST,
    DT_DEFAULT,
    E_AMPLITUDE_BY_GRID,
    EPSILON_W,
    KAPPA,
    LAMBDA_H,
    N_COLORS,
    N_GENERATIONS,
    OMEGA_LAMBDA,
    OMEGA_MATTER,
    SIN2_THETA_W,
)
from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.core.backends import get_backend, gpu_available
from lfm.core.evolver import Evolver
from lfm.fields import (
    equilibrate_chi,
    equilibrate_from_fields,
    gaussian_soliton,
    grid_positions,
    place_solitons,
    poisson_solve_fft,
    seed_noise,
    sparse_positions,
    tetrahedral_positions,
    uniform_chi,
    wave_kick,
)

__all__ = [
    "__version__",
    # Constants
    "CHI0",
    "D",
    "D_ST",
    "KAPPA",
    "LAMBDA_H",
    "EPSILON_W",
    "ALPHA_S",
    "ALPHA_EM",
    "OMEGA_LAMBDA",
    "OMEGA_MATTER",
    "SIN2_THETA_W",
    "N_COLORS",
    "N_GENERATIONS",
    "DT_DEFAULT",
    "E_AMPLITUDE_BY_GRID",
    # Config
    "SimulationConfig",
    "FieldLevel",
    "BoundaryType",
    # Backends & Simulation
    "Evolver",
    "get_backend",
    "gpu_available",
    # Fields
    "gaussian_soliton",
    "place_solitons",
    "wave_kick",
    "poisson_solve_fft",
    "equilibrate_chi",
    "equilibrate_from_fields",
    "seed_noise",
    "uniform_chi",
    "tetrahedral_positions",
    "sparse_positions",
    "grid_positions",
]
