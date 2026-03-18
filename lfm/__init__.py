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
]
