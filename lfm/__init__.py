"""
LFM — Lattice Field Medium Physics Library
===========================================

Simulate the universe from two equations.

Quick start::

    import lfm

    sim = lfm.Simulation(lfm.SimulationConfig(grid_size=64))
    sim.place_soliton((32, 32, 32), amplitude=6.0)
    sim.equilibrate()
    sim.run(steps=2000)
    print(sim.metrics())
"""

__version__ = "0.2.0"

from lfm.analysis import (
    chi_statistics,
    compute_metrics,
    continuity_residual,
    count_clusters,
    energy_components,
    energy_conservation_drift,
    find_peaks,
    fit_power_law,
    fluid_fields,
    interior_mask,
    measure_force,
    measure_separation,
    radial_profile,
    total_energy,
    void_fraction,
    well_fraction,
)
from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import (
    ALPHA_EM,
    ALPHA_S,
    AGE_UNIVERSE_GYR,
    CHI0,
    D_ST,
    DT_DEFAULT,
    E_AMPLITUDE_BY_GRID,
    EPSILON_W,
    KAPPA,
    LAMBDA_H,
    N_COLORS,
    N_EFOLDINGS,
    N_GENERATIONS,
    OBSERVABLE_RADIUS_PLANCK,
    OMEGA_LAMBDA,
    OMEGA_MATTER,
    PLANCK_LENGTH_M,
    PLANCK_TIME_SEC,
    SIN2_THETA_W,
    TOTAL_RADIUS_LOWER_BOUND_PLANCK,
    D,
)
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
from lfm.simulation import Simulation
from lfm.units import CosmicScale, PlanckScale

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
    "N_EFOLDINGS",
    "DT_DEFAULT",
    "E_AMPLITUDE_BY_GRID",
    # Cosmological scale
    "AGE_UNIVERSE_GYR",
    "OBSERVABLE_RADIUS_PLANCK",
    "TOTAL_RADIUS_LOWER_BOUND_PLANCK",
    "PLANCK_TIME_SEC",
    "PLANCK_LENGTH_M",
    # Config
    "SimulationConfig",
    "FieldLevel",
    "BoundaryType",
    # Backends & Simulation
    "Evolver",
    "Simulation",
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
    # Analysis
    "energy_components",
    "total_energy",
    "energy_conservation_drift",
    "chi_statistics",
    "well_fraction",
    "void_fraction",
    "count_clusters",
    "interior_mask",
    "compute_metrics",
    # Observables
    "radial_profile",
    "find_peaks",
    "fit_power_law",
    "measure_separation",
    "measure_force",
    "fluid_fields",
    "continuity_residual",
    # Units
    "CosmicScale",
    "PlanckScale",
]
