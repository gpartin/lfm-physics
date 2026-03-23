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

__version__ = "0.5.0"

from lfm.analysis import (
    angular_momentum_density,
    charge_density,
    chi_statistics,
    collider_event_display,
    color_current_variance,
    color_variance,
    compute_impact_parameter,
    compute_metrics,
    confinement_proxy,
    continuity_residual,
    coulomb_interaction_energy,
    count_clusters,
    detect_collision_events,
    effective_metric_00,
    energy_components,
    energy_conservation_drift,
    find_apparent_horizon,
    find_peaks,
    fit_power_law,
    flatten_trajectories,
    fluid_fields,
    flux_tube_profile,
    gravitational_potential,
    horizon_mass,
    interior_mask,
    keplerian_velocity,
    list_sparc_galaxies,
    measure_chi_midpoint,
    measure_force,
    measure_separation,
    metric_perturbation,
    momentum_density,
    phase_coherence,
    phase_field,
    power_spectrum,
    precession_rate,
    radial_profile,
    rotation_curve,
    rotation_curve_fit,
    schwarzschild_chi,
    smoothed_color_variance,
    sparc_load,
    string_tension,
    time_dilation_factor,
    total_angular_momentum,
    total_energy,
    track_peaks,
    void_fraction,
    weak_parity_asymmetry,
    well_fraction,
)
from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import (
    AGE_UNIVERSE_GYR,
    ALPHA_EM,
    ALPHA_S,
    BETA_0,
    CHI0,
    D_ST,
    DT_DEFAULT,
    E_AMPLITUDE_BY_GRID,
    EPSILON_CC,
    EPSILON_W,
    KAPPA,
    KAPPA_C,
    KAPPA_STRING,
    KAPPA_TUBE,
    LAMBDA_H,
    N_COLORS,
    N_EFOLDINGS,
    N_GENERATIONS,
    OBSERVABLE_RADIUS_PLANCK,
    OMEGA_LAMBDA,
    OMEGA_MATTER,
    PLANCK_LENGTH_M,
    PLANCK_TIME_SEC,
    RANK_SU3,
    SA_D,
    SA_GAMMA,
    SA_L,
    SIN2_THETA_W,
    TOTAL_RADIUS_LOWER_BOUND_PLANCK,
    Z2_COORD,
    D,
)
from lfm.core.backends import get_backend, gpu_available
from lfm.core.evolver import Evolver
from lfm.fields import (
    boosted_soliton,
    disk_positions,
    disk_velocities,
    equilibrate_chi,
    equilibrate_from_fields,
    gaussian_soliton,
    grid_positions,
    initialize_disk,
    place_solitons,
    poisson_solve_fft,
    seed_noise,
    sparse_positions,
    tetrahedral_positions,
    uniform_chi,
    wave_kick,
)
from lfm.planning import (
    FeasibilityReport,
    UseCaseName,
    assess_feasibility,
    estimate_memory_gb,
    scale_limit_note,
    use_case_preset,
)
from lfm.simulation import Simulation
from lfm.sweep import sweep, sweep_2d
from lfm.units import CosmicScale, PlanckScale

__all__ = [
    "__version__",
    # Constants
    "CHI0",
    "D",
    "D_ST",
    "KAPPA",
    "KAPPA_C",
    "KAPPA_STRING",
    "KAPPA_TUBE",
    "LAMBDA_H",
    "EPSILON_W",
    "EPSILON_CC",
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
    # S_a v16 constants
    "SA_GAMMA",
    "SA_L",
    "SA_D",
    "BETA_0",
    "Z2_COORD",
    "RANK_SU3",
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
    "disk_positions",
    "disk_velocities",
    "initialize_disk",
    # Planning
    "FeasibilityReport",
    "UseCaseName",
    "estimate_memory_gb",
    "assess_feasibility",
    "use_case_preset",
    "scale_limit_note",
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
    "keplerian_velocity",
    "measure_separation",
    "measure_force",
    "momentum_density",
    "rotation_curve",
    "rotation_curve_fit",
    "weak_parity_asymmetry",
    "confinement_proxy",
    "fluid_fields",
    "continuity_residual",
    # SPARC galaxy data
    "sparc_load",
    "list_sparc_galaxies",
    # Color / Confinement
    "color_variance",
    "smoothed_color_variance",
    "color_current_variance",
    "flux_tube_profile",
    "measure_chi_midpoint",
    "string_tension",
    # Spectrum & Tracker
    "power_spectrum",
    "track_peaks",
    "flatten_trajectories",
    "detect_collision_events",
    "compute_impact_parameter",
    "collider_event_display",
    # Metric (spacetime geometry)
    "effective_metric_00",
    "metric_perturbation",
    "time_dilation_factor",
    "gravitational_potential",
    "schwarzschild_chi",
    "find_apparent_horizon",
    "horizon_mass",
    # Phase (EM / charge)
    "phase_field",
    "charge_density",
    "phase_coherence",
    "coulomb_interaction_energy",
    # Angular momentum
    "angular_momentum_density",
    "total_angular_momentum",
    "precession_rate",
    # Boosted soliton
    "boosted_soliton",
    # Sweep
    "sweep",
    "sweep_2d",
    # Units
    "CosmicScale",
    "PlanckScale",
]
