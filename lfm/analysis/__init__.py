"""Post-processing analysis: structure detection, energy, observables."""

from lfm.analysis.angular_momentum import (
    angular_momentum_density,
    precession_rate,
    total_angular_momentum,
)
from lfm.analysis.color import (
    color_variance,
)
from lfm.analysis.confinement import (
    color_current_variance,
    flux_tube_profile,
    measure_chi_midpoint,
    smoothed_color_variance,
    string_tension,
)
from lfm.analysis.cosmology import (
    correlation_function,
    halo_mass_function,
    matter_power_spectrum,
    void_statistics,
)
from lfm.analysis.energy import (
    continuity_residual,
    energy_components,
    energy_conservation_drift,
    fluid_fields,
    total_energy,
)
from lfm.analysis.grav_waves import (
    gravitational_wave_strain,
    gw_power,
    gw_quadrupole,
)
from lfm.analysis.metric import (
    effective_metric_00,
    find_apparent_horizon,
    gravitational_potential,
    horizon_mass,
    metric_perturbation,
    schwarzschild_chi,
    time_dilation_factor,
)
from lfm.analysis.metrics import compute_metrics
from lfm.analysis.observables import (
    confinement_proxy,
    find_peaks,
    fit_power_law,
    keplerian_velocity,
    measure_force,
    measure_separation,
    momentum_density,
    radial_profile,
    rotation_curve,
    rotation_curve_fit,
    weak_parity_asymmetry,
)
from lfm.analysis.phase import (
    charge_density,
    coulomb_interaction_energy,
    phase_coherence,
    phase_field,
)
from lfm.analysis.sparc import list_sparc_galaxies, sparc_load
from lfm.analysis.spectrum import power_spectrum
from lfm.analysis.structure import (
    chi_statistics,
    count_clusters,
    interior_mask,
    void_fraction,
    well_fraction,
)
from lfm.analysis.tracker import (
    collider_event_display,
    compute_impact_parameter,
    detect_collision_events,
    flatten_trajectories,
    track_peaks,
)

__all__ = [
    # energy
    "energy_components",
    "total_energy",
    "energy_conservation_drift",
    "fluid_fields",
    "continuity_residual",
    # structure
    "chi_statistics",
    "well_fraction",
    "void_fraction",
    "count_clusters",
    "interior_mask",
    # metrics
    "compute_metrics",
    # observables
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
    # sparc
    "sparc_load",
    "list_sparc_galaxies",
    # color
    "color_variance",
    # confinement (v16 S_a)
    "smoothed_color_variance",
    "color_current_variance",
    "flux_tube_profile",
    "measure_chi_midpoint",
    "string_tension",
    # spectrum
    "power_spectrum",
    # tracker
    "track_peaks",
    "flatten_trajectories",
    "detect_collision_events",
    "compute_impact_parameter",
    "collider_event_display",
    # metric (spacetime geometry)
    "effective_metric_00",
    "metric_perturbation",
    "time_dilation_factor",
    "gravitational_potential",
    "schwarzschild_chi",
    "find_apparent_horizon",
    "horizon_mass",
    # phase (EM / charge)
    "phase_field",
    "charge_density",
    "phase_coherence",
    "coulomb_interaction_energy",
    # angular momentum
    "angular_momentum_density",
    "total_angular_momentum",
    "precession_rate",
    # cosmological statistics
    "correlation_function",
    "matter_power_spectrum",
    "halo_mass_function",
    "void_statistics",
    # gravitational waves
    "gravitational_wave_strain",
    "gw_quadrupole",
    "gw_power",
]
