"""Post-processing analysis for LFM simulation output.

This module collects statistics and derived observables from the two
LFM fields (Ψ and χ):

- **Energy** — kinetic, gradient, potential, and cross-coupling
  components tracked via :func:`energy_components`.
- **Structure detection** — χ-well / void / cluster counting via
  :func:`chi_statistics`.
- **Orbital mechanics** — Kepler fits, angular momentum, precession
  rates for N-body experiments.
- **Gravitational waves** — strain tensor and power spectrum from χ
  fluctuations.
- **Colour / QCD probes** — colour variance f_c and confinement
  metrics for Level-2 multi-component Ψ runs.

All functions accept plain NumPy (or CuPy) arrays and return NumPy
arrays / plain Python dicts, so they work both on CPU and GPU
simulations without copying.
"""

from lfm.analysis.angular_momentum import (
    angular_momentum_density,
    precession_rate,
    total_angular_momentum,
)
from lfm.analysis.coarse_graining import (
    block_window_magnitude_sq,
    blocked_static_propagator,
    inverse_response_intercept,
    response_log_slope,
)
from lfm.analysis.collective_geometry import (
    SOURCE_CASES,
    ContinuumFit,
    WeightedMoments,
    analytic_leapfrog_limit,
    apply_momentum_sponge,
    block_average,
    collective_initial_state,
    continuum_fit,
    dispersion_shell_metrics,
    energy_current_vector_and_tensor,
    minimum_image_mesh,
    periodic_weighted_moments,
    traceless,
)
from lfm.analysis.collective_spectrum import (
    RotatingBackground,
    berry_action_derivative_audit,
    collective_mode_eigenpairs,
    collective_qep_matrices,
    discrete_stiffness_19,
    gov02_background_residual,
    mode_polarization,
    principal_symbol_audit,
    rotating_background,
    vacuum_spectrum_audit,
    zero_chi_branch_audit,
)
from lfm.analysis.color import (
    color_variance,
)
from lfm.analysis.confinement import (  # noqa: F401
    classify_potential,
    color_current_variance,
    fit_cornell,
    fit_coulomb,
    fit_yukawa,
    flux_tube_profile,
    measure_chi_midpoint,
    smoothed_color_variance,
    static_interaction_potential,
    string_tension,
)
from lfm.analysis.cosmology import (
    correlation_function,
    halo_mass_function,
    matter_power_spectrum,
    void_statistics,
)
from lfm.analysis.emergence import (
    aggregate_wave_density,
    localized_state_observables,
    plaquette_winding_summary,
    principal_internal_projection,
)
from lfm.analysis.energy import (
    continuity_residual,
    energy_components,
    energy_conservation_drift,
    fluid_fields,
    total_energy,
)
from lfm.analysis.energy_current import (
    BareHamiltonRates,
    BareLFMParameters,
    BareLFMState,
    bare_energy_continuity_residual,
    bare_hamilton_rates,
    bare_site_energy,
    bare_site_energy_rate,
    bare_total_energy,
    energy_current_divergence,
    oriented_energy_currents,
    stencil_links,
    step_bare_lfm,
    wave_component_site_energy,
)
from lfm.analysis.frame_candidates import (
    CandidateAssessment,
    CandidateVerdict,
    FrameCandidate,
    assess_frame_candidate,
    current_frame_candidate_ledger,
)
from lfm.analysis.frame_completion import (
    FRAME_COMPONENT_COUNT,
    FRAME_COMPONENT_LABELS,
    FRAME_SCALE_COUNT,
    FRAME_SHAPE_COUNT,
    analytic_rest_energy_response,
    frame_projectors,
    frame_scale_direction,
    frame_static_operator,
    frame_static_response,
    minimized_source_cross_energy,
    rest_energy_source,
    source_projection_weights,
    zero_momentum_frame_spectrum,
)
from lfm.analysis.frame_links import (
    linked_frame_difference,
    loop_holonomy,
    loop_mismatch_energy,
    reconstructed_frame_link,
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
    metric_refractive_index,
    op05_spherical_chi_deflection,
    schwarzschild_chi,
    schwarzschild_radius_si,
    time_dilation_factor,
)
from lfm.analysis.metrics import compute_metrics
from lfm.analysis.modes import (
    leapfrog_branch_projection,
    periodic_mode_coefficient,
    project_leapfrog_mode,
)
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
from lfm.analysis.particle_kinematics import (
    component_noether_charges,
    fit_offset_power_convergence,
    flat_octic_hamiltonian_19pt,
    time_centered_momentum_19pt,
)
from lfm.analysis.phase import (
    bare_charge_continuity_residual,
    canonical_charge_density,
    charge_current_divergence,
    charge_density,
    coulomb_interaction_energy,
    noether_spatial_current,
    oriented_charge_currents,
    phase_coherence,
    phase_current_energy_density,
    phase_field,
    positive_noether_current,
)
from lfm.analysis.ringdown import (
    fit_ringdown_series,
    project_field_onto_modes,
    relative_spread,
    split_frequency_bands,
    target_band_summary,
)
from lfm.analysis.sparc import list_sparc_galaxies, sparc_load
from lfm.analysis.spectrum import power_spectrum
from lfm.analysis.spinor import (
    spinor_center_of_energy,
    spinor_density,
    spinor_interference_energy,
    spinor_sigma_x,
    spinor_sigma_y,
    spinor_sigma_z,
)
from lfm.analysis.structure import (
    chi_statistics,
    count_clusters,
    interior_mask,
    void_fraction,
    well_fraction,
)
from lfm.analysis.substrate_emergence import (
    axial_static_inverse_length,
    composite_connection_19pt,
    composite_curvature_19pt,
    curvature_rms,
    normalize_internal_field,
    relational_wave_scaling_scan,
)
from lfm.analysis.tracker import (
    collider_event_display,
    compute_impact_parameter,
    detect_collision_events,
    flatten_trajectories,
    localized_weighted_centroid,
    track_peaks,
)

__all__ = [
    # energy
    "energy_components",
    "total_energy",
    "energy_conservation_drift",
    "fluid_fields",
    "continuity_residual",
    "BareLFMParameters",
    "BareHamiltonRates",
    "BareLFMState",
    "stencil_links",
    "bare_site_energy",
    "bare_hamilton_rates",
    "bare_site_energy_rate",
    "oriented_energy_currents",
    "energy_current_divergence",
    "bare_energy_continuity_residual",
    "bare_total_energy",
    "step_bare_lfm",
    "wave_component_site_energy",
    "SOURCE_CASES",
    "WeightedMoments",
    "ContinuumFit",
    "minimum_image_mesh",
    "collective_initial_state",
    "apply_momentum_sponge",
    "periodic_weighted_moments",
    "block_average",
    "energy_current_vector_and_tensor",
    "traceless",
    "continuum_fit",
    "dispersion_shell_metrics",
    "analytic_leapfrog_limit",
    "FrameCandidate",
    "CandidateAssessment",
    "CandidateVerdict",
    "assess_frame_candidate",
    "current_frame_candidate_ledger",
    "FRAME_COMPONENT_LABELS",
    "FRAME_COMPONENT_COUNT",
    "FRAME_SCALE_COUNT",
    "FRAME_SHAPE_COUNT",
    "frame_scale_direction",
    "frame_projectors",
    "rest_energy_source",
    "source_projection_weights",
    "frame_static_operator",
    "frame_static_response",
    "analytic_rest_energy_response",
    "minimized_source_cross_energy",
    "zero_momentum_frame_spectrum",
    "reconstructed_frame_link",
    "linked_frame_difference",
    "loop_holonomy",
    "loop_mismatch_energy",
    "block_window_magnitude_sq",
    "blocked_static_propagator",
    "inverse_response_intercept",
    "response_log_slope",
    "RotatingBackground",
    "berry_action_derivative_audit",
    "collective_mode_eigenpairs",
    "collective_qep_matrices",
    "discrete_stiffness_19",
    "gov02_background_residual",
    "mode_polarization",
    "principal_symbol_audit",
    "rotating_background",
    "vacuum_spectrum_audit",
    "zero_chi_branch_audit",
    # substrate-only emergence diagnostics
    "aggregate_wave_density",
    "principal_internal_projection",
    "plaquette_winding_summary",
    "localized_state_observables",
    "axial_static_inverse_length",
    "relational_wave_scaling_scan",
    "normalize_internal_field",
    "composite_connection_19pt",
    "composite_curvature_19pt",
    "curvature_rms",
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
    "localized_weighted_centroid",
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
    "schwarzschild_radius_si",
    "metric_refractive_index",
    "op05_spherical_chi_deflection",
    "find_apparent_horizon",
    "horizon_mass",
    # phase (EM / charge)
    "phase_field",
    "charge_density",
    "canonical_charge_density",
    "oriented_charge_currents",
    "charge_current_divergence",
    "bare_charge_continuity_residual",
    "noether_spatial_current",
    "positive_noether_current",
    "phase_current_energy_density",
    "phase_coherence",
    "coulomb_interaction_energy",
    # external collective particle kinematics
    "component_noether_charges",
    "flat_octic_hamiltonian_19pt",
    "time_centered_momentum_19pt",
    "fit_offset_power_convergence",
    # periodic spatial/temporal mode projections
    "periodic_mode_coefficient",
    "leapfrog_branch_projection",
    "project_leapfrog_mode",
    # ringdown extraction
    "fit_ringdown_series",
    "project_field_onto_modes",
    "relative_spread",
    "split_frequency_bands",
    "target_band_summary",
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
    # spinor (Paper 048)
    "spinor_density",
    "spinor_sigma_z",
    "spinor_sigma_x",
    "spinor_sigma_y",
    "spinor_interference_energy",
    "spinor_center_of_energy",
]
