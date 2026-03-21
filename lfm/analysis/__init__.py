"""Post-processing analysis: structure detection, energy, observables."""

from lfm.analysis.angular_momentum import (
    angular_momentum_density,
    precession_rate,
    total_angular_momentum,
)
from lfm.analysis.color import (
    color_variance,
)
from lfm.analysis.energy import (
    continuity_residual,
    energy_components,
    energy_conservation_drift,
    fluid_fields,
    total_energy,
)
from lfm.analysis.metric import (
    effective_metric_00,
    gravitational_potential,
    metric_perturbation,
    schwarzschild_chi,
    time_dilation_factor,
)
from lfm.analysis.metrics import compute_metrics
from lfm.analysis.observables import (
    confinement_proxy,
    find_peaks,
    fit_power_law,
    measure_force,
    measure_separation,
    momentum_density,
    radial_profile,
    weak_parity_asymmetry,
)
from lfm.analysis.phase import (
    charge_density,
    coulomb_interaction_energy,
    phase_coherence,
    phase_field,
)
from lfm.analysis.spectrum import power_spectrum
from lfm.analysis.structure import (
    chi_statistics,
    count_clusters,
    interior_mask,
    void_fraction,
    well_fraction,
)
from lfm.analysis.tracker import flatten_trajectories, track_peaks

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
    "measure_separation",
    "measure_force",
    "momentum_density",
    "weak_parity_asymmetry",
    "confinement_proxy",
    # color
    "color_variance",
    # spectrum
    "power_spectrum",
    # tracker
    "track_peaks",
    "flatten_trajectories",
    # metric (spacetime geometry)
    "effective_metric_00",
    "metric_perturbation",
    "time_dilation_factor",
    "gravitational_potential",
    "schwarzschild_chi",
    # phase (EM / charge)
    "phase_field",
    "charge_density",
    "phase_coherence",
    "coulomb_interaction_energy",
    # angular momentum
    "angular_momentum_density",
    "total_angular_momentum",
    "precession_rate",
]
