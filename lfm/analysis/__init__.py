"""Post-processing analysis: structure detection, energy, observables."""

from lfm.analysis.energy import (
    continuity_residual,
    energy_components,
    energy_conservation_drift,
    fluid_fields,
    total_energy,
)
from lfm.analysis.metrics import compute_metrics
from lfm.analysis.observables import (
    find_peaks,
    fit_power_law,
    measure_force,
    measure_separation,
    radial_profile,
)
from lfm.analysis.structure import (
    chi_statistics,
    count_clusters,
    interior_mask,
    void_fraction,
    well_fraction,
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
    "measure_separation",
    "measure_force",
]
