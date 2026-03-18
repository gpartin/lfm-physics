"""Post-processing analysis: structure detection, energy, spectra."""

from lfm.analysis.energy import (
    energy_components,
    energy_conservation_drift,
    total_energy,
)
from lfm.analysis.metrics import compute_metrics
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
    # structure
    "chi_statistics",
    "well_fraction",
    "void_fraction",
    "count_clusters",
    "interior_mask",
    # metrics
    "compute_metrics",
]
