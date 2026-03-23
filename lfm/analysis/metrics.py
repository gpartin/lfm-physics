"""
Combined Metrics
================

All-in-one snapshot metrics matching the production ``compute_metrics()``
pattern from primordial_soup and universe_simulator.

Returns a flat dictionary suitable for logging, JSON, or DataFrame rows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lfm.analysis.energy import energy_components
from lfm.analysis.structure import (
    chi_statistics,
    count_clusters,
    void_fraction,
    well_fraction,
)


def compute_metrics(
    psi_r: NDArray[np.floating],
    psi_r_prev: NDArray[np.floating],
    chi: NDArray[np.floating],
    dt: float,
    c: float = 1.0,
    psi_i: NDArray[np.floating] | None = None,
    psi_i_prev: NDArray[np.floating] | None = None,
    interior_mask: NDArray[np.bool_] | None = None,
    well_threshold: float = 17.0,
    void_threshold: float = 18.0,
    cluster_percentile: float = 90.0,
) -> dict[str, float]:
    """Compute a full snapshot of simulation metrics.

    Parameters
    ----------
    psi_r, psi_r_prev : ndarray
        Real part of Ψ, current and previous steps.
    chi : ndarray, shape (N, N, N)
        The χ field.
    dt : float
        Timestep.
    c : float
        Wave speed.
    psi_i, psi_i_prev : ndarray or None
        Imaginary part of Ψ (None for real fields).
    interior_mask : ndarray of bool or None
        Interior region mask (excludes frozen boundary).
    well_threshold : float
        χ well threshold.
    void_threshold : float
        χ void threshold.
    cluster_percentile : float
        Percentile for cluster detection.

    Returns
    -------
    dict[str, float]
        Flat dictionary with all metrics.
    """
    # Energy components
    T, G, V = energy_components(psi_r, psi_r_prev, chi, dt, c, psi_i, psi_i_prev)
    e_kinetic = float(np.sum(T))
    e_gradient = float(np.sum(G))
    e_potential = float(np.sum(V))
    e_total = e_kinetic + e_gradient + e_potential

    # Chi statistics
    chi_stats = chi_statistics(chi, interior_mask)

    # Structure
    wells = well_fraction(chi, well_threshold, interior_mask)
    voids = void_fraction(chi, void_threshold, interior_mask)

    # |Ψ|² for cluster detection
    if psi_r.ndim == 4:
        psi_sq = np.sum(psi_r.astype(np.float64) ** 2, axis=0)
    else:
        psi_sq = psi_r.astype(np.float64) ** 2
    if psi_i is not None:
        if psi_i.ndim == 4:
            psi_sq += np.sum(psi_i.astype(np.float64) ** 2, axis=0)
        else:
            psi_sq += psi_i.astype(np.float64) ** 2

    clusters = count_clusters(psi_sq, cluster_percentile, interior_mask)
    psi_sq_total = float(np.sum(psi_sq))

    return {
        "energy_kinetic": e_kinetic,
        "energy_gradient": e_gradient,
        "energy_potential": e_potential,
        "energy_total": e_total,
        "chi_min": chi_stats["min"],
        "chi_max": chi_stats["max"],
        "chi_mean": chi_stats["mean"],
        "chi_std": chi_stats["std"],
        "well_fraction": wells,
        "void_fraction": voids,
        "n_clusters": float(clusters),
        "psi_sq_total": psi_sq_total,
    }
