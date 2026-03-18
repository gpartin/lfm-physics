"""
Structure Detection
===================

Detect gravitational wells, voids, and clusters in LFM fields.

Production patterns from primordial_soup_v8_four_forces.py:
- Wells: χ < WELL_THRESHOLD (17.0)
- Voids: χ > VOID_THRESHOLD (18.0)
- Clusters: connected components of high |Ψ|² via scipy.ndimage.label
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from lfm.constants import VOID_THRESHOLD, WELL_THRESHOLD


def chi_statistics(
    chi: NDArray[np.floating],
    interior_mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Compute χ field statistics.

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
        The χ field.
    interior_mask : ndarray of bool or None
        If given, only compute stats where mask is True (excludes boundary).

    Returns
    -------
    dict with keys: min, max, mean, std
    """
    region = chi[interior_mask] if interior_mask is not None else chi.ravel()
    return {
        "min": float(np.min(region)),
        "max": float(np.max(region)),
        "mean": float(np.mean(region)),
        "std": float(np.std(region)),
    }


def well_fraction(
    chi: NDArray[np.floating],
    threshold: float = WELL_THRESHOLD,
    interior_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Fraction of grid points in gravitational wells (χ < threshold).

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
        The χ field.
    threshold : float
        Well threshold (default 17.0).
    interior_mask : ndarray of bool or None
        If given, only consider interior points.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    region = chi[interior_mask] if interior_mask is not None else chi.ravel()
    if region.size == 0:
        return 0.0
    return float(np.mean(region < threshold))


def void_fraction(
    chi: NDArray[np.floating],
    threshold: float = VOID_THRESHOLD,
    interior_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Fraction of grid points in voids (χ > threshold).

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
        The χ field.
    threshold : float
        Void threshold (default 18.0).
    interior_mask : ndarray of bool or None
        If given, only consider interior points.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    region = chi[interior_mask] if interior_mask is not None else chi.ravel()
    if region.size == 0:
        return 0.0
    return float(np.mean(region > threshold))


def count_clusters(
    field: NDArray[np.floating],
    threshold_percentile: float = 90.0,
    interior_mask: NDArray[np.bool_] | None = None,
) -> int:
    """Count connected clusters above a percentile threshold.

    Uses scipy.ndimage.label on the binary mask (field > percentile).

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        Scalar field to cluster (typically |Ψ|²).
    threshold_percentile : float
        Percentile threshold (default 90th) above which points count.
    interior_mask : ndarray of bool or None
        If given, mask out boundary before labeling.

    Returns
    -------
    int
        Number of connected clusters.
    """
    if interior_mask is not None:
        vals = field[interior_mask]
    else:
        vals = field.ravel()

    if vals.size == 0:
        return 0

    thresh = np.percentile(vals, threshold_percentile)
    binary = field > thresh
    if interior_mask is not None:
        binary = binary & interior_mask

    _, n_clusters = ndimage.label(binary)
    return int(n_clusters)


def interior_mask(
    N: int,
    boundary_fraction: float = 0.3,
) -> NDArray[np.bool_]:
    """Create a boolean mask for the interior region (excludes frozen boundary).

    Parameters
    ----------
    N : int
        Grid size per axis.
    boundary_fraction : float
        Fraction of grid radius for boundary shell.

    Returns
    -------
    ndarray of bool, shape (N, N, N)
        True in the interior, False in the boundary shell.
    """
    center = N / 2.0
    radius = center * (1.0 - boundary_fraction)
    x = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)
    return dist <= radius
