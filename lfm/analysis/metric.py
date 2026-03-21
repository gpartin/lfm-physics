"""
Metric Analysis
===============

Extract effective spacetime geometry from the χ field.

g₀₀ = -(χ/χ₀)²  follows from the GOV-01 dispersion relation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lfm.constants import CHI0


def effective_metric_00(
    chi: NDArray,
    chi0: float = CHI0,
) -> NDArray:
    """Compute the g₀₀ component of the effective metric.

    g₀₀ = -(χ/χ₀)²

    Parameters
    ----------
    chi : ndarray (N, N, N)
        Current χ field.
    chi0 : float
        Background χ value (default 19.0).

    Returns
    -------
    g00 : ndarray
        Metric component, ≤ 0 everywhere. −1 at vacuum.
    """
    return -(chi / chi0) ** 2


def metric_perturbation(
    chi: NDArray,
    chi0: float = CHI0,
) -> NDArray:
    """Compute the metric perturbation h₀₀ = g₀₀ − η₀₀.

    h₀₀ = -(χ/χ₀)² + 1 = 1 − (χ/χ₀)²

    In the weak-field limit, h₀₀ ≈ 2Φ/c² where Φ is the Newtonian potential.

    Parameters
    ----------
    chi : ndarray (N, N, N)
        Current χ field.
    chi0 : float
        Background χ value.

    Returns
    -------
    h00 : ndarray
        Perturbation. Positive where χ < χ₀ (inside wells).
    """
    return 1.0 - (chi / chi0) ** 2


def time_dilation_factor(
    chi: NDArray,
    chi0: float = CHI0,
) -> NDArray:
    """Compute the gravitational time dilation factor.

    dτ/dt = √(−g₀₀) = χ/χ₀

    Clocks run slower where χ < χ₀ (inside wells).

    Parameters
    ----------
    chi : ndarray (N, N, N)
        Current χ field.
    chi0 : float
        Background χ value.

    Returns
    -------
    factor : ndarray
        Time dilation factor, 1.0 at vacuum, < 1 in wells.
    """
    return np.abs(chi) / chi0


def gravitational_potential(
    chi: NDArray,
    chi0: float = CHI0,
) -> NDArray:
    """Estimate the Newtonian-limit gravitational potential from χ.

    Φ/c² ≈ h₀₀/2 = (1 − (χ/χ₀)²) / 2

    Parameters
    ----------
    chi : ndarray (N, N, N)
        Current χ field.
    chi0 : float
        Background χ value.

    Returns
    -------
    phi : ndarray
        Dimensionless potential Φ/c². Negative in wells.
    """
    return 0.5 * (1.0 - (chi / chi0) ** 2)


def schwarzschild_chi(
    N: int,
    center: tuple[float, float, float],
    r_s: float,
    chi0: float = CHI0,
) -> NDArray[np.float32]:
    """Create the Schwarzschild-metric χ profile on a grid.

    χ(r) = χ₀ √(1 − r_s/r)   for r > r_s
    χ(r) = 0                   for r ≤ r_s  (inside horizon)

    Parameters
    ----------
    N : int
        Grid size per axis.
    center : tuple
        (x, y, z) center in grid coordinates.
    r_s : float
        Schwarzschild radius in grid units.
    chi0 : float
        Background χ value.

    Returns
    -------
    chi : ndarray of float32 (N, N, N)
    """
    x = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    cx, cy, cz = center
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    r = np.maximum(r, 1e-10)  # avoid division by zero at center

    safe = np.where(r > r_s, 1.0 - r_s / r, 0.0)
    chi = np.where(r > r_s, chi0 * np.sqrt(safe), 0.0)
    return chi.astype(np.float32)
