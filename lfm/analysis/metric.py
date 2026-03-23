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
    return -((chi / chi0) ** 2)


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


# ---------------------------------------------------------------------------
# Apparent horizon detection (v16 black-hole analysis)
# ---------------------------------------------------------------------------


def find_apparent_horizon(
    chi: NDArray,
    center: tuple[int, int, int] | None = None,
    chi0: float = CHI0,
    threshold: float | None = None,
) -> dict:
    """Locate the apparent horizon — the surface where χ → 0.

    In LFM the effective metric g₀₀ = −(χ/χ₀)² vanishes where χ = 0,
    so the horizon is defined as the closed χ = 0 surface.  We find it
    by detecting the outermost contiguous region where χ ≤ ``threshold``
    and fitting an effective sphere to it.

    Parameters
    ----------
    chi : ndarray (N, N, N)
        χ field.
    center : (x, y, z) or None
        Black-hole centre.  If None, estimated as the χ minimum.
    chi0 : float
        Vacuum χ value (default 19).
    threshold : float or None
        χ value below which a cell is "inside the horizon".
        Defaults to 0.05 * chi0 (5 % — avoids discretisation noise
        near the exact zero crossing).

    Returns
    -------
    dict with keys:
        ``found``         — bool: True if ≥ 1 sub-threshold cell found
        ``r_horizon``     — effective horizon radius in cells (float)
        ``center``        — (x, y, z) used as black-hole centre
        ``n_cells``       — number of cells inside horizon
        ``chi_min``       — global χ minimum
        ``chi_min_pos``   — grid position of χ minimum
        ``time_dilation`` — mean χ/χ₀ just outside horizon (clock rate)
    """
    chi_arr = np.asarray(chi, dtype=np.float64)
    N = chi_arr.shape[0]

    if threshold is None:
        threshold = 0.05 * chi0

    chi_min_flat = float(chi_arr.min())
    min_pos = tuple(int(x) for x in np.unravel_index(np.argmin(chi_arr), chi_arr.shape))

    if center is None:
        center = min_pos

    inside = chi_arr <= threshold
    n_cells = int(inside.sum())

    if n_cells == 0:
        return {
            "found": False,
            "r_horizon": 0.0,
            "center": center,
            "n_cells": 0,
            "chi_min": chi_min_flat,
            "chi_min_pos": min_pos,
            "time_dilation": 1.0,
        }

    # Fit effective spherical radius from volume  r = (3V / 4π)^(1/3)
    r_horizon = float((3.0 * n_cells / (4.0 * np.pi)) ** (1.0 / 3.0))

    # Mean time dilation just outside the horizon (r_horizon ± 2 cells)
    cx, cy, cz = center
    idx = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    shell = (R >= r_horizon) & (R <= r_horizon + 2)
    if shell.sum() > 0:
        td = float(np.abs(chi_arr[shell]).mean() / chi0)
    else:
        td = float(np.abs(chi_arr).mean() / chi0)

    return {
        "found": True,
        "r_horizon": r_horizon,
        "center": center,
        "n_cells": n_cells,
        "chi_min": chi_min_flat,
        "chi_min_pos": min_pos,
        "time_dilation": td,
    }


def horizon_mass(
    r_s: float,
    chi0: float = CHI0,
    kappa: float = 1.0 / 63.0,
    c: float = 1.0,
) -> float:
    """Estimate LFM black-hole mass from the Schwarzschild radius.

    In LFM::

        G_eff = c⁴ / (κ · χ₀²)

    so the Schwarzschild relation  r_s = 2 G_eff M / c²  gives::

        M = r_s · κ · χ₀² / (2 · c²)       (natural units: c = 1)

    Parameters
    ----------
    r_s : float
        Schwarzschild (horizon) radius in lattice cells.
    chi0 : float
        Vacuum χ.
    kappa : float
        GOV-02 coupling constant.
    c : float
        Wave speed.

    Returns
    -------
    float — effective BH mass in |Ψ|² units.
    """
    G_eff = c**4 / (kappa * chi0**2)
    return float(r_s * c**2 / (2.0 * G_eff))
