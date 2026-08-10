"""
Metric Analysis
===============

Extract effective spacetime geometry from the χ field.

g₀₀ = -(χ/χ₀)²  follows from the GOV-01 dispersion relation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from lfm.constants import ARCSEC_PER_RADIAN, C_SI, CHI0, G_SI

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def schwarzschild_radius_si(
    mass_kg: float,
    gravitational_constant: float = G_SI,
    c_si: float = C_SI,
) -> float:
    """Return the Schwarzschild radius in meters for an SI mass input."""
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive")
    if gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    if c_si <= 0.0:
        raise ValueError("c_si must be positive")
    return float(2.0 * gravitational_constant * mass_kg / (c_si * c_si))


def metric_refractive_index(
    chi: NDArray,
    chi0: float = CHI0,
    ppn_gamma: float = 1.0,
) -> NDArray:
    """Return the LFM geometric-optics refractive index from the chi metric.

    The GOV-01 metric map gives g00 = -(chi/chi0)^2. In the weak-field
    optical limit, the PPN spatial metric contribution gives
    n = (chi0 / chi) ** (1 + gamma). The canonical LFM weak-GR closure has
    gamma = 1, so n = (chi0 / chi) ** 2.
    """
    if chi0 <= 0.0:
        raise ValueError("chi0 must be positive")
    if ppn_gamma < 0.0:
        raise ValueError("ppn_gamma must be non-negative")
    chi_f = np.asarray(chi, dtype=np.float64)
    if np.any(chi_f <= 0.0):
        raise ValueError("chi must be positive for metric refractive index")
    return np.power(chi0 / chi_f, 1.0 + ppn_gamma)


def op05_spherical_chi_deflection(
    mass_kg: float,
    impact_parameter_m: float,
    x_extent_multiplier: float = 500.0,
    sample_count: int = 20001,
    ppn_gamma: float = 1.0,
) -> dict[str, object]:
    """Integrate OP-05 for a spherical GR-16 chi profile.

    This uses the canonical LFM chain:

    - GR-16: chi/chi0 = sqrt(1 - r_s / r)
    - GR-24: gamma = 1 for the weak-field spatial metric
    - OP-05: dtheta/dx = (1/n) * partial_y n

    The returned comparator is not used by the integration; it is the
    closed-form weak-field value 2*r_s/b for checking the numerical result.
    """
    if impact_parameter_m <= 0.0:
        raise ValueError("impact_parameter_m must be positive")
    if x_extent_multiplier <= 1.0:
        raise ValueError("x_extent_multiplier must be greater than 1")
    if sample_count < 101:
        raise ValueError("sample_count must be at least 101")
    if sample_count % 2 == 0:
        sample_count += 1

    rs_m = schwarzschild_radius_si(mass_kg)
    x_extent_m = float(x_extent_multiplier) * impact_parameter_m
    x_m = np.linspace(-x_extent_m, x_extent_m, sample_count, dtype=np.float64)
    y0_m = np.full_like(x_m, impact_parameter_m)
    radius_m = np.sqrt(x_m * x_m + y0_m * y0_m)
    exponent = 0.5 * (1.0 + ppn_gamma)
    safe = np.maximum(1.0 - rs_m / radius_m, 1.0e-15)
    n_eff = np.power(safe, -exponent)

    dn_dr = -exponent * rs_m / (radius_m * radius_m) * np.power(safe, -exponent - 1.0)
    dn_dy = dn_dr * y0_m / radius_m
    dtheta_dx = dn_dy / n_eff
    angle_rad = abs(float(np.trapezoid(dtheta_dx, x_m)))
    comparator_rad = 2.0 * rs_m / impact_parameter_m

    increments = 0.5 * (dtheta_dx[1:] + dtheta_dx[:-1]) * np.diff(x_m)
    theta_rad = np.concatenate([[0.0], np.cumsum(increments)])
    y_increments = 0.5 * (theta_rad[1:] + theta_rad[:-1]) * np.diff(x_m)
    y_m = impact_parameter_m + np.concatenate([[0.0], np.cumsum(y_increments)])

    return {
        "mass_kg": float(mass_kg),
        "impact_parameter_m": float(impact_parameter_m),
        "schwarzschild_radius_m": float(rs_m),
        "x_extent_multiplier": float(x_extent_multiplier),
        "sample_count": int(sample_count),
        "ppn_gamma": float(ppn_gamma),
        "x_over_b": (x_m / impact_parameter_m).tolist(),
        "y_over_b": (y_m / impact_parameter_m).tolist(),
        "theta_arcsec": (np.abs(theta_rad) * ARCSEC_PER_RADIAN).tolist(),
        "n_eff": n_eff.tolist(),
        "dtheta_dx": dtheta_dx.tolist(),
        "recovered_angle_radians": float(angle_rad),
        "recovered_angle_arcsec": float(angle_rad * ARCSEC_PER_RADIAN),
        "canonical_comparator_arcsec": float(comparator_rad * ARCSEC_PER_RADIAN),
        "comparator_relative_error": float((angle_rad - comparator_rad) / comparator_rad),
    }


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
        center = cast("tuple[int, int, int]", min_pos)

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
    assert center is not None
    cx, cy, cz = center
    idx = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    shell = (r_horizon <= R) & (r_horizon + 2 >= R)
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
