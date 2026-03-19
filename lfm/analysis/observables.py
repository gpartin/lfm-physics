"""Physical observables measured from simulation state.

These functions extract physics from raw field data — turning
lattice values into quantities you can compare to textbook results.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def radial_profile(
    field: NDArray,
    center: tuple[int, int, int] | None = None,
    max_radius: float | None = None,
) -> dict[str, NDArray]:
    """Compute the azimuthally-averaged radial profile of a 3D field.

    Parameters
    ----------
    field : ndarray (N,N,N)
        Any 3D scalar field (chi, |psi|^2, energy density, ...).
    center : (x, y, z) or None
        Center point.  Defaults to grid center.
    max_radius : float or None
        Maximum radius in cells.  Defaults to half the grid size.

    Returns
    -------
    dict with keys:
        'r'       — 1D array of radii (cell units)
        'profile' — mean field value at each radius
        'std'     — standard deviation at each radius
        'counts'  — number of cells in each radial bin
    """
    N = field.shape[0]
    if center is None:
        center = (N // 2, N // 2, N // 2)
    if max_radius is None:
        max_radius = N // 2 - 2

    # Build distance array
    ix = np.arange(N) - center[0]
    iy = np.arange(N) - center[1]
    iz = np.arange(N) - center[2]
    dx, dy, dz = np.meshgrid(ix, iy, iz, indexing="ij")
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Bin by integer radius
    r_int = np.rint(r).astype(int)
    n_bins = int(max_radius) + 1
    radii = np.arange(n_bins, dtype=float)
    profile = np.zeros(n_bins)
    std = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = r_int == i
        vals = field[mask]
        counts[i] = len(vals)
        if counts[i] > 0:
            profile[i] = vals.mean()
            std[i] = vals.std()

    return {"r": radii, "profile": profile, "std": std, "counts": counts}


def find_peaks(
    psi_sq: NDArray,
    n: int = 2,
    min_separation: int = 5,
) -> list[tuple[int, int, int]]:
    """Find the N brightest |Ψ|² peaks on the grid.

    Parameters
    ----------
    psi_sq : ndarray (N,N,N)
        Energy density |Ψ|².
    n : int
        Number of peaks to find.
    min_separation : int
        Minimum distance between peaks (cells).

    Returns
    -------
    List of (x, y, z) positions, brightest first.
    """
    from scipy.ndimage import maximum_filter

    # Suppress non-maxima
    local_max = maximum_filter(psi_sq, size=min_separation)
    candidates = np.argwhere((psi_sq == local_max) & (psi_sq > 0))

    if len(candidates) == 0:
        return []

    # Sort by brightness (descending)
    values = psi_sq[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    order = np.argsort(-values)
    candidates = candidates[order]

    # Greedy selection with minimum separation
    selected: list[tuple[int, int, int]] = []
    for c in candidates:
        pos = (int(c[0]), int(c[1]), int(c[2]))
        too_close = False
        for s in selected:
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(pos, s)))
            if dist < min_separation:
                too_close = True
                break
        if not too_close:
            selected.append(pos)
        if len(selected) >= n:
            break
    return selected


def measure_separation(
    psi_sq: NDArray,
    min_peak_separation: int = 5,
) -> float:
    """Distance between the two brightest |Ψ|² peaks.

    Returns
    -------
    float — Euclidean distance in cell units.
    """
    peaks = find_peaks(psi_sq, n=2, min_separation=min_peak_separation)
    if len(peaks) < 2:
        return 0.0
    a, b = peaks[0], peaks[1]
    return float(np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b))))


def measure_force(
    chi_before: NDArray,
    chi_after: NDArray,
    center: tuple[int, int, int],
    dt: float = 0.02,
    n_steps: int = 1,
) -> NDArray:
    """Estimate the force on a soliton from the change in chi gradient.

    The acceleration of a wave packet is proportional to -∇χ.
    This function measures the gradient at a given center point.

    Parameters
    ----------
    chi_before, chi_after : ndarray (N,N,N)
        Chi field at two timesteps.
    center : (x, y, z)
        Point to evaluate the gradient.
    dt : float
        Timestep.
    n_steps : int
        Number of steps between before and after.

    Returns
    -------
    ndarray of shape (3,) — force components (fx, fy, fz) in lattice units.
    """
    # Use midpoint chi for gradient
    chi_mid = 0.5 * (chi_before + chi_after)
    cx, cy, cz = center

    # Central difference gradient at the center
    grad = np.array([
        chi_mid[cx + 1, cy, cz] - chi_mid[cx - 1, cy, cz],
        chi_mid[cx, cy + 1, cz] - chi_mid[cx, cy - 1, cz],
        chi_mid[cx, cy, cz + 1] - chi_mid[cx, cy, cz - 1],
    ]) / 2.0

    # Force = -∇χ (particles move toward lower χ)
    return -grad


def fit_power_law(
    r: NDArray,
    profile: NDArray,
    r_min: float = 2.0,
    r_max: float | None = None,
) -> tuple[float, float]:
    """Fit a power-law to a radial profile via log-log regression.

    Fits ``profile(r) ∝ r^exponent`` over the range ``[r_min, r_max]``.
    Useful for checking that a χ-depression follows 1/r (Newtonian gravity)::

        exponent, r_sq = lfm.fit_power_law(prof["r"], delta_chi)
        # Expect exponent ≈ -1.00, r_sq ≈ 1.0 for perfect 1/r

    Parameters
    ----------
    r : array-like
        Radial distances (e.g. from :func:`radial_profile`).
    profile : array-like
        Values at each radius.  Only positive values are included.
    r_min : float
        Minimum radius to include in the fit.
    r_max : float or None
        Maximum radius to include.  Defaults to ``max(r)``.

    Returns
    -------
    exponent : float
        Best-fit power-law exponent.
    r_squared : float
        Coefficient of determination R² of the log-log fit.
        Returns (nan, 0.0) if fewer than 3 usable points.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    p_arr = np.asarray(profile, dtype=np.float64)
    if r_max is None:
        r_max = float(r_arr.max())

    mask = (r_arr >= r_min) & (r_arr <= r_max) & (p_arr > 0) & (r_arr > 0)
    if mask.sum() < 3:
        return float("nan"), 0.0

    log_r = np.log(r_arr[mask])
    log_p = np.log(p_arr[mask])

    # Linear regression on log-log data: log_p = exponent * log_r + const
    A = np.column_stack([log_r, np.ones_like(log_r)])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_p, rcond=None)

    pred = A @ coeffs
    ss_res = float(np.sum((log_p - pred) ** 2))
    ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(coeffs[0]), float(r_sq)
