"""Physical observables measured from simulation state.

These functions extract physics from raw field data — turning
lattice values into quantities you can compare to textbook results.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def momentum_density(
    psi_real: NDArray,
    psi_imag: NDArray,
) -> dict[str, NDArray]:
    """Compute momentum density j = Im(Ψ*∇Ψ) from complex fields.

    Parameters
    ----------
    psi_real, psi_imag : ndarray
        Complex field components. Supported shapes:
        - (N, N, N): single complex field
        - (C, N, N, N): multi-component (e.g. color) field

    Returns
    -------
    dict
        Keys:
        - ``j_x``, ``j_y``, ``j_z``: momentum-density components
        - ``j_total``: scalar parity-odd combination used by GOV-02
    """
    if psi_real.shape != psi_imag.shape:
        raise ValueError("psi_real and psi_imag must have identical shapes")

    if psi_real.ndim == 3:
        pr = psi_real
        pi = psi_imag
        dpr_dx = 0.5 * (np.roll(pr, -1, axis=0) - np.roll(pr, 1, axis=0))
        dpr_dy = 0.5 * (np.roll(pr, -1, axis=1) - np.roll(pr, 1, axis=1))
        dpr_dz = 0.5 * (np.roll(pr, -1, axis=2) - np.roll(pr, 1, axis=2))
        dpi_dx = 0.5 * (np.roll(pi, -1, axis=0) - np.roll(pi, 1, axis=0))
        dpi_dy = 0.5 * (np.roll(pi, -1, axis=1) - np.roll(pi, 1, axis=1))
        dpi_dz = 0.5 * (np.roll(pi, -1, axis=2) - np.roll(pi, 1, axis=2))

        j_x = pr * dpi_dx - pi * dpr_dx
        j_y = pr * dpi_dy - pi * dpr_dy
        j_z = pr * dpi_dz - pi * dpr_dz
        j_total = 0.5 * (j_x + j_y + j_z)
        return {"j_x": j_x, "j_y": j_y, "j_z": j_z, "j_total": j_total}

    if psi_real.ndim == 4:
        j_x = np.zeros_like(psi_real[0], dtype=np.float32)
        j_y = np.zeros_like(psi_real[0], dtype=np.float32)
        j_z = np.zeros_like(psi_real[0], dtype=np.float32)

        for a in range(psi_real.shape[0]):
            comp = momentum_density(psi_real[a], psi_imag[a])
            j_x += comp["j_x"]
            j_y += comp["j_y"]
            j_z += comp["j_z"]

        j_total = 0.5 * (j_x + j_y + j_z)
        return {"j_x": j_x, "j_y": j_y, "j_z": j_z, "j_total": j_total}

    raise ValueError("Expected psi arrays with ndim 3 or 4")


def weak_parity_asymmetry(
    chi: NDArray,
    axis: int = 0,
) -> dict[str, float]:
    """Measure parity asymmetry in χ depressions along one axis.

    Uses only the simulated χ field: depression = max(χ) - χ.
    The metric is

        A = (W_plus - W_minus) / (W_plus + W_minus)

    where W_plus and W_minus are summed depressions in opposite
    hemispheres split at the lattice center along ``axis``.

    Parameters
    ----------
    chi : ndarray (N,N,N)
        Chi field.
    axis : int
        Axis to test (0, 1, or 2).

    Returns
    -------
    dict
        ``plus_weight``, ``minus_weight``, ``asymmetry``.
    """
    if chi.ndim != 3:
        raise ValueError("chi must be a 3D array")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    n = chi.shape[0]
    center = n // 2
    dep = np.max(chi) - chi

    idx = np.arange(n)
    slicer_plus = [slice(None), slice(None), slice(None)]
    slicer_minus = [slice(None), slice(None), slice(None)]
    slicer_plus[axis] = idx > center
    slicer_minus[axis] = idx < center

    plus = float(np.sum(dep[tuple(slicer_plus)]))
    minus = float(np.sum(dep[tuple(slicer_minus)]))
    denom = plus + minus
    asym = (plus - minus) / denom if denom > 0 else 0.0
    return {
        "plus_weight": plus,
        "minus_weight": minus,
        "asymmetry": float(asym),
    }


def confinement_proxy(
    chi: NDArray,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    samples: int = 64,
) -> dict[str, float]:
    """Estimate line-like confinement energy between two points.

    Computes a line integral of χ depression (max(χ)-χ) along the
    segment p0→p1 with nearest-grid sampling:

        I = ∫ (χ_ref - χ) ds,  with χ_ref = max(χ)

    For a flux-tube-like configuration, I grows approximately linearly
    with segment length.

    Parameters
    ----------
    chi : ndarray (N,N,N)
        Chi field.
    p0, p1 : tuple[float, float, float]
        Segment endpoints in grid coordinates.
    samples : int
        Number of samples along the segment.

    Returns
    -------
    dict
        ``distance``, ``line_integral``, ``mean_depression``.
    """
    if chi.ndim != 3:
        raise ValueError("chi must be a 3D array")
    if samples < 2:
        raise ValueError("samples must be >= 2")

    n = chi.shape[0]
    p0v = np.asarray(p0, dtype=np.float64)
    p1v = np.asarray(p1, dtype=np.float64)
    distance = float(np.linalg.norm(p1v - p0v))

    ts = np.linspace(0.0, 1.0, samples)
    pts = p0v[None, :] + (p1v - p0v)[None, :] * ts[:, None]
    idx = np.rint(pts).astype(int)
    idx = np.clip(idx, 0, n - 1)

    chi_ref = float(np.max(chi))
    vals = chi_ref - chi[idx[:, 0], idx[:, 1], idx[:, 2]]
    if samples > 1:
        ds = distance / (samples - 1)
    else:
        ds = 0.0
    line_integral = float(np.sum(vals) * ds)
    mean_dep = float(np.mean(vals))

    return {
        "distance": distance,
        "line_integral": line_integral,
        "mean_depression": mean_dep,
    }


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


def rotation_curve(
    chi: NDArray,
    energy_density: NDArray,
    center: tuple[int, int, int] | None = None,
    c: float = 1.0,
    chi0: float = 19.0,
    kappa: float = 1.0 / 63.0,
    plane_axis: int = 2,
    max_radius: float | None = None,
    n_bins: int | None = None,
) -> dict[str, NDArray]:
    """Compute the radial rotation curve v_circ(r) from χ field and |Ψ|².

    Two complementary estimates are returned:

    * **chi_gradient** — uses the local gravitational acceleration directly::

        v_circ²(r) = r · (c²/χ₀) · |dχ/dr|

      This is measurement-based and works in the weak and strong field regimes.

    * **enclosed_mass** — uses the Newtonian analogy::

        v_circ²(r) = G_eff · M_enc(r) / r,  G_eff = c⁴ / (κ · χ₀² · c²)

      Useful for comparing with MOND-like predictions.

    Parameters
    ----------
    chi : ndarray (N, N, N)
        χ field.
    energy_density : ndarray (N, N, N)
        |Ψ|² energy density (the "mass" source for GOV-02).
    center : (x, y, z) or None
        Rotation-curve center.  Defaults to grid centre.
    c : float
        Wave speed (default 1.0 = lattice units).
    chi0 : float
        Vacuum χ (default 19.0).
    kappa : float
        GOV-02 coupling constant (default κ = 1/63).
    plane_axis : int
        Axis *perpendicular* to the disk (0, 1, or 2).
    max_radius : float or None
        Maximum radius in cells (default = N/2 − 2).
    n_bins : int or None
        Number of radial bins (default = max_radius).

    Returns
    -------
    dict with keys:
        ``r``            — radii (cell units)
        ``v_chi``        — v_circ from χ gradient (lattice units, c = 1)
        ``v_enc``        — v_circ from enclosed mass
        ``v_keplerian``  — pure-Keplerian (total mass at r_max, Newtonian)
        ``m_enclosed``   — cumulative enclosed |Ψ|² vs r
        ``chi_profile``  — mean χ vs r (for plotting)
    """
    chi_arr = np.asarray(chi, dtype=np.float64)
    e_arr   = np.asarray(energy_density, dtype=np.float64)
    N = chi_arr.shape[0]
    if center is None:
        center = (N // 2, N // 2, N // 2)
    if max_radius is None:
        max_radius = N // 2 - 2
    if n_bins is None:
        n_bins = int(max_radius)

    cx, cy, cz = center
    ix = np.arange(N, dtype=np.float64) - cx
    iy = np.arange(N, dtype=np.float64) - cy
    iz = np.arange(N, dtype=np.float64) - cz
    DX, DY, DZ = np.meshgrid(ix, iy, iz, indexing="ij")

    # 3-D radius
    R = np.sqrt(DX**2 + DY**2 + DZ**2)

    r_edges = np.linspace(0.0, max_radius, n_bins + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

    chi_profile   = np.zeros(n_bins)
    m_enclosed_all = np.zeros(n_bins)  # cumulative mass up to bin i

    # chi radial derivative via finite diff on binned profile
    chi_r_mean   = np.full(n_bins, chi0)  # fallback: vacuum
    r_int        = (R / (max_radius / n_bins)).astype(int)
    r_int        = np.clip(r_int, 0, n_bins - 1)

    counts = np.bincount(r_int.ravel(), minlength=n_bins)
    chi_sum = np.bincount(r_int.ravel(), weights=chi_arr.ravel(), minlength=n_bins)
    e_sum   = np.bincount(r_int.ravel(), weights=e_arr.ravel(), minlength=n_bins)

    for b in range(n_bins):
        if counts[b] > 0:
            chi_r_mean[b] = chi_sum[b] / counts[b]

    # Enclosed mass: sum of |Ψ|² in shells out to r (discrete cumsum)
    dr = max_radius / n_bins
    shell_mass = e_sum  # each element ≈ sum of |Ψ|² in the bin
    m_enclosed_all = np.cumsum(shell_mass)

    chi_profile = chi_r_mean

    # dchi/dr via central difference
    dchi_dr = np.gradient(chi_r_mean, r_centres)

    # v_circ from chi gradient (|dchi/dr| should be positive = chi drops inward)
    v_chi_sq = r_centres * (c**2 / chi0) * np.abs(dchi_dr)
    v_chi = np.sqrt(np.maximum(v_chi_sq, 0.0))

    # Effective Newton constant: G_eff = c^4 / (kappa * chi0^2)
    G_eff = c**4 / (kappa * chi0**2)
    m_enc_positive = np.maximum(m_enclosed_all, 0.0)
    r_safe = np.maximum(r_centres, 0.1)
    v_enc_sq = G_eff * m_enc_positive / r_safe
    v_enc = np.sqrt(v_enc_sq)

    # Pure Keplerian (all mass at galactic centre)
    M_total = float(m_enclosed_all[-1])
    v_kep_sq = G_eff * M_total / r_safe
    v_keplerian = np.sqrt(v_kep_sq)

    return {
        "r":           r_centres.astype(np.float32),
        "v_chi":       v_chi.astype(np.float32),
        "v_enc":       v_enc.astype(np.float32),
        "v_keplerian": v_keplerian.astype(np.float32),
        "m_enclosed":  m_enclosed_all.astype(np.float32),
        "chi_profile": chi_profile.astype(np.float32),
    }


def keplerian_velocity(
    r: NDArray,
    m_total: float,
    kappa: float = 1.0 / 63.0,
    chi0: float = 19.0,
    c: float = 1.0,
) -> NDArray[np.float32]:
    """Compute the pure-Keplerian circular velocity for a point mass.

    In LFM natural units, the effective gravitational constant is::

        G_eff = c⁴ / (κ · χ₀²)

    so the Keplerian velocity is::

        v_K(r) = sqrt(G_eff · M / r)

    Parameters
    ----------
    r : 1-D array
        Radii in lattice cells.
    m_total : float
        Total mass (sum of |Ψ|²) — treated as a point mass at the origin.
    kappa : float
        GOV-02 coupling (default 1/63).
    chi0 : float
        Vacuum χ (default 19).
    c : float
        Wave speed (default 1).

    Returns
    -------
    ndarray, float32 — Keplerian velocity at each r.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    G_eff = c**4 / (kappa * chi0**2)
    r_safe = np.maximum(r_arr, 1e-30)
    return np.sqrt(G_eff * m_total / r_safe).astype(np.float32)


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
