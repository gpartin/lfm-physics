"""Confinement analysis for v16 S_a auxiliary fields.

Provides observables for measuring whether the LFM v16 confinement
mechanism (S_a flux tubes) produces string-tension behaviour.

All functions accept plain NumPy arrays so they work after ``get_*``
calls on either CPU or GPU simulations.

Physics background
------------------
GOV-02 v16 adds a term  −κ_tube · SCV  where SCV is the *smoothed*
colour variance:

    SCV = Σ_a S_a² − (1/3)(Σ_a S_a)²

S_a satisfy  dS_a/dt = D ∇²S_a + γ(|Ψ_a|² − S_a)  which smooths
the colour energy density over a length scale L = sqrt(D/γ) ≈ 7 cells
(the LFM S_a correlation length, derived from β₀ = 7).

String tension σ is extracted by measuring χ_midpoint vs colour-source
separation r and fitting  χ_mid(r) = χ₀ − σ·r  for large r.  A positive
slope (χ drops as r grows) indicates an energy cost proportional to
separation — flux-tube confinement.

References: Paper 080 Session 143; LFM_CONFINEMENT_MECHANISM.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Smoothed colour variance  (SCV)
# ---------------------------------------------------------------------------


def smoothed_color_variance(
    sa_fields: NDArray,
) -> NDArray[np.float32]:
    """Compute the smoothed colour variance SCV from S_a fields.

    SCV = Σ_a S_a² − (1/3) (Σ_a S_a)²

    Parameters
    ----------
    sa_fields : ndarray, shape (3, N, N, N) or (3*N³,)
        Smoothed per-colour energy density.  Values must be ≥ 0.

    Returns
    -------
    scv : ndarray, shape (N, N, N), float32
        SCV at every lattice point.  Zero when all colours are balanced,
        positive when one colour dominates.
    """
    sa = np.asarray(sa_fields, dtype=np.float32)
    if sa.ndim == 1:
        n3 = sa.size // 3
        n = round(n3 ** (1 / 3))
        sa = sa.reshape(3, n, n, n)
    if sa.ndim != 4 or sa.shape[0] != 3:
        raise ValueError(f"sa_fields must have shape (3, N, N, N), got {sa.shape}")

    sa_sq_sum = sa[0] ** 2 + sa[1] ** 2 + sa[2] ** 2
    sa_sum_sq = (sa[0] + sa[1] + sa[2]) ** 2
    return (sa_sq_sum - sa_sum_sq / 3.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Colour-current variance  (CCV)
# ---------------------------------------------------------------------------


def color_current_variance(
    psi_r: NDArray,
    psi_i: NDArray,
    dx: float = 1.0,
) -> NDArray[np.float32]:
    """Compute the colour-current variance CCV from the Ψ_a fields.

    CCV = Σ_d [ Σ_a j²_{a,d} − (1/3)(Σ_a j_{a,d})² ]

    where  j_{a,d} = Im(Ψ_a* ∂_d Ψ_a) is the d-th component of the
    Noether current for colour a.

    Uses second-order central differences for ∂_d Ψ_a.

    Parameters
    ----------
    psi_r : ndarray, shape (3, N, N, N)
        Real parts of the 3-colour field.
    psi_i : ndarray, shape (3, N, N, N)
        Imaginary parts.
    dx : float
        Lattice spacing (default 1.0 = lattice units).

    Returns
    -------
    ccv : ndarray, shape (N, N, N), float32
    """
    psi_r = np.asarray(psi_r, dtype=np.float32)
    psi_i = np.asarray(psi_i, dtype=np.float32)
    if psi_r.ndim != 4 or psi_r.shape[0] != 3:
        raise ValueError("psi_r must have shape (3, N, N, N)")

    ccv = np.zeros(psi_r.shape[1:], dtype=np.float32)
    axes = [1, 2, 3]  # x, y, z
    for d in axes:
        # Central-difference derivative for each colour
        j = np.empty_like(psi_r)  # j[a] = j_{a,d}
        for a in range(3):
            dpr = np.gradient(psi_r[a], dx, axis=d - 1)
            dpi = np.gradient(psi_i[a], dx, axis=d - 1)
            # j_{a,d} = Im(Ψ_a* ∂_d Ψ_a) = Pr * dPi - Pi * dPr
            j[a] = psi_r[a] * dpi - psi_i[a] * dpr

        j_sq_sum = j[0] ** 2 + j[1] ** 2 + j[2] ** 2
        j_sum_sq = (j[0] + j[1] + j[2]) ** 2
        ccv += j_sq_sum - j_sum_sq / 3.0

    return ccv.astype(np.float32)


# ---------------------------------------------------------------------------
# χ midpoint between two colour sources
# ---------------------------------------------------------------------------


def measure_chi_midpoint(
    chi: NDArray,
    pos_a: tuple[int, int, int],
    pos_b: tuple[int, int, int],
    half_width: int = 2,
) -> float:
    """Return the mean χ in a small cube at the midpoint of pos_a→pos_b.

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
        χ field.
    pos_a, pos_b : (i, j, k)
        Grid indices of the two colour sources.
    half_width : int
        Half-side of the averaging cube (default 2 → 5³ cells).

    Returns
    -------
    float
        Mean χ in the midpoint cube.
    """
    chi = np.asarray(chi, dtype=np.float32)
    N = chi.shape[0]

    mx = (pos_a[0] + pos_b[0]) // 2
    my = (pos_a[1] + pos_b[1]) // 2
    mz = (pos_a[2] + pos_b[2]) // 2

    xi = np.clip(mx - half_width, 0, N)
    xf = np.clip(mx + half_width + 1, 0, N)
    yi = np.clip(my - half_width, 0, N)
    yf = np.clip(my + half_width + 1, 0, N)
    zi = np.clip(mz - half_width, 0, N)
    zf = np.clip(mz + half_width + 1, 0, N)

    region = chi[xi:xf, yi:yf, zi:zf]
    return float(region.mean())


# ---------------------------------------------------------------------------
# Flux-tube radial profile
# ---------------------------------------------------------------------------


def flux_tube_profile(
    chi: NDArray,
    scv: NDArray,
    pos_a: tuple[int, int, int],
    pos_b: tuple[int, int, int],
    n_bins: int = 20,
    tube_radius: float = 8.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute azimuthal averages of χ and SCV around the axis pos_a→pos_b.

    Samples lattice points within `tube_radius` of the axis, bins them
    by perpendicular distance, and returns means.

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
    scv : ndarray, shape (N, N, N)
    pos_a, pos_b : grid-index tuples
    n_bins : int
        Number of radial bins.
    tube_radius : float
        Maximum perpendicular distance to include.

    Returns
    -------
    r_bins : ndarray, shape (n_bins,)
        Bin centres (perpendicular distance from axis).
    chi_profile : ndarray, shape (n_bins,)
        Mean χ per bin.
    scv_profile : ndarray, shape (n_bins,)
        Mean SCV per bin.
    """
    chi = np.asarray(chi, dtype=np.float32)
    scv = np.asarray(scv, dtype=np.float32)
    N = chi.shape[0]

    # Axis unit vector
    ax = np.array([pos_b[0] - pos_a[0], pos_b[1] - pos_a[1], pos_b[2] - pos_a[2]], dtype=float)
    length = np.linalg.norm(ax)
    if length < 1e-10:
        raise ValueError("pos_a and pos_b must be distinct.")
    ax /= length

    # Build coordinate arrays
    idx = np.arange(N, dtype=np.float32)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    rx = X - pos_a[0]
    ry = Y - pos_a[1]
    rz = Z - pos_a[2]

    # Project off axis
    proj = rx * ax[0] + ry * ax[1] + rz * ax[2]
    perp_x = rx - proj * ax[0]
    perp_y = ry - proj * ax[1]
    perp_z = rz - proj * ax[2]
    r_perp = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

    # Keep points that are (a) close to the axis and (b) between the sources
    inside = (r_perp <= tube_radius) & (proj >= 0) & (proj <= length)

    r_vals = r_perp[inside]
    chi_vals = chi[inside]
    scv_vals = scv[inside]

    r_bins = np.linspace(0.0, tube_radius, n_bins + 1)
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    chi_profile = np.zeros(n_bins, dtype=np.float32)
    scv_profile = np.zeros(n_bins, dtype=np.float32)

    for b in range(n_bins):
        mask = (r_vals >= r_bins[b]) & (r_vals < r_bins[b + 1])
        if mask.sum() > 0:
            chi_profile[b] = chi_vals[mask].mean()
            scv_profile[b] = scv_vals[mask].mean()

    return r_centres.astype(np.float32), chi_profile, scv_profile


# ---------------------------------------------------------------------------
# String tension extraction
# ---------------------------------------------------------------------------


def string_tension(
    separations: NDArray,
    chi_midpoints: NDArray,
    chi0: float = 19.0,
    r_min_frac: float = 0.25,
) -> tuple[float, float, dict]:
    """Estimate the LFM string tension σ from χ midpoint vs separation data.

    Fits  Δχ(r) = chi0 − χ_mid(r) = σ·r + offset  over the *large-r*
    portion of the data (r ≥ r_min_frac × r_max).

    A positive slope σ > 0 means χ drops further as sources separate —
    the hallmark of linear confinement.

    Parameters
    ----------
    separations : 1-D array of floats
        Source–source separations (lattice cells).
    chi_midpoints : 1-D array of floats
        χ value at the midpoint for each separation.
    chi0 : float
        Vacuum χ (default 19).
    r_min_frac : float
        Fraction of max separation below which to ignore (short-range artefacts).

    Returns
    -------
    sigma : float
        Fitted slope  dΔχ/dr  (positive ↔ confinement).
    intercept : float
        Fitted intercept.
    info : dict
        ``r2``, ``n_points``, ``r_min``, ``fit_separations``, ``fit_dchi``.
    """
    sep = np.asarray(separations, dtype=float)
    chi_mid = np.asarray(chi_midpoints, dtype=float)
    dchi = chi0 - chi_mid

    r_max = sep.max()
    r_min = r_min_frac * r_max
    mask = sep >= r_min

    r_fit = sep[mask]
    d_fit = dchi[mask]
    n = mask.sum()

    if n < 2:
        return 0.0, float(dchi.mean()), {"r2": 0.0, "n_points": int(n)}

    # Linear regression
    r_mean = r_fit.mean()
    d_mean = d_fit.mean()
    slope = np.sum((r_fit - r_mean) * (d_fit - d_mean)) / (np.sum((r_fit - r_mean) ** 2) + 1e-30)
    intercept = d_mean - slope * r_mean

    residuals = d_fit - (slope * r_fit + intercept)
    ss_res = float((residuals**2).sum())
    ss_tot = float(((d_fit - d_mean) ** 2).sum()) + 1e-30
    r2 = 1.0 - ss_res / ss_tot

    return (
        float(slope),
        float(intercept),
        {
            "r2": r2,
            "n_points": int(n),
            "r_min": float(r_min),
            "fit_separations": r_fit.tolist(),
            "fit_dchi": d_fit.tolist(),
        },
    )
