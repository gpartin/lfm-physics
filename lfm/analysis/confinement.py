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

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from lfm.particles.solver import SolitonSolution

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


# ---------------------------------------------------------------------------
# Potential shape fitting  (Yukawa / Coulomb / Cornell)
# ---------------------------------------------------------------------------


def fit_yukawa(
    r: NDArray,
    V_r: NDArray,
) -> tuple[float, float, float, dict]:
    """Fit V(r) = A * exp(-m * r) / r  (screened Coulomb / Yukawa).

    Expected for LFM with mass gap χ₀ = 19: screening mass m ≈ χ₀ = 19
    in lattice units, ξ = 1/m ≈ 0.053 cells.

    Parameters
    ----------
    r : 1-D array of floats
        Separations (lattice units).
    V_r : 1-D array of floats
        Potential values V(r).

    Returns
    -------
    A, m, r2, info : float, float, float, dict
        Amplitude, screening mass, coefficient of determination, and
        ``info`` dict with ``popt``, ``pcov``, ``residuals``.
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V_r, dtype=float)

    if len(r) < 3:
        return 0.0, 0.0, 0.0, {"error": "not enough points"}

    def _model(x, A, m):
        return A * np.exp(-m * x) / (x + 1e-30)

    V_sign = float(np.sign(V[np.abs(V).argmax()])) or 1.0
    p0 = [V_sign * float(np.abs(V * r).mean()), 0.5]
    bounds = ([-np.inf, 1e-6], [np.inf, np.inf])

    try:
        popt, pcov = curve_fit(_model, r, V, p0=p0, bounds=bounds, maxfev=5000)
        A_fit, m_fit = float(popt[0]), float(popt[1])
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, {"error": "fit failed"}

    residuals = V - _model(r, *popt)
    ss_res = float((residuals**2).sum())
    ss_tot = float(((V - V.mean()) ** 2).sum()) + 1e-30
    r2 = max(0.0, 1.0 - ss_res / ss_tot)

    return (
        A_fit,
        m_fit,
        r2,
        {"popt": popt.tolist(), "pcov": pcov.tolist(), "residuals": residuals.tolist()},
    )


def fit_coulomb(
    r: NDArray,
    V_r: NDArray,
) -> tuple[float, float, float, dict]:
    """Fit V(r) = A / r + B  (pure Coulomb / massless gauge boson).

    Expected if LFM colour interaction is mediated by a massless field,
    like QED or tree-level gluon exchange.

    Parameters
    ----------
    r, V_r : 1-D arrays of floats

    Returns
    -------
    A, B, r2, info : float, float, float, dict
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V_r, dtype=float)

    if len(r) < 2:
        return 0.0, 0.0, 0.0, {"error": "not enough points"}

    # Linear regression on V vs 1/r
    inv_r = 1.0 / (r + 1e-30)
    X = np.column_stack([inv_r, np.ones_like(inv_r)])
    try:
        coeffs, residuals_arr, _, _ = np.linalg.lstsq(X, V, rcond=None)
        A_fit, B_fit = float(coeffs[0]), float(coeffs[1])
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, {"error": "fit failed"}

    V_pred = A_fit * inv_r + B_fit
    residuals = V - V_pred
    ss_res = float((residuals**2).sum())
    ss_tot = float(((V - V.mean()) ** 2).sum()) + 1e-30
    r2 = max(0.0, 1.0 - ss_res / ss_tot)

    return A_fit, B_fit, r2, {"residuals": residuals.tolist()}


def fit_cornell(
    r: NDArray,
    V_r: NDArray,
) -> tuple[float, float, float, float, dict]:
    """Fit V(r) = -A/r + sigma*r + C  (Cornell potential — QCD confinement).

    Expected if LFM generates linear flux tubes (string tension σ > 0).
    The standard QCD Cornell potential combines one-gluon exchange (−A/r)
    with a linear confining term (σ·r).

    Parameters
    ----------
    r, V_r : 1-D arrays of floats

    Returns
    -------
    A, sigma, C, r2, info : float, float, float, float, dict
        Coulomb coefficient, string tension, offset, R², and info dict.
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V_r, dtype=float)

    if len(r) < 3:
        return 0.0, 0.0, 0.0, 0.0, {"error": "not enough points"}

    def _model(x, A, sigma, C):
        return -A / (x + 1e-30) + sigma * x + C

    p0 = [1.0, 0.1, float(V.mean())]

    try:
        popt, pcov = curve_fit(_model, r, V, p0=p0, maxfev=5000)
        A_fit, sigma_fit, C_fit = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, 0.0, {"error": "fit failed"}

    residuals = V - _model(r, *popt)
    ss_res = float((residuals**2).sum())
    ss_tot = float(((V - V.mean()) ** 2).sum()) + 1e-30
    r2 = max(0.0, 1.0 - ss_res / ss_tot)

    return (
        A_fit,
        sigma_fit,
        C_fit,
        r2,
        {"popt": popt.tolist(), "pcov": pcov.tolist(), "residuals": residuals.tolist()},
    )


def classify_potential(
    r: NDArray,
    V_r: NDArray,
) -> dict:
    """Classify V(r) as Yukawa, Coulomb, or Cornell by best-fit R².

    Runs all three fits and returns the highest-R² winner, along with
    the full parameter sets for all three models.

    Parameters
    ----------
    r, V_r : 1-D arrays of floats

    Returns
    -------
    dict with keys:
        ``best_fit``   — ``'yukawa'``, ``'coulomb'``, or ``'cornell'``
        ``r2``         — R² of the best fit
        ``yukawa``     — dict with ``A``, ``m``, ``r2``
        ``coulomb``    — dict with ``A``, ``B``, ``r2``
        ``cornell``    — dict with ``A``, ``sigma``, ``C``, ``r2``
        ``verdict``    — brief interpretation string
    """
    A_y, m_y, r2_y, _ = fit_yukawa(r, V_r)
    A_c, B_c, r2_c, _ = fit_coulomb(r, V_r)
    A_cn, sig_cn, C_cn, r2_cn, _ = fit_cornell(r, V_r)

    # ------------------------------------------------------------------
    # Cornell confinement guard: Cornell can win on R² even when V(r)
    # remains negative throughout the measured range.  True confinement
    # requires V to actually rise above zero somewhere in the data.
    # If every measured V < 0, the potential is screened (Yukawa-like),
    # not confining.
    # ------------------------------------------------------------------
    cornell_confined = bool(sig_cn > 0 and np.any(V_r > 0))

    candidates = [("yukawa", r2_y), ("coulomb", r2_c)]
    if cornell_confined:
        candidates.append(("cornell", r2_cn))

    best_name, best_r2 = max(candidates, key=lambda x: x[1])
    # Always include cornell in output for reference, even when not selected
    best_r2 = max(best_r2, r2_cn) if best_name == "cornell" else best_r2

    verdicts = {
        "yukawa": f"Yukawa screened (m ≈ {m_y:.3f}): LFM mass-gap screening — NOT confined",
        "coulomb": "Coulomb 1/r: massless-gauge-boson exchange — NOT confined",
        "cornell": f"Cornell (σ ≈ {sig_cn:.4f}): linear confinement — STRING TENSION PRESENT",
    }

    return {
        "best_fit": best_name,
        "r2": best_r2,
        "yukawa": {"A": A_y, "m": m_y, "r2": r2_y},
        "coulomb": {"A": A_c, "B": B_c, "r2": r2_c},
        "cornell": {
            "A": A_cn,
            "sigma": sig_cn,
            "C": C_cn,
            "r2": r2_cn,
            "confined": cornell_confined,
        },
        "verdict": verdicts[best_name],
    }


# ---------------------------------------------------------------------------
# Static quark–quark interaction potential  V(r) = E_total(r) − 2·E_self
# ---------------------------------------------------------------------------


def static_interaction_potential(
    sol: "SolitonSolution",
    separations: NDArray,
    axis: int = 0,
    chi0: float = 19.0,
    kappa: float = 1.0 / 63.0,
    c: float = 1.0,
) -> dict:
    """Measure the static two-body interaction potential V(r).

    Uses the Born-Oppenheimer / quenched approximation:

    1. Fix two copies of ``sol`` at separation ``r`` along ``axis``
       by rolling the arrays (periodic-lattice displacement).
    2. Poisson-equilibrate χ from the total |Ψ_A + Ψ_B|².
    3. Compute the total static energy E_total(r).
    4. V(r) = E_total(r) − 2 · E_self

    The potential shape distinguishes the three LFM interpretations:

    * **Yukawa**  V ~ A·e^{−m·r}/r  ← LFM prediction (screened by mass gap χ₀)
    * **Coulomb** V ~ A/r            ← massless-gluon exchange
    * **Cornell** V ~ −A/r + σ·r    ← QCD linear confinement

    Parameters
    ----------
    sol : SolitonSolution
        Solved single-particle eigenmode.  Must have ``psi_r``, ``chi``,
        and optionally ``psi_i`` arrays of shape (N, N, N).
    separations : 1-D array-like of ints
        Centre-to-centre separations in lattice cells (must each be < N).
    axis : int
        Spatial axis along which to separate the particles (0 = x).
    chi0 : float
        Background χ (default 19).
    kappa : float
        Coupling constant (default 1/63).
    c : float
        Wave speed (default 1.0 lattice units).

    Returns
    -------
    dict with keys:
        ``r``         — separation array (float)
        ``V_r``       — interaction potential at each r (float array)
        ``E_total``   — total two-body energy at each r
        ``E_self``    — single-particle self-energy (scalar)
        ``N``         — lattice size used
    """
    from lfm.analysis.energy import total_energy as _total_energy
    from lfm.fields.equilibrium import equilibrate_chi

    psi_r = np.asarray(sol.psi_r, dtype=np.float64)
    psi_i = np.asarray(sol.psi_i, dtype=np.float64) if sol.psi_i is not None else None
    chi_sol = np.asarray(sol.chi, dtype=np.float32)

    # Determine if this is a multi-colour (Level 2) particle
    # Level 2: psi_r.shape = (3, N, N, N) → roll axis = axis+1
    # Level 0/1: psi_r.shape = (N, N, N) → roll axis = axis
    multi_color = psi_r.ndim == 4
    roll_axis = axis + 1 if multi_color else axis
    N = int(psi_r.shape[roll_axis])

    # E_self: static energy of the isolated soliton (zero KE → prev = current)
    E_self = _total_energy(
        psi_r, psi_r, chi_sol.astype(np.float64), dt=1.0, c=c, psi_i=psi_i, psi_i_prev=psi_i
    )

    seps = np.asarray(separations, dtype=int)
    V_arr = np.zeros(len(seps), dtype=np.float64)
    E_total_arr = np.zeros(len(seps), dtype=np.float64)

    for i, r in enumerate(seps):
        r = int(r)
        half = r // 2

        if r >= N:
            V_arr[i] = np.nan
            E_total_arr[i] = np.nan
            continue

        # Roll copies along chosen axis: A left, B right
        psi_A_r = np.roll(psi_r, -half, axis=roll_axis)
        psi_B_r = np.roll(psi_r, +half, axis=roll_axis)
        psi_tot_r = psi_A_r + psi_B_r

        psi_tot_i = None
        if psi_i is not None:
            psi_A_i = np.roll(psi_i, -half, axis=roll_axis)
            psi_B_i = np.roll(psi_i, +half, axis=roll_axis)
            psi_tot_i = psi_A_i + psi_B_i

        # Total |Ψ|² collapsed to (N,N,N) for the Poisson solver
        psi_sq = psi_tot_r**2
        if psi_tot_i is not None:
            psi_sq = psi_sq + psi_tot_i**2
        if psi_sq.ndim == 4:  # multi-colour: sum over colour axis
            psi_sq = psi_sq.sum(axis=0)

        chi_eq = equilibrate_chi(psi_sq.astype(np.float32), chi0=chi0, kappa=kappa)

        # Static energy (zero KE: prev = current)
        E_tot = _total_energy(
            psi_tot_r,
            psi_tot_r,
            chi_eq.astype(np.float64),
            dt=1.0,
            c=c,
            psi_i=psi_tot_i,
            psi_i_prev=psi_tot_i,
        )
        E_total_arr[i] = E_tot
        V_arr[i] = E_tot - 2.0 * E_self

    return {
        "r": seps.astype(float),
        "V_r": V_arr,
        "E_total": E_total_arr,
        "E_self": E_self,
        "N": N,
    }
