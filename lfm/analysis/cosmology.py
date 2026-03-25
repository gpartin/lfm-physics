"""
Cosmological statistics
=======================

Functions for analysing large-scale structure in LFM simulations:

* :func:`correlation_function` — two-point correlation ξ(r)
* :func:`matter_power_spectrum` — dimensionless P(k) of density contrast δ
* :func:`halo_mass_function` — differential and cumulative N(M)
* :func:`void_statistics` — void size distribution

All functions work on plain NumPy arrays and have no dependency on the
:class:`~lfm.Simulation` class so they can be applied to data loaded
from checkpoints or other sources.

References
----------
* Peebles 1980 (large-scale structure textbook)
* Press & Schechter 1974 (halo mass function)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def correlation_function(
    field: NDArray,
    n_bins: int = 32,
    normalize: bool = True,
    chi0: float | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Two-point correlation function ξ(r) via FFT.

    Computes the isotropically binned three-dimensional two-point correlation
    function by Fourier-space multiplication:

    .. math::

        \\xi(r) = \\langle \\delta(\\mathbf{x})\\,\\delta(\\mathbf{x}+\\mathbf{r}) \\rangle

    where δ = (field − mean) / mean is the overdensity.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        Scalar field (e.g. energy density, |Ψ|², or χ).
    n_bins : int
        Number of radial bins.
    normalize : bool
        If True, divide by ⟨δ⟩² (= 1 for zero-mean δ, makes ξ dimensionless).
    chi0 : float or None
        Background value used as the mean when computing δ.  If None, uses
        the spatial mean of *field*.

    Returns
    -------
    dict with keys:
        ``r``  — bin-centre radii in grid units
        ``xi`` — ξ(r) values
        ``n_pairs`` — number of grid-cell pairs in each bin
    """
    f = np.asarray(field, dtype=np.float64)
    mean = float(chi0) if chi0 is not None else float(f.mean())
    if mean == 0.0:
        mean = 1.0  # guard against divide-by-zero
    delta = (f - mean) / mean  # overdensity, shape (N,N,N)

    N = f.shape[0]
    # Power spectrum via FFT
    dk = np.fft.fftn(delta)
    pk_3d = np.abs(dk) ** 2 / N**3  # shape (N,N,N)

    # IFFT → correlation function (Wiener–Khinchin)
    xi_3d = np.fft.ifftn(pk_3d).real  # shape (N,N,N)

    # Radial binning
    idx = np.fft.fftfreq(N, d=1.0 / N).astype(int)  # grid-index frequencies
    ix, iy, iz = np.meshgrid(idx, idx, idx, indexing="ij")
    r = np.sqrt(ix**2 + iy**2 + iz**2).astype(np.float64)

    r_max = N / 2 * np.sqrt(3)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    xi_bins = np.zeros(n_bins)
    n_pairs = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = (r >= bin_edges[b]) & (r < bin_edges[b + 1])
        count = int(mask.sum())
        if count > 0:
            xi_bins[b] = float(xi_3d[mask].mean())
            n_pairs[b] = count

    return {
        "r": bin_centres,
        "xi": xi_bins,
        "n_pairs": n_pairs,
    }


def matter_power_spectrum(
    energy_density: NDArray,
    n_bins: int = 32,
    k_nyquist_fraction: float = 1.0,
) -> dict[str, NDArray[np.float64]]:
    """Dimensionless matter power spectrum Δ²(k) = k³P(k)/(2π²).

    Parameters
    ----------
    energy_density : ndarray, shape (N, N, N)
        Total energy density |Ψ|² (proxy for matter density ρ).
    n_bins : int
        Number of radial k-bins.
    k_nyquist_fraction : float
        Upper k limit as a fraction of the Nyquist frequency.

    Returns
    -------
    dict with keys:
        ``k``   — bin-centre wavenumbers in units of 2π / grid_length
        ``pk``  — dimensionless power Δ²(k)
        ``pk_raw`` — raw P(k) (not dimensionless)
        ``n_modes`` — number of Fourier modes in each bin
    """
    rho = np.asarray(energy_density, dtype=np.float64)
    N = rho.shape[0]
    rho_mean = float(rho.mean())
    if rho_mean == 0.0:
        rho_mean = 1.0

    delta = (rho - rho_mean) / rho_mean  # overdensity

    dk = np.fft.fftn(delta) / N**3  # normalise by volume
    pk_3d = np.abs(dk) ** 2 * N**3  # units: volume

    # Wavenumber grid in units of 2π/N (fundamental mode)
    kf = np.fft.fftfreq(N) * N  # integer grid frequencies
    kx, ky, kz = np.meshgrid(kf, kf, kf, indexing="ij")
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    k_nyquist = N / 2.0
    k_max = k_nyquist * k_nyquist_fraction
    bin_edges = np.linspace(0.5, k_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pk_bins = np.zeros(n_bins)
    n_modes = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = (k_mag >= bin_edges[b]) & (k_mag < bin_edges[b + 1])
        count = int(mask.sum())
        if count > 0:
            pk_bins[b] = float(pk_3d[mask].mean())
            n_modes[b] = count

    # Dimensionless power: Δ²(k) = k³ P(k) / (2π²)
    k_phys = bin_centres * (2 * np.pi / N)  # physical wavenumber
    delta2 = k_phys**3 * pk_bins / (2 * np.pi**2)

    return {
        "k": bin_centres,
        "pk": delta2,
        "pk_raw": pk_bins,
        "n_modes": n_modes,
    }


def halo_mass_function(
    chi: NDArray,
    energy_density: NDArray,
    chi_threshold: float = 17.0,
    n_bins: int = 20,
) -> dict[str, NDArray[np.float64]]:
    """Differential and cumulative halo mass function N(M).

    Identifies halos as connected regions where χ < *chi_threshold*
    (gravitational wells), sums the energy density within each halo to
    estimate its mass, then bins by mass.

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
        Current χ field.
    energy_density : ndarray, shape (N, N, N)
        |Ψ|² field (proxy for mass).
    chi_threshold : float
        χ value below which a cell is considered part of a halo.
    n_bins : int
        Number of logarithmic mass bins.

    Returns
    -------
    dict with keys:
        ``m_bins`` — bin-centre masses (in units of total grid energy density)
        ``dn_dlnm`` — differential count per ln(M) bin
        ``n_cumulative`` — cumulative count N(>M)
        ``masses`` — raw per-halo masses (unsorted)
        ``n_halos`` — total number of identified halos
    """
    try:
        from scipy.ndimage import label  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "halo_mass_function requires scipy.  Install it: pip install scipy"
        ) from exc

    chi_arr = np.asarray(chi, dtype=np.float64)
    rho = np.asarray(energy_density, dtype=np.float64)

    well_mask = chi_arr < chi_threshold
    labeled, n_halos = label(well_mask)

    if n_halos == 0:
        empty: NDArray[np.float64] = np.array([], dtype=np.float64)
        return {
            "m_bins": empty,
            "dn_dlnm": empty,
            "n_cumulative": empty,
            "masses": empty,
            "n_halos": 0,
        }

    masses = np.zeros(n_halos, dtype=np.float64)
    for h in range(1, n_halos + 1):
        masses[h - 1] = float(rho[labeled == h].sum())

    # Remove zero-mass halos (uninitialised cells)
    masses = masses[masses > 0.0]
    if len(masses) == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "m_bins": empty,
            "dn_dlnm": empty,
            "n_cumulative": empty,
            "masses": empty,
            "n_halos": 0,
        }

    m_min = masses.min()
    m_max = masses.max()
    if m_min == m_max:
        m_min = m_max * 0.5
    bin_edges = np.logspace(np.log10(m_min), np.log10(m_max), n_bins + 1)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    counts, _ = np.histogram(masses, bins=bin_edges)
    dlnm = np.diff(np.log(bin_edges))
    dn_dlnm = counts.astype(np.float64) / dlnm

    # Cumulative: N(> M)
    masses_sorted = np.sort(masses)[::-1]
    n_cumulative = np.arange(1, len(masses_sorted) + 1, dtype=np.float64)

    return {
        "m_bins": bin_centres,
        "dn_dlnm": dn_dlnm,
        "n_cumulative": n_cumulative,
        "masses": masses,
        "n_halos": int(n_halos),
    }


def void_statistics(
    chi: NDArray,
    chi_min: float = 18.5,
    n_bins: int = 20,
) -> dict[str, NDArray[np.float64]]:
    """Size distribution of underdense void regions (χ ≈ χ₀).

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
        χ field.
    chi_min : float
        Lower χ threshold for void identification (default 18.5).
        Cells with χ ≥ *chi_min* are considered void.
    n_bins : int
        Number of size bins.

    Returns
    -------
    dict with keys:
        ``r_bins`` — bin-centre effective radii in grid cells
        ``dn_dr``  — differential distribution dN/dr
        ``n_voids`` — total number of identified voids
        ``sizes``  — per-void volumes in grid cells³
    """
    try:
        from scipy.ndimage import label  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("void_statistics requires scipy.  Install it: pip install scipy") from exc

    chi_arr = np.asarray(chi, dtype=np.float64)
    void_mask = chi_arr >= chi_min
    labeled, n_voids = label(void_mask)

    if n_voids == 0:
        empty: NDArray[np.float64] = np.array([], dtype=np.float64)
        return {"r_bins": empty, "dn_dr": empty, "n_voids": 0, "sizes": empty}

    sizes = np.array(
        [float((labeled == v).sum()) for v in range(1, n_voids + 1)],
        dtype=np.float64,
    )
    # Effective radius assuming spherical void: V = 4/3 π r³
    r_eff = (3.0 * sizes / (4.0 * np.pi)) ** (1.0 / 3.0)

    r_min, r_max = r_eff.min(), r_eff.max()
    if r_min == r_max:
        r_min = r_max * 0.5
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts, _ = np.histogram(r_eff, bins=bin_edges)
    dr = np.diff(bin_edges)
    dn_dr = counts.astype(np.float64) / dr

    return {
        "r_bins": bin_centres,
        "dn_dr": dn_dr,
        "n_voids": int(n_voids),
        "sizes": sizes,
    }
