"""
Poisson Equilibration
=====================

Solve the GOV-04 quasi-static limit via FFT to create self-consistent
χ wells from a given Ψ field.

    ∇²δχ = (κ/c²)(|Ψ|² − E₀²)  →  χ = χ₀ + δχ

This is CRITICAL for stable simulations: without equilibration,
Gaussian blobs radiate >90% of energy before wells form.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lfm.constants import CHI0, KAPPA


def poisson_solve_fft(
    source: NDArray[np.floating],
    N: int,
) -> NDArray[np.float32]:
    """Solve ∇²φ = source on a periodic N³ grid via FFT.

    Returns φ with DC component = 0 (background = 0).

    Parameters
    ----------
    source : ndarray, shape (N, N, N)
        Right-hand side of the Poisson equation.
    N : int
        Grid size per axis.

    Returns
    -------
    ndarray of float32, shape (N, N, N)
        Solution φ with zero mean.
    """
    src_hat = np.fft.rfftn(source)

    kx = np.fft.fftfreq(N) * 2.0 * np.pi
    ky = np.fft.fftfreq(N) * 2.0 * np.pi
    kz = np.fft.rfftfreq(N) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2

    # Avoid division by zero at DC
    K2[0, 0, 0] = 1.0
    phi_hat = -src_hat / K2
    phi_hat[0, 0, 0] = 0.0

    return np.fft.irfftn(phi_hat, s=(N, N, N), axes=(0, 1, 2)).astype(np.float32)


def equilibrate_chi(
    psi_sq: NDArray[np.floating],
    chi0: float = CHI0,
    kappa: float = KAPPA,
    e0_sq: float = 0.0,
    boundary_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Compute Poisson-equilibrated χ from energy density |Ψ|².

    Solves GOV-04: ∇²δχ = κ(|Ψ|² − E₀²), then χ = χ₀ + δχ.

    Parameters
    ----------
    psi_sq : ndarray, shape (N, N, N)
        Total energy density |Ψ|² (colorblind sum for Level 2).
    chi0 : float
        Background χ value.
    kappa : float
        Coupling constant.
    e0_sq : float
        Background energy density.
    boundary_mask : ndarray of bool or None
        True where boundary is frozen. χ is reset to χ₀ there.

    Returns
    -------
    ndarray of float32, shape (N, N, N)
        Equilibrated χ field.
    """
    N = psi_sq.shape[0]
    rhs = kappa * (psi_sq - e0_sq)
    delta_chi = poisson_solve_fft(rhs, N)
    chi = (chi0 + delta_chi).astype(np.float32)

    if boundary_mask is not None:
        chi[boundary_mask] = chi0

    return chi


def equilibrate_from_fields(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32] | None = None,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    e0_sq: float = 0.0,
    boundary_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Compute equilibrated χ directly from Ψ field components.

    Handles all field levels:
    - Real E: psi_r shape (N,N,N), psi_i=None → |Ψ|² = E²
    - Complex: psi_r/psi_i shape (N,N,N) → |Ψ|² = Pr² + Pi²
    - 3-color: psi_r/psi_i shape (n_colors,N,N,N) → Σₐ(Prₐ² + Piₐ²)

    Parameters
    ----------
    psi_r : ndarray of float32
        Real part of Ψ.
    psi_i : ndarray of float32 or None
        Imaginary part (None for real fields).
    chi0, kappa, e0_sq : float
        Physics parameters.
    boundary_mask : ndarray of bool or None
        Frozen boundary locations.

    Returns
    -------
    ndarray of float32, shape (N, N, N)
        Equilibrated χ field.
    """
    if psi_r.ndim == 3:
        # Single component: (N, N, N)
        psi_sq = psi_r**2
        if psi_i is not None:
            psi_sq = psi_sq + psi_i**2
    elif psi_r.ndim == 4:
        # Multi-color: (n_colors, N, N, N)
        psi_sq = np.sum(psi_r**2, axis=0)
        if psi_i is not None:
            psi_sq = psi_sq + np.sum(psi_i**2, axis=0)
    else:
        raise ValueError(f"Unexpected psi_r shape: {psi_r.shape}")

    return equilibrate_chi(psi_sq, chi0, kappa, e0_sq, boundary_mask)
