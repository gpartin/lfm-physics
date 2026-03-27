"""
Angular Momentum Analysis
=========================

Compute angular momentum and orbital elements from LFM fields.

Uses the stress-energy tensor momentum density g = -Re(∂ₜΨ* · ∇Ψ)
for energy-based L, which is the correct fluid-level quantity (Session 64).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def angular_momentum_density(
    psi_r: NDArray,
    psi_i: NDArray,
    psi_r_prev: NDArray,
    psi_i_prev: NDArray,
    dt: float,
    center: tuple[float, float, float] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute angular momentum density L = r × g at each lattice point.

    g = -Re(∂ₜΨ* · ∇Ψ)  (stress-energy momentum density)
    L = r × g            (angular momentum density)

    Parameters
    ----------
    psi_r, psi_i : ndarray (N, N, N)
        Current real/imaginary parts.
    psi_r_prev, psi_i_prev : ndarray (N, N, N)
        Previous-step fields (for time derivative via finite difference).
    dt : float
        Timestep.
    center : tuple or None
        Origin for angular momentum. Defaults to grid center.

    Returns
    -------
    Lx, Ly, Lz : ndarray (N, N, N)
        Angular momentum density components.
    """
    N = psi_r.shape[0]
    if center is None:
        center = (N / 2.0, N / 2.0, N / 2.0)

    # Time derivatives
    dpsi_r_dt = (psi_r - psi_r_prev) / dt
    dpsi_i_dt = (psi_i - psi_i_prev) / dt

    # Spatial gradients (central diff, periodic-compatible)
    grad_r = np.stack(np.gradient(psi_r), axis=0)  # (3, N, N, N)
    grad_i = np.stack(np.gradient(psi_i), axis=0)

    # Momentum density g = -Re(∂ₜΨ* · ∇Ψ) = -(dpsi_r_dt * grad_r + dpsi_i_dt * grad_i)
    gx = -(dpsi_r_dt * grad_r[0] + dpsi_i_dt * grad_i[0])
    gy = -(dpsi_r_dt * grad_r[1] + dpsi_i_dt * grad_i[1])
    gz = -(dpsi_r_dt * grad_r[2] + dpsi_i_dt * grad_i[2])

    # Position vectors relative to center
    x = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rx = X - center[0]
    ry = Y - center[1]
    rz = Z - center[2]

    # L = r × g
    Lx = ry * gz - rz * gy
    Ly = rz * gx - rx * gz
    Lz = rx * gy - ry * gx

    return Lx, Ly, Lz


def total_angular_momentum(
    psi_r: NDArray,
    psi_i: NDArray,
    psi_r_prev: NDArray,
    psi_i_prev: NDArray,
    dt: float,
    center: tuple[float, float, float] | None = None,
    mask: NDArray | None = None,
) -> tuple[float, float, float]:
    """Compute total angular momentum (integrated over lattice).

    Parameters
    ----------
    psi_r, psi_i : ndarray (N, N, N)
        Current field.
    psi_r_prev, psi_i_prev : ndarray (N, N, N)
        Previous step.
    dt : float
        Timestep.
    center : tuple or None
        Origin for angular momentum.
    mask : ndarray of bool or None
        If provided, integrate only over True voxels.

    Returns
    -------
    Lx, Ly, Lz : float
        Integrated angular momentum components.
    """
    Lx, Ly, Lz = angular_momentum_density(psi_r, psi_i, psi_r_prev, psi_i_prev, dt, center)
    if mask is not None:
        Lx = Lx[mask]
        Ly = Ly[mask]
        Lz = Lz[mask]
    return float(np.sum(Lx)), float(np.sum(Ly)), float(np.sum(Lz))


def precession_rate(
    L_history: list[tuple[float, float, float]],
    dt_between: float,
) -> float:
    """Estimate orbital precession rate from angular momentum history.

    Fits the precession angle of the L-vector projection in the L_x–L_y plane.

    Parameters
    ----------
    L_history : list of (Lx, Ly, Lz) tuples
        Sequential angular momentum measurements.
    dt_between : float
        Time interval between successive measurements.

    Returns
    -------
    omega_prec : float
        Precession angular frequency (rad per time unit).
        Returns 0.0 if fewer than 3 data points.
    """
    if len(L_history) < 3:
        return 0.0
    angles = [np.arctan2(ly, lx) for lx, ly, _ in L_history]
    angles = np.unwrap(angles)
    n = len(angles)
    t = np.arange(n) * dt_between
    # Linear fit: angle = omega * t + phi
    coeffs = np.polyfit(t, angles, 1)
    return float(coeffs[0])
