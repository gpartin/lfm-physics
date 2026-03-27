"""
Soliton Construction
====================

Create Gaussian soliton blobs on the LFM lattice.

A soliton is a localized Gaussian envelope: amp * exp(-r²/2σ²),
optionally with a complex phase θ (charge from EM interference).

Production patterns from universe_simulator and primordial_soup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gaussian_soliton(
    N: int,
    position: tuple[float, float, float],
    amplitude: float,
    sigma: float,
    phase: float = 0.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Create a single Gaussian soliton on an N³ grid.

    Parameters
    ----------
    N : int
        Grid size per axis.
    position : tuple of float
        (x, y, z) center in grid coordinates.
    amplitude : float
        Peak amplitude of the envelope.
    sigma : float
        Gaussian width (in grid cells).
    phase : float
        Complex phase θ in radians. θ=0 → negative charge (electron),
        θ=π → positive charge (positron).

    Returns
    -------
    psi_r, psi_i : ndarray of float32, shape (N, N, N)
        Real and imaginary parts: amplitude * exp(-r²/2σ²) * e^(iθ).
    """
    x = np.arange(N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    px, py, pz = position
    r2 = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
    envelope = amplitude * np.exp(-r2 / (2.0 * sigma**2))

    psi_r = (envelope * np.cos(phase)).astype(np.float32)
    psi_i = (envelope * np.sin(phase)).astype(np.float32)
    return psi_r, psi_i


def place_solitons(
    N: int,
    positions: list[tuple[float, float, float]],
    amplitude: float,
    sigma: float,
    phases: NDArray[np.floating] | list[float] | None = None,
    colors: list[int] | None = None,
    n_colors: int = 3,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Place multiple solitons on a 3-color grid (Level 2).

    Parameters
    ----------
    N : int
        Grid size per axis.
    positions : list of (x, y, z) tuples
        Center positions in grid units.
    amplitude : float
        Peak amplitude for all solitons.
    sigma : float
        Gaussian width in grid cells.
    phases : array-like or None
        Phase per soliton. If None, all zero (same charge).
    colors : list of int or None
        Color index per soliton. If None, round-robin assignment.
    n_colors : int
        Number of color components (default 3).

    Returns
    -------
    psi_r, psi_i : ndarray of float32, shape (n_colors, N, N, N)
        Real and imaginary parts of the 3-color field.
    """
    n_solitons = len(positions)
    if phases is None:
        phases = np.zeros(n_solitons)
    phases = np.asarray(phases, dtype=np.float64)

    if colors is None:
        colors = [i % n_colors for i in range(n_solitons)]

    psi_r = np.zeros((n_colors, N, N, N), dtype=np.float32)
    psi_i = np.zeros((n_colors, N, N, N), dtype=np.float32)

    x = np.arange(N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    for idx, (px, py, pz) in enumerate(positions):
        c = colors[idx]
        r2 = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
        envelope = amplitude * np.exp(-r2 / (2.0 * sigma**2))
        psi_r[c] += envelope * np.cos(phases[idx])
        psi_i[c] += envelope * np.sin(phases[idx])

    return psi_r.astype(np.float32), psi_i.astype(np.float32)


def wave_kick(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
    chi: NDArray[np.float32],
    dt: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply a wave kick to create the time-shifted previous state.

    This pre-shifts Ψ(t=-dt) = Ψ(t=0) * e^(+iχdt) to activate the
    imaginary component from t=0, giving proper wave oscillation.

    Parameters
    ----------
    psi_r, psi_i : ndarray of float32
        Current real/imaginary fields (any shape).
    chi : ndarray of float32
        Current χ field (must broadcast with psi).
    dt : float
        Timestep.

    Returns
    -------
    psi_r_prev, psi_i_prev : ndarray of float32
        Time-shifted previous state.
    """
    chi_dt = chi * dt
    cos_term = np.cos(chi_dt)
    sin_term = np.sin(chi_dt)

    psi_r_prev = (psi_r * cos_term - psi_i * sin_term).astype(np.float32)
    psi_i_prev = (psi_i * cos_term + psi_r * sin_term).astype(np.float32)
    return psi_r_prev, psi_i_prev
