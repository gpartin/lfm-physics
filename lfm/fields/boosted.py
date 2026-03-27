"""
Boosted Soliton Construction
=============================

Create solitons with encoded momentum for scattering experiments.

A static soliton has zero net momentum. To set a soliton in motion,
we apply the correct relativistic boost: a phase gradient (for complex
fields) or a time-derivative kick (for real fields).

See CALC-31: the momentum encoding cost f_mom(v) = χ₀(v/c)² must be
supported by the available χ-well depth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.constants import CHI0
from lfm.fields.soliton import gaussian_soliton

if TYPE_CHECKING:
    from numpy.typing import NDArray


def boosted_soliton(
    N: int,
    position: tuple[float, float, float],
    amplitude: float,
    sigma: float,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    phase: float = 0.0,
    chi0: float = CHI0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Create a Gaussian soliton with an initial momentum boost.

    For a complex field (phase ≠ 0 or any charge), the boost is encoded
    as a phase gradient: Ψ → Ψ · exp(i k·r) where k = χ₀ v/c².
    For a real field, the boost is encoded in the time-derivative field:
    Ė = -v·∇E (Galilean approximation valid for v ≪ c).

    Parameters
    ----------
    N : int
        Grid size per axis.
    position : tuple
        (x, y, z) center in grid units.
    amplitude : float
        Peak amplitude.
    sigma : float
        Gaussian width in grid cells.
    velocity : tuple
        (vx, vy, vz) in lattice units per timestep. Must satisfy |v| < c.
    phase : float
        Base complex phase (charge). Default 0.0 (electron).
    chi0 : float
        Background χ value (for computing k from velocity).

    Returns
    -------
    psi_r : ndarray of float32 (N, N, N)
        Real part of boosted soliton.
    psi_i : ndarray of float32 (N, N, N)
        Imaginary part.
    e_dot : ndarray of float32 (N, N, N)
        Time-derivative field (for real-field mode), zero for complex.

    Notes
    -----
    The momentum encoding cost (CALC-31) is f_mom = χ₀(v/c)².
    Ensure amplitude is large enough that the χ-well depth exceeds f_mom.
    """
    vx, vy, vz = velocity
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Build static soliton
    psi_r, psi_i = gaussian_soliton(N, position, amplitude, sigma, phase=phase)

    e_dot = np.zeros((N, N, N), dtype=np.float32)

    if speed < 1e-15:
        return psi_r, psi_i, e_dot

    # Position grid
    x = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # Wavevector from velocity: k = chi0 * v / c²  (natural units c=1)
    kx = chi0 * vx
    ky = chi0 * vy
    kz = chi0 * vz
    kr = kx * X + ky * Y + kz * Z

    is_complex = (abs(phase) > 1e-12) or (np.max(np.abs(psi_i)) > 1e-12)

    if is_complex:
        # Phase-gradient boost: multiply by exp(i k·r)
        cos_kr = np.cos(kr).astype(np.float32)
        sin_kr = np.sin(kr).astype(np.float32)
        new_r = psi_r * cos_kr - psi_i * sin_kr
        new_i = psi_r * sin_kr + psi_i * cos_kr
        return new_r, new_i, e_dot
    else:
        # Real-field boost via time-derivative kick: Ė = -v·∇E
        grad_x = np.gradient(psi_r, axis=0)
        grad_y = np.gradient(psi_r, axis=1)
        grad_z = np.gradient(psi_r, axis=2)
        e_dot = -(vx * grad_x + vy * grad_y + vz * grad_z)
        return psi_r, psi_i, e_dot.astype(np.float32)
