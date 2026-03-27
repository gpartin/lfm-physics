"""
Random Field Initialization
============================

Generate random noise fields for parametric resonance seeding,
thermal initialization, and perturbation studies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def seed_noise(
    N: int,
    amplitude: float = 1e-6,
    n_colors: int = 1,
    rng: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate tiny random noise for Ψ real and imaginary parts.

    Used to seed parametric resonance (Mathieu instability) or
    provide initial perturbations for structure formation.

    Parameters
    ----------
    N : int
        Grid size per axis.
    amplitude : float
        RMS amplitude of the noise (default 1e-6).
    n_colors : int
        Number of color components (1 or 3).
    rng : Generator, int, or None
        Random number generator or seed.

    Returns
    -------
    psi_r, psi_i : ndarray of float32
        Random noise fields.  Shape (N, N, N) if n_colors=1,
        else (n_colors, N, N, N).
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    shape = (N, N, N) if n_colors == 1 else (n_colors, N, N, N)

    psi_r = (rng.standard_normal(shape) * amplitude).astype(np.float32)
    psi_i = (rng.standard_normal(shape) * amplitude).astype(np.float32)
    return psi_r, psi_i


def uniform_chi(N: int, chi0: float = 19.0) -> NDArray[np.float32]:
    """Create a uniform χ field at the vacuum value.

    Parameters
    ----------
    N : int
        Grid size per axis.
    chi0 : float
        Background χ value.

    Returns
    -------
    ndarray of float32, shape (N, N, N)
    """
    return np.full((N, N, N), chi0, dtype=np.float32)
