"""
Energy Diagnostics
==================

Three-component energy decomposition for LFM fields:
    T = ½(∂Ψ/∂t)²    — kinetic energy density
    G = ½c²|∇Ψ|²     — gradient energy density
    V = ½χ²|Ψ|²      — potential energy density

Total energy H = ∫(T + G + V) d³x.

Production patterns from exp_sm_02_complex_em_interaction.py and
lfm_energy_conservation_test.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def energy_components(
    psi_r: NDArray[np.floating],
    psi_r_prev: NDArray[np.floating],
    chi: NDArray[np.floating],
    dt: float,
    c: float = 1.0,
    psi_i: NDArray[np.floating] | None = None,
    psi_i_prev: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute three-component energy density fields.

    Supports all field levels:
    - Real E: psi_r shape (N,N,N), psi_i=None
    - Complex Ψ: psi_r & psi_i shape (N,N,N)
    - 3-color Ψₐ: psi_r & psi_i shape (n_colors, N, N, N)

    Parameters
    ----------
    psi_r : ndarray
        Real part of Ψ, current step.
    psi_r_prev : ndarray
        Real part of Ψ, previous step.
    chi : ndarray, shape (N, N, N)
        χ field at current step.
    dt : float
        Timestep for finite-difference time derivative.
    c : float
        Wave speed (default 1.0).
    psi_i : ndarray or None
        Imaginary part of Ψ, current step.
    psi_i_prev : ndarray or None
        Imaginary part, previous step.

    Returns
    -------
    kinetic, gradient, potential : ndarray of float64, shape (N, N, N)
        The three energy density components.
    """
    # Time derivative via finite difference
    dpsi_r_dt = (psi_r.astype(np.float64) - psi_r_prev.astype(np.float64)) / dt

    # Kinetic: ½(∂Ψ/∂t)²
    if psi_r.ndim == 4:  # color: (n_colors, N, N, N)
        kinetic = 0.5 * np.sum(dpsi_r_dt**2, axis=0)
    else:
        kinetic = 0.5 * dpsi_r_dt**2

    if psi_i is not None and psi_i_prev is not None:
        dpsi_i_dt = (psi_i.astype(np.float64) - psi_i_prev.astype(np.float64)) / dt
        if psi_i.ndim == 4:
            kinetic += 0.5 * np.sum(dpsi_i_dt**2, axis=0)
        else:
            kinetic += 0.5 * dpsi_i_dt**2

    # Gradient: ½c²|∇Ψ|²
    c2 = c**2

    def _grad_sq(f: NDArray) -> NDArray:
        """Sum of squared gradients over spatial axes."""
        if f.ndim == 4:  # (n_colors, N, N, N) → grad each color, sum
            total = np.zeros(f.shape[1:], dtype=np.float64)
            for a in range(f.shape[0]):
                for ax in range(3):
                    g = np.gradient(f[a].astype(np.float64), axis=ax)
                    total += g**2
            return total
        total = np.zeros(f.shape, dtype=np.float64)
        for ax in range(3):
            g = np.gradient(f.astype(np.float64), axis=ax)
            total += g**2
        return total

    gradient = 0.5 * c2 * _grad_sq(psi_r)
    if psi_i is not None:
        gradient += 0.5 * c2 * _grad_sq(psi_i)

    # Potential: ½χ²|Ψ|²
    chi64 = chi.astype(np.float64)
    if psi_r.ndim == 4:
        psi_sq = np.sum(psi_r.astype(np.float64) ** 2, axis=0)
    else:
        psi_sq = psi_r.astype(np.float64) ** 2
    if psi_i is not None:
        if psi_i.ndim == 4:
            psi_sq += np.sum(psi_i.astype(np.float64) ** 2, axis=0)
        else:
            psi_sq += psi_i.astype(np.float64) ** 2

    potential = 0.5 * chi64**2 * psi_sq

    return kinetic, gradient, potential


def total_energy(
    psi_r: NDArray[np.floating],
    psi_r_prev: NDArray[np.floating],
    chi: NDArray[np.floating],
    dt: float,
    c: float = 1.0,
    psi_i: NDArray[np.floating] | None = None,
    psi_i_prev: NDArray[np.floating] | None = None,
) -> float:
    """Compute total integrated energy H = ∫(T + G + V) d³x.

    Parameters are the same as :func:`energy_components`.

    Returns
    -------
    float
        Scalar total energy (sum over all grid points).
    """
    T, G, V = energy_components(
        psi_r, psi_r_prev, chi, dt, c, psi_i, psi_i_prev
    )
    return float(np.sum(T + G + V))


def energy_conservation_drift(
    e_initial: float,
    e_final: float,
) -> float:
    """Compute percentage energy drift.

    Parameters
    ----------
    e_initial : float
        Energy at start of simulation.
    e_final : float
        Energy at end of simulation.

    Returns
    -------
    float
        |E_final − E_initial| / |E_initial| × 100, or 0 if E_initial ≈ 0.
    """
    if abs(e_initial) < 1e-30:
        return 0.0
    return abs(e_final - e_initial) / abs(e_initial) * 100.0
