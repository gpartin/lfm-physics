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


def fluid_fields(
    psi_r: NDArray[np.floating],
    psi_r_prev: NDArray[np.floating],
    chi: NDArray[np.floating],
    dt: float,
    c: float = 1.0,
    psi_i: NDArray[np.floating] | None = None,
    psi_i_prev: NDArray[np.floating] | None = None,
) -> dict:
    """Compute fluid-dynamics observables from the GOV-01 stress-energy tensor.

    Uses the covariant stress-energy approach (not Klein-Gordon charge current)
    so that the energy density ε is always positive even with random phases::

        ε = ½[(∂Ψ/∂t)² + c²|∇Ψ|² + χ²|Ψ|²]
        g = −Re[(∂Ψ*/∂t) ∇Ψ]     (energy flux / momentum density)
        v = g / ε                  (fluid velocity)
        P = ½c²|∇Ψ|²              (pressure)

    Parameters
    ----------
    psi_r : ndarray (N,N,N)
        Real part of Ψ at the current step.
    psi_r_prev : ndarray (N,N,N)
        Real part of Ψ at the previous step.
    chi : ndarray (N,N,N)
        χ field at the current step.
    dt : float
        Timestep used in the simulation.
    c : float
        Wave speed (default 1.0).
    psi_i : ndarray or None
        Imaginary part of Ψ at the current step.
    psi_i_prev : ndarray or None
        Imaginary part of Ψ at the previous step.

    Returns
    -------
    dict with keys:
        'epsilon'        — 3D energy density field
        'gx','gy','gz'   — 3D energy-flux (momentum density) components
        'vx','vy','vz'   — 3D fluid velocity components
        'pressure'       — 3D pressure field (gradient energy density)
        'epsilon_mean'   — mean energy density (scalar)
        'pressure_mean'  — mean pressure (scalar)
        'v_rms'          — RMS fluid speed (scalar)
    """
    psi_r64 = psi_r.astype(np.float64)
    psi_r_prev64 = psi_r_prev.astype(np.float64)
    chi64 = chi.astype(np.float64)
    c2 = c ** 2

    dpsr_dt = (psi_r64 - psi_r_prev64) / dt

    has_imag = psi_i is not None and psi_i_prev is not None
    if has_imag:
        psi_i64 = psi_i.astype(np.float64)
        psi_i_prev64 = psi_i_prev.astype(np.float64)
        dpsi_dt = (psi_i64 - psi_i_prev64) / dt
    else:
        psi_i64 = np.zeros_like(psi_r64)
        dpsi_dt = np.zeros_like(dpsr_dt)

    def _grad(f: NDArray, ax: int) -> NDArray:
        return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / 2.0

    # Energy density
    ke = 0.5 * (dpsr_dt ** 2 + dpsi_dt ** 2)
    grad_sq = sum(_grad(psi_r64, ax) ** 2 + _grad(psi_i64, ax) ** 2 for ax in range(3))
    ge = 0.5 * c2 * grad_sq
    psi_sq = psi_r64 ** 2 + psi_i64 ** 2
    pot = 0.5 * chi64 ** 2 * psi_sq
    epsilon = ke + ge + pot

    # Energy flux
    gx = -(dpsr_dt * _grad(psi_r64, 0) + dpsi_dt * _grad(psi_i64, 0))
    gy = -(dpsr_dt * _grad(psi_r64, 1) + dpsi_dt * _grad(psi_i64, 1))
    gz = -(dpsr_dt * _grad(psi_r64, 2) + dpsi_dt * _grad(psi_i64, 2))

    # Velocity (ε is always ≥ 0; protect against division by ~0)
    eps_safe = np.where(epsilon > 0, epsilon, np.finfo(np.float64).tiny)
    vx = gx / eps_safe
    vy = gy / eps_safe
    vz = gz / eps_safe

    return {
        "epsilon": epsilon,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "pressure": ge,
        "epsilon_mean": float(np.mean(epsilon)),
        "pressure_mean": float(np.mean(ge)),
        "v_rms": float(np.sqrt(np.mean(vx ** 2 + vy ** 2 + vz ** 2))),
    }


def continuity_residual(
    eps_t0: NDArray[np.floating],
    eps_t1: NDArray[np.floating],
    gx: NDArray[np.floating],
    gy: NDArray[np.floating],
    gz: NDArray[np.floating],
    dt: float,
) -> float:
    """RMS residual of the energy continuity equation ∂ε/∂t + ∇·g = 0,
    normalised by mean |ε|.

    A value near 0 confirms that the LFM stress-energy tensor satisfies
    the Euler equation for fluid dynamics.

    Parameters
    ----------
    eps_t0, eps_t1 : ndarray (N,N,N)
        Energy density at consecutive steps.
    gx, gy, gz : ndarray (N,N,N)
        Energy-flux components from :func:`fluid_fields` at step t0.
    dt : float
        Timestep.

    Returns
    -------
    float
        RMS(∂ε/∂t + ∇·g) / mean(|ε|).  Returns NaN if mean ε ≈ 0.
    """
    eps_t0_64 = np.asarray(eps_t0, dtype=np.float64)
    eps_t1_64 = np.asarray(eps_t1, dtype=np.float64)
    deps_dt = (eps_t1_64 - eps_t0_64) / dt

    def _dc(g: NDArray, ax: int) -> NDArray:
        g64 = np.asarray(g, dtype=np.float64)
        return (np.roll(g64, -1, ax) - np.roll(g64, 1, ax)) / 2.0

    div_g = _dc(gx, 0) + _dc(gy, 1) + _dc(gz, 2)
    residual = deps_dt + div_g

    # Normalise by RMS(|∂ε/∂t|) — the natural scale of energy fluctuations.
    # Result ≈ 0 means Euler equation is satisfied; ≈ 1 means it isn't.
    # (Dividing by mean ε would explode in turbulent systems.)
    rms_deps_dt = float(np.sqrt(np.mean(deps_dt ** 2)))
    if rms_deps_dt < 1e-30:
        return float("nan")
    return float(np.sqrt(np.mean(residual ** 2)) / rms_deps_dt)
