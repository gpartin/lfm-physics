"""Collective kinematics readouts for source-free LFM field states.

The functions in this module are external-grid diagnostics. They do not alter
live evolution and they do not insert a particle trajectory or force law.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares

from lfm.constants import CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import laplacian_19pt


@njit(inline="always")
def _gradient19_at(
    field: np.ndarray,
    i: int,
    j: int,
    k: int,
    axis: int,
    inv_two_dx: float,
) -> float:
    size = field.shape[0]
    ip = i + 1 if i + 1 < size else 0
    im = i - 1 if i > 0 else size - 1
    jp = j + 1 if j + 1 < size else 0
    jm = j - 1 if j > 0 else size - 1
    kp = k + 1 if k + 1 < size else 0
    km = k - 1 if k > 0 else size - 1
    if axis == 0:
        face = field[ip, j, k] - field[im, j, k]
        edges = (
            field[ip, jp, k]
            + field[ip, jm, k]
            + field[ip, j, kp]
            + field[ip, j, km]
            - field[im, jp, k]
            - field[im, jm, k]
            - field[im, j, kp]
            - field[im, j, km]
        )
    elif axis == 1:
        face = field[i, jp, k] - field[i, jm, k]
        edges = (
            field[ip, jp, k]
            + field[im, jp, k]
            + field[i, jp, kp]
            + field[i, jp, km]
            - field[ip, jm, k]
            - field[im, jm, k]
            - field[i, jm, kp]
            - field[i, jm, km]
        )
    else:
        face = field[i, j, kp] - field[i, j, km]
        edges = (
            field[ip, j, kp]
            + field[im, j, kp]
            + field[i, jp, kp]
            + field[i, jm, kp]
            - field[ip, j, km]
            - field[im, j, km]
            - field[i, jp, km]
            - field[i, jm, km]
        )
    return ((1.0 / 3.0) * face + (1.0 / 6.0) * edges) * inv_two_dx


@njit(inline="always")
def _average_gradient19_at(
    current: np.ndarray,
    previous: np.ndarray,
    i: int,
    j: int,
    k: int,
    axis: int,
    inv_two_dx: float,
) -> float:
    return 0.5 * (
        _gradient19_at(current, i, j, k, axis, inv_two_dx)
        + _gradient19_at(previous, i, j, k, axis, inv_two_dx)
    )


@njit(parallel=True, cache=True)
def _time_centered_momentum19(
    psi_real: np.ndarray,
    psi_real_prev: np.ndarray,
    psi_imag: np.ndarray,
    psi_imag_prev: np.ndarray,
    chi: np.ndarray,
    chi_prev: np.ndarray,
    dt: float,
    dx: float,
    b_chi: float,
) -> np.ndarray:
    size = chi.shape[0]
    total = size * size * size
    inv_dt = 1.0 / dt
    inv_two_dx = 1.0 / (2.0 * dx)
    px = 0.0
    py = 0.0
    pz = 0.0
    for index in prange(total):
        i = index // (size * size)
        j = (index // size) % size
        k = index % size
        local_x = 0.0
        local_y = 0.0
        local_z = 0.0
        for component in range(psi_real.shape[0]):
            rate_r = (psi_real[component, i, j, k] - psi_real_prev[component, i, j, k]) * inv_dt
            rate_i = (psi_imag[component, i, j, k] - psi_imag_prev[component, i, j, k]) * inv_dt
            local_x += rate_r * _average_gradient19_at(
                psi_real[component],
                psi_real_prev[component],
                i,
                j,
                k,
                0,
                inv_two_dx,
            )
            local_x += rate_i * _average_gradient19_at(
                psi_imag[component],
                psi_imag_prev[component],
                i,
                j,
                k,
                0,
                inv_two_dx,
            )
            local_y += rate_r * _average_gradient19_at(
                psi_real[component],
                psi_real_prev[component],
                i,
                j,
                k,
                1,
                inv_two_dx,
            )
            local_y += rate_i * _average_gradient19_at(
                psi_imag[component],
                psi_imag_prev[component],
                i,
                j,
                k,
                1,
                inv_two_dx,
            )
            local_z += rate_r * _average_gradient19_at(
                psi_real[component],
                psi_real_prev[component],
                i,
                j,
                k,
                2,
                inv_two_dx,
            )
            local_z += rate_i * _average_gradient19_at(
                psi_imag[component],
                psi_imag_prev[component],
                i,
                j,
                k,
                2,
                inv_two_dx,
            )
        rate_chi = (chi[i, j, k] - chi_prev[i, j, k]) * inv_dt
        local_x += b_chi * rate_chi * _average_gradient19_at(chi, chi_prev, i, j, k, 0, inv_two_dx)
        local_y += b_chi * rate_chi * _average_gradient19_at(chi, chi_prev, i, j, k, 1, inv_two_dx)
        local_z += b_chi * rate_chi * _average_gradient19_at(chi, chi_prev, i, j, k, 2, inv_two_dx)
        px += -local_x
        py += -local_y
        pz += -local_z
    volume = dx**3
    return np.asarray((px * volume, py * volume, pz * volume))


def time_centered_momentum_19pt(
    psi_real: np.ndarray,
    psi_real_prev: np.ndarray,
    psi_imag: np.ndarray,
    psi_imag_prev: np.ndarray,
    chi: np.ndarray,
    chi_prev: np.ndarray,
    *,
    dt: float,
    dx: float,
    b_chi: float = CHI0 / KAPPA,
) -> np.ndarray:
    """Return the leapfrog-time-centered 19-point collective momentum."""
    arrays = tuple(
        np.ascontiguousarray(value)
        for value in (
            psi_real,
            psi_real_prev,
            psi_imag,
            psi_imag_prev,
            chi,
            chi_prev,
        )
    )
    pr, pp, pi, pip, chi_value, chi_previous = arrays
    if pr.ndim != 4 or pp.shape != pr.shape or pi.shape != pr.shape or pip.shape != pr.shape:
        raise ValueError("complex component arrays must share shape (components,N,N,N)")
    if chi_value.ndim != 3 or chi_previous.shape != chi_value.shape:
        raise ValueError("chi arrays must share shape (N,N,N)")
    if pr.shape[1:] != chi_value.shape:
        raise ValueError("matter and chi spatial shapes differ")
    if dt <= 0.0 or dx <= 0.0 or b_chi <= 0.0:
        raise ValueError("dt, dx, and b_chi must be positive")
    return _time_centered_momentum19(
        pr,
        pp,
        pi,
        pip,
        chi_value,
        chi_previous,
        float(dt),
        float(dx),
        float(b_chi),
    )


def component_noether_charges(
    psi_real: np.ndarray,
    psi_real_prev: np.ndarray,
    psi_imag: np.ndarray,
    psi_imag_prev: np.ndarray,
    *,
    dt: float,
    dx: float,
) -> np.ndarray:
    """Return one leapfrog Noether charge for each complex component."""
    if dt <= 0.0 or dx <= 0.0:
        raise ValueError("dt and dx must be positive")
    arrays = tuple(
        np.asarray(value) for value in (psi_real, psi_real_prev, psi_imag, psi_imag_prev)
    )
    if arrays[0].ndim != 4 or any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("all matter arrays must share shape (components,N,N,N)")
    factor = dx**3 / dt
    charges = []
    for component in range(arrays[0].shape[0]):
        bilinear = (
            arrays[1][component] * arrays[2][component]
            - arrays[3][component] * arrays[0][component]
        )
        charges.append(float(np.sum(bilinear, dtype=np.float64)) * factor)
    return np.asarray(charges, dtype=np.float64)


def flat_octic_hamiltonian_19pt(
    psi_real: np.ndarray,
    psi_real_prev: np.ndarray,
    psi_imag: np.ndarray,
    psi_imag_prev: np.ndarray,
    chi: np.ndarray,
    chi_prev: np.ndarray,
    *,
    dt: float,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
) -> dict[str, float]:
    """Return the phase-space Hamiltonian of the flat-octic C3 system."""
    if dt <= 0.0 or dx <= 0.0 or chi0 <= 0.0 or kappa <= 0.0 or lambda_h <= 0.0:
        raise ValueError("scales and couplings must be positive")
    pr = np.asarray(psi_real)
    pp = np.asarray(psi_real_prev)
    pi = np.asarray(psi_imag)
    pip = np.asarray(psi_imag_prev)
    ch = np.asarray(chi)
    ch_prev = np.asarray(chi_prev)
    if pr.ndim != 4 or pp.shape != pr.shape or pi.shape != pr.shape or pip.shape != pr.shape:
        raise ValueError("complex component arrays must share shape (components,N,N,N)")
    if ch.shape != pr.shape[1:] or ch_prev.shape != ch.shape:
        raise ValueError("chi and matter spatial shapes differ")

    volume = dx**3
    inv_dx2 = 1.0 / dx**2
    b_chi = chi0 / kappa
    matter_temporal = 0.0
    matter_gradient = 0.0
    matter_mass = 0.0
    chi_sq = ch * ch
    for component in range(pr.shape[0]):
        rate_r = (pr[component] - pp[component]) / dt
        rate_i = (pi[component] - pip[component]) / dt
        matter_temporal += (
            0.5 * volume * float(np.sum(rate_r * rate_r + rate_i * rate_i, dtype=np.float64))
        )
        lap_r = laplacian_19pt(pr[component])
        lap_i = laplacian_19pt(pi[component])
        matter_gradient += (
            -0.5
            * volume
            * inv_dx2
            * float(np.sum(pr[component] * lap_r + pi[component] * lap_i, dtype=np.float64))
        )
        matter_mass += (
            0.5
            * volume
            * float(
                np.sum(
                    chi_sq * (pr[component] * pr[component] + pi[component] * pi[component]),
                    dtype=np.float64,
                )
            )
        )
    chi_rate = (ch - ch_prev) / dt
    chi_temporal = 0.5 * b_chi * volume * float(np.sum(chi_rate * chi_rate, dtype=np.float64))
    lap_chi = laplacian_19pt(ch)
    chi_gradient = -0.5 * b_chi * volume * inv_dx2 * float(np.sum(ch * lap_chi, dtype=np.float64))
    delta = chi_sq - chi0**2
    chi_potential = b_chi * lambda_h / chi0**4 * volume * float(np.sum(delta**4, dtype=np.float64))
    total = (
        matter_temporal
        + matter_gradient
        + matter_mass
        + chi_temporal
        + chi_gradient
        + chi_potential
    )
    return {
        "total": total,
        "matter_temporal": matter_temporal,
        "matter_gradient": matter_gradient,
        "matter_mass": matter_mass,
        "chi_temporal": chi_temporal,
        "chi_gradient": chi_gradient,
        "chi_potential": chi_potential,
    }


def fit_offset_power_convergence(
    spacings: np.ndarray,
    errors: np.ndarray,
) -> dict[str, Any]:
    """Fit nonnegative errors to error(h) = offset + coefficient*h**order."""
    h = np.asarray(spacings, dtype=np.float64)
    y = np.asarray(errors, dtype=np.float64)
    if h.ndim != 1 or y.shape != h.shape or h.size < 4:
        raise ValueError("at least four matched one-dimensional samples are required")
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(y)):
        raise ValueError("samples must be finite")
    if np.any(h <= 0.0) or np.any(y < 0.0):
        raise ValueError("spacings must be positive and errors nonnegative")
    scale = max(float(np.max(y)), 1.0e-15)
    initial_offset = max(0.0, min(float(np.min(y)) * 0.25, scale))
    initial_order = 2.0
    initial_coefficient = max(
        (float(np.max(y)) - initial_offset) / float(np.max(h) ** initial_order),
        1.0e-15,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        offset, coefficient, order = parameters
        return (offset + coefficient * h**order - y) / scale

    solution = least_squares(
        residual,
        x0=np.asarray((initial_offset, initial_coefficient, initial_order)),
        bounds=(
            np.asarray((0.0, 0.0, 0.1)),
            np.asarray((2.0 * scale, np.inf, 6.0)),
        ),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=20000,
    )
    offset, coefficient, order = (float(value) for value in solution.x)
    predicted = offset + coefficient * h**order
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = (
        1.0
        if total_sum <= 1.0e-30 and residual_sum <= 1.0e-30
        else 1.0 - residual_sum / max(total_sum, 1.0e-30)
    )
    return {
        "offset": offset,
        "coefficient": coefficient,
        "order": order,
        "r_squared": r_squared,
        "predicted": predicted.tolist(),
        "converged": bool(solution.success),
        "message": str(solution.message),
    }
