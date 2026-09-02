"""Gauss-constrained static U(1) probes for the experimental R5 action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from lfm.foundations.r3_link_frame_live import _link_table
from lfm.foundations.r4_color_static import color_gauss_divergence
from lfm.foundations.r5_unified_live import R5Parameters


@dataclass
class R5U1StaticState:
    """Minimum R5 U(1) electric energy satisfying a periodic Gauss source."""

    potential: np.ndarray
    electric: np.ndarray
    charge: np.ndarray
    gauss_residual: float
    electric_energy: float


def solve_u1_gauss_minimum(
    charge: np.ndarray,
    parameters: R5Parameters = R5Parameters(),
    *,
    tolerance: float = 1.0e-11,
    initial_potential: np.ndarray | None = None,
) -> R5U1StaticState:
    """Minimize the existing R5 U(1) electric energy under Gauss law."""

    charge_values = np.asarray(charge, dtype=np.float64)
    if charge_values.ndim != 3:
        raise ValueError("charge must be a 3D array")
    if abs(float(np.sum(charge_values))) > 1.0e-10:
        raise ValueError("periodic U1 charge must sum to zero")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    unique, _ = _link_table(parameters.stencil)
    shape = charge_values.shape
    count = int(np.prod(shape))

    def electric_from_potential(potential: np.ndarray) -> np.ndarray:
        electric = np.empty(shape + (len(unique),), dtype=np.float64)
        for index, (offset, _) in enumerate(unique):
            neighbor = np.roll(
                potential,
                shift=tuple(-value for value in offset),
                axis=(0, 1, 2),
            )
            electric[..., index] = potential - neighbor
        return electric

    def matvec(vector: np.ndarray) -> np.ndarray:
        potential = np.asarray(vector, dtype=np.float64).reshape(shape)
        electric = electric_from_potential(potential)
        divergence = color_gauss_divergence(
            electric,
            parameters.r4,
        )
        divergence += np.mean(potential)
        return divergence.reshape(-1)

    operator = LinearOperator(
        (count, count),
        matvec=matvec,
        dtype=np.float64,
    )
    guess = (
        np.zeros(shape, dtype=np.float64)
        if initial_potential is None
        else np.asarray(initial_potential, dtype=np.float64)
    )
    if guess.shape != shape:
        raise ValueError("initial_potential must match charge")
    solution, info = cg(
        operator,
        charge_values.reshape(-1),
        x0=guess.reshape(-1),
        rtol=tolerance,
        atol=0.0,
        maxiter=20 * count,
    )
    if info != 0:
        raise RuntimeError(f"U1 Gauss iteration did not converge: {info}")
    potential = solution.reshape(shape)
    potential -= np.mean(potential)
    electric = electric_from_potential(potential)
    residual = color_gauss_divergence(electric, parameters.r4) - charge_values
    scale = max(float(np.max(np.abs(charge_values))), 1.0)
    return R5U1StaticState(
        potential=potential,
        electric=electric,
        charge=charge_values.copy(),
        gauss_residual=float(np.max(np.abs(residual)) / scale),
        electric_energy=float(0.5 * np.sum(electric**2)),
    )


def periodic_point_pair_charge(
    size: int,
    separation: int,
    relative_sign: int,
) -> np.ndarray:
    """Return two unit point charges with the required neutral background."""

    if size < 5 or separation < 1 or separation >= size // 2:
        raise ValueError("point pair must fit inside half the periodic box")
    if relative_sign not in (-1, 1):
        raise ValueError("relative_sign must be -1 or +1")
    charge = np.full(
        (size, size, size),
        -(1.0 + relative_sign) / size**3,
        dtype=np.float64,
    )
    center = size // 2
    left = center - separation // 2
    right = left + separation
    charge[left, center, center] += 1.0
    charge[right, center, center] += float(relative_sign)
    return charge
