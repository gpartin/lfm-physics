"""Mode projections for periodic LFM fields and leapfrog phase space."""

from __future__ import annotations

import numpy as np


def periodic_mode_coefficient(
    field: np.ndarray,
    mode: int,
    *,
    axis: int = -1,
    background: float | complex = 0.0,
) -> np.ndarray | complex:
    """Project periodic lines onto one spatial Fourier mode.

    Leading dimensions are preserved, so a bank of lines can be measured in
    one call. The returned coefficient uses the convention

    ``mean((field-background) * exp(-2*pi*i*mode*x/N))``.

    This function is a terminal observable. It does not alter the field.
    """
    arr = np.asarray(field)
    if arr.ndim == 0:
        raise ValueError("field must have at least one dimension")
    normalized_axis = int(axis)
    if not -arr.ndim <= normalized_axis < arr.ndim:
        raise ValueError(
            f"axis {normalized_axis} is out of bounds for dimension {arr.ndim}"
        )
    normalized_axis %= arr.ndim
    n = int(arr.shape[normalized_axis])
    if n <= 0:
        raise ValueError("projection axis must be non-empty")
    coordinate = np.arange(n, dtype=np.float64)
    carrier = np.exp(-2j * np.pi * int(mode) * coordinate / float(n))
    shape = [1] * arr.ndim
    shape[normalized_axis] = n
    coefficient = np.mean(
        (arr - background) * carrier.reshape(shape),
        axis=normalized_axis,
    )
    if np.ndim(coefficient) == 0:
        return complex(coefficient)
    return np.asarray(coefficient, dtype=np.complex128)


def leapfrog_branch_projection(
    current: np.ndarray | complex,
    previous: np.ndarray | complex,
    theta: float,
) -> tuple[np.ndarray | complex, np.ndarray | complex]:
    """Resolve a two-buffer leapfrog state into temporal branches.

    The convention is

    ``current = forward + backward``

    ``previous = forward*exp(+i*theta) + backward*exp(-i*theta)``.

    This is a terminal observable for a known temporal frequency. It does not
    evolve, relocate, or otherwise modify the input fields.
    """
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    if abs(float(np.sin(theta))) <= 1.0e-15:
        raise ValueError("theta does not separate the two temporal branches")

    current_arr = np.asarray(current, dtype=np.complex128)
    previous_arr = np.asarray(previous, dtype=np.complex128)
    try:
        current_arr, previous_arr = np.broadcast_arrays(
            current_arr, previous_arr
        )
    except ValueError as exc:
        raise ValueError(
            "current and previous must be broadcast-compatible"
        ) from exc

    positive = np.exp(1j * float(theta))
    negative = np.exp(-1j * float(theta))
    denominator = positive - negative
    forward = (previous_arr - current_arr * negative) / denominator
    backward = (current_arr * positive - previous_arr) / denominator
    if forward.ndim == 0:
        return complex(forward), complex(backward)
    return forward, backward


def project_leapfrog_mode(
    current_field: np.ndarray,
    previous_field: np.ndarray,
    mode: int,
    theta: float,
    *,
    axis: int = -1,
    background: float | complex = 0.0,
) -> tuple[np.ndarray | complex, np.ndarray | complex]:
    """Project one spatial mode and split its two temporal branches."""
    current = periodic_mode_coefficient(
        current_field, mode, axis=axis, background=background
    )
    previous = periodic_mode_coefficient(
        previous_field, mode, axis=axis, background=background
    )
    return leapfrog_branch_projection(current, previous, theta)
