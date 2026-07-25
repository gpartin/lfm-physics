"""Algebra for an unpromoted spacetime cube-frame completion of LFM.

The current canonical LFM register does not contain these frame variables.
This module is deliberately limited to a candidate structural audit. It
decomposes a symmetric four-direction frame strain into one overall-scale
component and nine first-order volume-preserving shape components.

No Newtonian, relativistic-gravity, trajectory, or inverse-Laplacian solver is
implemented here. Static response is evaluated directly from a supplied
lattice stiffness eigenvalue.
"""

from __future__ import annotations

import numpy as np


FRAME_COMPONENT_LABELS = (
    "00",
    "11",
    "22",
    "33",
    "01",
    "02",
    "03",
    "12",
    "13",
    "23",
)
FRAME_COMPONENT_COUNT = len(FRAME_COMPONENT_LABELS)
FRAME_SCALE_COUNT = 1
FRAME_SHAPE_COUNT = FRAME_COMPONENT_COUNT - FRAME_SCALE_COUNT


def frame_scale_direction() -> np.ndarray:
    """Return the unit overall-scale direction in symmetric-frame space."""

    direction = np.zeros(FRAME_COMPONENT_COUNT, dtype=np.float64)
    direction[:4] = 0.5
    return direction


def frame_projectors() -> tuple[np.ndarray, np.ndarray]:
    """Return orthogonal projectors onto scale and shape sectors."""

    scale_direction = frame_scale_direction()
    scale = np.outer(scale_direction, scale_direction)
    shape = np.eye(FRAME_COMPONENT_COUNT, dtype=np.float64) - scale
    return scale, shape


def rest_energy_source() -> np.ndarray:
    """Return a unit source on the temporal frame component."""

    source = np.zeros(FRAME_COMPONENT_COUNT, dtype=np.float64)
    source[0] = 1.0
    return source


def source_projection_weights(
    source: np.ndarray | None = None,
) -> dict[str, float]:
    """Return squared source weights in scale and shape sectors."""

    vector = rest_energy_source() if source is None else np.asarray(
        source,
        dtype=np.float64,
    )
    if vector.shape != (FRAME_COMPONENT_COUNT,):
        raise ValueError("source must have shape (10,)")
    scale, shape = frame_projectors()
    norm_sq = float(vector @ vector)
    if norm_sq <= 0.0:
        raise ValueError("source must be nonzero")
    return {
        "scale": float(vector @ scale @ vector / norm_sq),
        "shape": float(vector @ shape @ vector / norm_sq),
    }


def frame_static_operator(
    lattice_stiffness: float,
    *,
    radial_mass_sq: float,
    normalization: float,
) -> np.ndarray:
    """Return the positive candidate static quadratic operator."""

    stiffness = float(lattice_stiffness)
    mass_sq = float(radial_mass_sq)
    inertia = float(normalization)
    if stiffness < 0.0 or mass_sq <= 0.0 or inertia <= 0.0:
        raise ValueError(
            "stiffness must be nonnegative; mass and normalization positive"
        )
    scale, shape = frame_projectors()
    return inertia * (
        (stiffness + mass_sq) * scale + stiffness * shape
    )


def frame_static_response(
    lattice_stiffness: float,
    *,
    radial_mass_sq: float,
    normalization: float,
    source: np.ndarray | None = None,
) -> float:
    """Return source-projected response for a positive nonzero stiffness."""

    stiffness = float(lattice_stiffness)
    if stiffness <= 0.0:
        raise ValueError("static response requires nonzero positive stiffness")
    vector = rest_energy_source() if source is None else np.asarray(
        source,
        dtype=np.float64,
    )
    operator = frame_static_operator(
        stiffness,
        radial_mass_sq=radial_mass_sq,
        normalization=normalization,
    )
    response = np.linalg.solve(operator, vector)
    return float(vector @ response)


def analytic_rest_energy_response(
    lattice_stiffness: float,
    *,
    radial_mass_sq: float,
    normalization: float,
) -> float:
    """Return the closed-form temporal-source response."""

    stiffness = float(lattice_stiffness)
    mass_sq = float(radial_mass_sq)
    inertia = float(normalization)
    if stiffness <= 0.0 or mass_sq <= 0.0 or inertia <= 0.0:
        raise ValueError("all arguments must be positive")
    return (
        0.75 / (inertia * stiffness)
        + 0.25 / (inertia * (stiffness + mass_sq))
    )


def minimized_source_cross_energy(
    lattice_stiffness: float,
    source_product: float,
    *,
    radial_mass_sq: float,
    normalization: float,
    coupling: float = 1.0,
) -> float:
    """Return the cross term after minimizing the quadratic field energy."""

    response = analytic_rest_energy_response(
        lattice_stiffness,
        radial_mass_sq=radial_mass_sq,
        normalization=normalization,
    )
    return -float(coupling) ** 2 * float(source_product) * response


def zero_momentum_frame_spectrum(
    *,
    radial_mass_sq: float,
    normalization: float = 1.0,
) -> np.ndarray:
    """Return the ten static Hessian eigenvalues at zero momentum."""

    operator = frame_static_operator(
        0.0,
        radial_mass_sq=radial_mass_sq,
        normalization=normalization,
    )
    return np.linalg.eigvalsh(operator)
