"""
Laplacian Stencils
==================

Discrete Laplacian operators on cubic lattices.

The 19-point stencil (6 faces + 12 edges, weights 1/3 and 1/6) achieves
O(h⁴) isotropy — the most isotropic possible on a cubic lattice.
The 8 corners (distance √3) are NOT included as they worsen isotropy.

χ₀ = 1 (center) + 6 (faces) + 12 (edges) = 19.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.constants import STENCIL_CENTER_WEIGHT, STENCIL_EDGE_WEIGHT, STENCIL_FACE_WEIGHT

if TYPE_CHECKING:
    from numpy.typing import NDArray


def laplacian_19pt(field: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute 19-point isotropic Laplacian on a 3D periodic grid.

    Uses 6 face neighbors (weight 1/3) + 12 edge neighbors (weight 1/6).
    Center weight = -4. Assumes dx = 1.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        3D scalar field on periodic cubic lattice.

    Returns
    -------
    ndarray, shape (N, N, N)
        Laplacian ∇²field.
    """
    # Face neighbors (distance 1): 6 terms, weight 1/3
    faces = (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=2)
        + np.roll(field, -1, axis=2)
    )

    # Edge neighbors (distance √2): 12 terms, weight 1/6
    edges = (
        # xy edges (4)
        np.roll(np.roll(field, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), -1, axis=1)
        # xz edges (4)
        + np.roll(np.roll(field, 1, axis=0), 1, axis=2)
        + np.roll(np.roll(field, 1, axis=0), -1, axis=2)
        + np.roll(np.roll(field, -1, axis=0), 1, axis=2)
        + np.roll(np.roll(field, -1, axis=0), -1, axis=2)
        # yz edges (4)
        + np.roll(np.roll(field, 1, axis=1), 1, axis=2)
        + np.roll(np.roll(field, 1, axis=1), -1, axis=2)
        + np.roll(np.roll(field, -1, axis=1), 1, axis=2)
        + np.roll(np.roll(field, -1, axis=1), -1, axis=2)
    )

    return STENCIL_FACE_WEIGHT * faces + STENCIL_EDGE_WEIGHT * edges + STENCIL_CENTER_WEIGHT * field


def gradient_19pt(
    field: NDArray[np.floating],
    dx: float = 1.0,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Return the isotropic site-centred gradient paired with the 19-point grid.

    Face differences carry weight ``1/3`` and the two edge planes touching
    each axis carry weight ``1/6``. The final factor of one half converts the
    symmetric two-cell difference to a derivative. Periodic boundaries match
    :func:`laplacian_19pt` and the LIMIT-02 FFT solver.
    """
    if field.ndim != 3:
        raise ValueError("field must be a 3-D array")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    gradients = []
    for axis in range(3):
        plus = np.roll(field, -1, axis=axis)
        minus = np.roll(field, 1, axis=axis)
        directional = STENCIL_FACE_WEIGHT * (plus - minus)

        other_axes = [candidate for candidate in range(3) if candidate != axis]
        for other_axis in other_axes:
            edge_difference = np.zeros_like(field)
            for other_shift in (-1, 1):
                plus_edge = np.roll(
                    np.roll(field, -1, axis=axis),
                    other_shift,
                    axis=other_axis,
                )
                minus_edge = np.roll(
                    np.roll(field, 1, axis=axis),
                    other_shift,
                    axis=other_axis,
                )
                edge_difference += plus_edge - minus_edge
            directional += STENCIL_EDGE_WEIGHT * edge_difference

        gradients.append(directional / (2.0 * dx))
    return gradients[0], gradients[1], gradients[2]


def eigenvalue_19pt(
    kx: NDArray[np.floating],
    ky: NDArray[np.floating],
    kz: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return the spectral eigenvalue of the 19-point stencil.

    The result matches :func:`laplacian_19pt` exactly on a periodic grid
    with dx = 1. It is useful for FFT Poisson solves whose equilibrium
    must be consistent with the same lattice operator used for evolution.
    """
    face = (
        (2.0 * np.cos(kx) - 2.0) / 3.0
        + (2.0 * np.cos(ky) - 2.0) / 3.0
        + (2.0 * np.cos(kz) - 2.0) / 3.0
    )
    edge = (
        np.cos(kx + ky)
        + np.cos(kx - ky)
        + np.cos(kx + kz)
        + np.cos(kx - kz)
        + np.cos(ky + kz)
        + np.cos(ky - kz)
        - 6.0
    ) / 3.0
    return face + edge


def laplacian_27pt(field: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the ablation-only 27-point isotropic Laplacian.

    Uses 6 face neighbors (weight 4/9), 12 edge neighbors (weight 1/9),
    and 8 corner neighbors (weight 1/36). The center weight is -38/9.
    This operator is provided for explicitly labeled stencil ablations;
    it is not the canonical LFM propagation default.
    """
    faces = (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=2)
        + np.roll(field, -1, axis=2)
    )
    edges = (
        np.roll(np.roll(field, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), 1, axis=2)
        + np.roll(np.roll(field, 1, axis=0), -1, axis=2)
        + np.roll(np.roll(field, -1, axis=0), 1, axis=2)
        + np.roll(np.roll(field, -1, axis=0), -1, axis=2)
        + np.roll(np.roll(field, 1, axis=1), 1, axis=2)
        + np.roll(np.roll(field, 1, axis=1), -1, axis=2)
        + np.roll(np.roll(field, -1, axis=1), 1, axis=2)
        + np.roll(np.roll(field, -1, axis=1), -1, axis=2)
    )
    corners = np.zeros_like(field)
    for shift_x in (-1, 1):
        for shift_y in (-1, 1):
            for shift_z in (-1, 1):
                corners += np.roll(
                    field,
                    shift=(shift_x, shift_y, shift_z),
                    axis=(0, 1, 2),
                )
    return (4.0 / 9.0) * faces + (1.0 / 9.0) * edges + (1.0 / 36.0) * corners - (38.0 / 9.0) * field


def eigenvalue_27pt(
    kx: NDArray[np.floating],
    ky: NDArray[np.floating],
    kz: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return the spectral eigenvalue of the ablation-only 27-point stencil."""
    cos_x = np.cos(kx)
    cos_y = np.cos(ky)
    cos_z = np.cos(kz)
    faces = (8.0 / 9.0) * (cos_x + cos_y + cos_z)
    edges = (4.0 / 9.0) * (cos_x * cos_y + cos_x * cos_z + cos_y * cos_z)
    corners = (2.0 / 9.0) * cos_x * cos_y * cos_z
    return faces + edges + corners - (38.0 / 9.0)


def noether_current_19pt_raw(
    psi_real: NDArray[np.floating],
    psi_imag: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Return raw site-centered current components for the 19-point stencil.

    This is the face-and-edge link current paired with
    :func:`laplacian_19pt`. For a plane wave with amplitude ``A`` it gives

    ``J_x = 2 A^2 sin(k_x) (1 + cos(k_y) + cos(k_z)) / 3``

    and cyclic permutations. Equivalently, each raw component is minus the
    corresponding derivative of the 19-point stencil eigenvalue times
    ``A^2``. The production current convention applies a factor of one half
    when forming the physical scalar source.

    The result is an observable for the optional current-feedback extension.
    It does not promote that extension into the bare GOV-02 action.
    """
    if psi_real.shape != psi_imag.shape:
        raise ValueError("psi_real and psi_imag must have identical shapes")
    if psi_real.ndim != 3:
        raise ValueError("19-point Noether current requires 3D fields")

    def shifted(field: NDArray[np.floating], dx: int, dy: int, dz: int):
        return np.roll(field, shift=(-dx, -dy, -dz), axis=(0, 1, 2))

    def directional_difference(field: NDArray[np.floating], axis: int):
        plus = [0, 0, 0]
        minus = [0, 0, 0]
        plus[axis] = 1
        minus[axis] = -1
        result = STENCIL_FACE_WEIGHT * (shifted(field, *plus) - shifted(field, *minus))

        other_axes = [candidate for candidate in range(3) if candidate != axis]
        for other_axis in other_axes:
            edge_sum = np.zeros_like(field)
            for other_sign in (-1, 1):
                plus_edge = plus.copy()
                minus_edge = minus.copy()
                plus_edge[other_axis] = other_sign
                minus_edge[other_axis] = other_sign
                edge_sum += shifted(field, *plus_edge)
                edge_sum -= shifted(field, *minus_edge)
            result += STENCIL_EDGE_WEIGHT * edge_sum
        return result

    currents = []
    for axis in range(3):
        d_real = directional_difference(psi_real, axis)
        d_imag = directional_difference(psi_imag, axis)
        currents.append(psi_real * d_imag - psi_imag * d_real)
    return currents[0], currents[1], currents[2]


def laplacian_7pt(field: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute standard 7-point Laplacian on a 3D periodic grid.

    Uses only 6 face neighbors (weight 1). Center weight = -6.
    O(h²) accuracy, 12.3% group velocity anisotropy at |k|=1.
    Use 19-point stencil for production; this is for comparison only.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        3D scalar field on periodic cubic lattice.

    Returns
    -------
    ndarray, shape (N, N, N)
        Laplacian ∇²field.
    """
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=2)
        + np.roll(field, -1, axis=2)
        - 6.0 * field
    )
