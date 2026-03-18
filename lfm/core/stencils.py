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

import numpy as np
from numpy.typing import NDArray

from lfm.constants import STENCIL_CENTER_WEIGHT, STENCIL_EDGE_WEIGHT, STENCIL_FACE_WEIGHT


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

    return (
        STENCIL_FACE_WEIGHT * faces
        + STENCIL_EDGE_WEIGHT * edges
        + STENCIL_CENTER_WEIGHT * field
    )


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
