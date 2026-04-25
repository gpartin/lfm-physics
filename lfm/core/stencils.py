"""
Laplacian Stencils
==================

Discrete Laplacian operators on cubic lattices.

Three valid cubic-symmetric stencils are supported:

7-point (face only):
    6 face neighbors, O(h^2) accuracy, 12.3% group velocity anisotropy.
    chi0_implied = 7. Use only for comparison; NOT the canonical LFM stencil.

19-point (face + edge)  [CANONICAL LFM]:
    6 face + 12 edge neighbors, O(h^4) isotropic. chi0_implied = 19.
    This is the ONLY stencil consistent with all LFM observational tests.
    chi0 = 1 (center) + 6 (faces) + 12 (edges) = 19 = 3^D - 2^D at D=3.

27-point (face + edge + corner):
    6 face + 12 edge + 8 corner neighbors, O(h^4) isotropic. chi0_implied = 27.
    chi0=27 fails all observational tests (N_gen=4.33, N_gluons=16).
    Provided for comparison with alternative stencil proposals.
    Weights: wf=1/2, we=1/12, wc=1/24, center=-13/3.
    lambda_max = 20/3 at k=(pi,pi,pi). CFL barely tighter than 19-pt
    because chi0^2=361 >> lambda_max difference.

Derivation of 19-pt S2=2 normalization:
    The stencil (wf*(faces) + we*(edges) + center*field) approximates nabla^2 f
    with coefficient wf + 4*we = 1/3 + 2/3 = 1. (Exact at O(h^2), O(h^4) isotropic.)

Derivation of 27-pt weights (LFM canonical, S2=2):
    O(h^4) isotropy requires: wf = 2*we + 8*wc (from Taylor expansion isotropy condition).
    Normalization requires: wf + 4*we + 4*wc = 1.
    The unique O(h^4) isotropic 27-pt family is: wf = 1/3+4*wc, we = 1/6-2*wc.
    Choosing wc = 1/24 gives the simplest non-zero-corner stencil:
    wf = 1/2, we = 1/12, wc = 1/24, center = -13/3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.constants import (
    STENCIL_CENTER_WEIGHT,
    STENCIL_EDGE_WEIGHT,
    STENCIL_FACE_WEIGHT,
    STENCIL_27PT_CENTER_WEIGHT,
    STENCIL_27PT_CORNER_WEIGHT,
    STENCIL_27PT_EDGE_WEIGHT,
    STENCIL_27PT_FACE_WEIGHT,
)

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


def laplacian_7pt(field: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute standard 7-point Laplacian on a 3D periodic grid.

    Uses only 6 face neighbors (weight 1). Center weight = -6.
    O(h^2) accuracy, 12.3% group velocity anisotropy at |k|=1.
    Use 19-point stencil for production; this is for comparison only.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        3D scalar field on periodic cubic lattice.

    Returns
    -------
    ndarray, shape (N, N, N)
        Laplacian nabla^2 field.
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


def laplacian_27pt(field: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute 27-point full-cube Laplacian on a 3D periodic grid.

    Extends the 19-point stencil with 8 corner neighbors (distance sqrt(3)).
    Weights are derived from the O(h^4) isotropy + S2=2 normalization conditions,
    using the canonical LFM choice wc=1/24 from the one-parameter family.

    Weights (LFM canonical, S2=2):
        Face   (6,  dist=1):     wf = 1/2
        Edge   (12, dist=sqrt2): we = 1/12
        Corner (8,  dist=sqrt3): wc = 1/24
        Center:                  -(6*(1/2) + 12*(1/12) + 8*(1/24)) = -13/3

    O(h^4) isotropy verified: cross/diag ratio = (we+2*wc)/(wf/12+we/3+wc/3)
        = (1/12+1/12)/(1/24+1/36+1/72) = (1/6)/(1/12) = 2 = ratio in nabla^4.
    Normalization: wf + 4*we + 4*wc = 1/2 + 1/3 + 1/6 = 1. Gives nabla^2 f exactly.
    lambda_max = 20/3 at k=(pi,pi,pi) vs 16/3 for 19-pt. CFL barely tighter
        because chi0^2=361 >> lambda difference.

    IMPORTANT: chi0_implied = 27 fails all LFM observational tests (N_gen=4.33,
    N_gluons=16, etc.). Use laplacian_19pt for all canonical LFM simulations.
    This function is provided for stencil comparison and response to Procaccia (2026).

    NOTE on Procaccia (2026) weights: that paper uses wf=1/6, we=1/12, wc=1/24
    with center=-7/3, giving (2/3)*nabla^2 f (S2=4/3 normalization, NOT S2=2).
    Combined with kappa=sqrt(3)/361 (3.3x smaller than canonical 1/63) and
    an extra FORCE_COEFF=3/19, this caused the 19-pt leapfrog to diverge at
    step ~26500 -- a parameter artifact, not a stencil instability.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        3D scalar field on periodic cubic lattice.

    Returns
    -------
    ndarray, shape (N, N, N)
        Laplacian nabla^2 field.
    """
    # Face neighbors (distance 1): 6 terms, weight 1/2
    faces = (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=2)
        + np.roll(field, -1, axis=2)
    )

    # Edge neighbors (distance sqrt(2)): 12 terms, weight 1/12
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

    # Corner neighbors (distance sqrt(3)): 8 terms, weight 1/24
    corners = (
        np.roll(np.roll(np.roll(field, 1, axis=0), 1, axis=1), 1, axis=2)
        + np.roll(np.roll(np.roll(field, 1, axis=0), 1, axis=1), -1, axis=2)
        + np.roll(np.roll(np.roll(field, 1, axis=0), -1, axis=1), 1, axis=2)
        + np.roll(np.roll(np.roll(field, 1, axis=0), -1, axis=1), -1, axis=2)
        + np.roll(np.roll(np.roll(field, -1, axis=0), 1, axis=1), 1, axis=2)
        + np.roll(np.roll(np.roll(field, -1, axis=0), 1, axis=1), -1, axis=2)
        + np.roll(np.roll(np.roll(field, -1, axis=0), -1, axis=1), 1, axis=2)
        + np.roll(np.roll(np.roll(field, -1, axis=0), -1, axis=1), -1, axis=2)
    )

    return (
        STENCIL_27PT_FACE_WEIGHT * faces
        + STENCIL_27PT_EDGE_WEIGHT * edges
        + STENCIL_27PT_CORNER_WEIGHT * corners
        + STENCIL_27PT_CENTER_WEIGHT * field
    )
