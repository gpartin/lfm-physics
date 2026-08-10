"""Local frame-comparison algebra for LFM candidate provenance audits.

These helpers do not add frame links to the canonical simulation register.
They distinguish a comparator reconstructed from site frames, which is flat
by construction, from an independent link capable of nontrivial loop
mismatch.
"""

from __future__ import annotations

import numpy as np


def reconstructed_frame_link(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
) -> np.ndarray:
    """Return the comparator carrying local-j components to local i."""

    source = np.asarray(frame_i, dtype=np.float64)
    target = np.asarray(frame_j, dtype=np.float64)
    if source.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("frames must have shape (4,4)")
    return np.linalg.solve(source, target)


def linked_frame_difference(
    value_i: np.ndarray,
    value_j: np.ndarray,
    link_ij: np.ndarray,
) -> np.ndarray:
    """Return the locally covariant neighbor difference."""

    left = np.asarray(value_i, dtype=np.float64)
    right = np.asarray(value_j, dtype=np.float64)
    link = np.asarray(link_ij, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,) or link.shape != (4, 4):
        raise ValueError("values must be four-vectors and link must be 4x4")
    return link @ right - left


def loop_holonomy(*oriented_links: np.ndarray) -> np.ndarray:
    """Return an ordered closed-loop product of local comparators."""

    if not oriented_links:
        raise ValueError("at least one oriented link is required")
    product = np.eye(4, dtype=np.float64)
    for link in oriented_links:
        array = np.asarray(link, dtype=np.float64)
        if array.shape != (4, 4):
            raise ValueError("links must have shape (4,4)")
        product = product @ array
    return product


def loop_mismatch_energy(
    holonomy: np.ndarray,
    *,
    coefficient: float = 1.0,
) -> float:
    """Return a nonnegative candidate loop-mismatch diagnostic."""

    matrix = np.asarray(holonomy, dtype=np.float64)
    strength = float(coefficient)
    if matrix.shape != (4, 4):
        raise ValueError("holonomy must have shape (4,4)")
    if strength <= 0.0:
        raise ValueError("coefficient must be positive")
    mismatch = matrix - np.eye(4, dtype=np.float64)
    return 0.5 * strength * float(np.sum(mismatch**2))
