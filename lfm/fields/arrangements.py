"""
Geometric Arrangements
======================

Standard spatial arrangements for soliton placement on an N³ lattice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tetrahedral_positions(N: int) -> NDArray[np.float64]:
    """Four positions at the vertices of a regular tetrahedron.

    Places vertices at the body-diagonal corners of a cube inscribed
    in the lattice interior, matching the production universe simulator.

    Parameters
    ----------
    N : int
        Grid size per axis.

    Returns
    -------
    ndarray, shape (4, 3)
        Positions [[x, y, z], ...].
    """
    q = N / 4.0
    return np.array(
        [
            [q, q, q],
            [3 * q, 3 * q, q],
            [q, 3 * q, 3 * q],
            [3 * q, q, 3 * q],
        ]
    )


def sparse_positions(
    N: int,
    n_seeds: int,
    boundary_fraction: float = 0.15,
    sigma: float = 3.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Random positions inside an interior sphere.

    Generates non-overlapping positions uniformly distributed within
    a sphere of radius (N/2 - boundary_fraction*N - 3σ), ensuring
    solitons stay away from the frozen boundary.

    Parameters
    ----------
    N : int
        Grid size per axis.
    n_seeds : int
        Number of positions to generate.
    boundary_fraction : float
        Fraction of N reserved for boundary (default 0.15).
    sigma : float
        Soliton width; positions are kept 3σ from boundary sphere.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (n_seeds, 3)
        Positions [[x, y, z], ...].
    """
    rng = np.random.default_rng(seed)
    center = N / 2.0
    r_inner = N / 2.0 - boundary_fraction * N
    r_max = r_inner - 3.0 * sigma

    if r_max <= 0:
        raise ValueError(
            f"No room for solitons: N={N}, boundary_fraction={boundary_fraction}, "
            f"sigma={sigma} → r_max={r_max:.1f}"
        )

    positions = np.empty((n_seeds, 3), dtype=np.float64)
    placed = 0
    min_sep = 2.0 * sigma  # minimum separation between centers

    max_attempts = n_seeds * 1000
    attempts = 0

    while placed < n_seeds and attempts < max_attempts:
        attempts += 1
        # Uniform in sphere via rejection sampling
        pt = rng.uniform(-r_max, r_max, size=3)
        if np.linalg.norm(pt) > r_max:
            continue

        pos = pt + center

        # Check separation from already-placed solitons
        if placed > 0:
            dists = np.linalg.norm(positions[:placed] - pos, axis=1)
            if np.min(dists) < min_sep:
                continue

        positions[placed] = pos
        placed += 1

    if placed < n_seeds:
        raise ValueError(
            f"Could only place {placed}/{n_seeds} solitons in {max_attempts} "
            f"attempts.  Increase N or decrease n_seeds/sigma."
        )

    return positions


def grid_positions(
    N: int,
    n_per_axis: int,
    margin: float | None = None,
) -> NDArray[np.float64]:
    """Evenly spaced positions on a cubic sub-grid.

    Parameters
    ----------
    N : int
        Grid size per axis.
    n_per_axis : int
        Number of positions along each axis.
    margin : float or None
        Distance from edge (default N / (2 * n_per_axis)).

    Returns
    -------
    ndarray, shape (n_per_axis³, 3)
        Positions [[x, y, z], ...].
    """
    if margin is None:
        margin = N / (2.0 * n_per_axis)

    coords = np.linspace(margin, N - margin, n_per_axis)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
