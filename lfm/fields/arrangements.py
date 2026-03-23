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


# ---------------------------------------------------------------------------
# Galactic disk initializers (P011)
# ---------------------------------------------------------------------------

def disk_positions(
    N: int,
    n_solitons: int,
    r_inner: float = 5.0,
    r_outer: float | None = None,
    plane_axis: int = 2,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Positions uniformly distributed (by area) in a galactic disk.

    Parameters
    ----------
    N : int
        Grid size per axis.
    n_solitons : int
        Number of positions to generate.
    r_inner : float
        Inner radius of the disk in grid cells.
    r_outer : float or None
        Outer radius.  Defaults to 75 % of the half-grid (``0.375 * N``).
    plane_axis : int
        Axis normal to the disk plane (0=X, 1=Y, 2=Z).  Solitons are placed
        in the plane perpendicular to this axis through the grid center.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (n_solitons, 3)
        Positions [[x, y, z], ...] in grid-cell coordinates.
    """
    rng = np.random.default_rng(seed)
    if r_outer is None:
        r_outer = 0.375 * N
    if r_outer <= r_inner:
        raise ValueError(f"r_outer ({r_outer}) must exceed r_inner ({r_inner})")

    center = N / 2.0
    # Uniform area density: r = sqrt(U*(r_outer² - r_inner²) + r_inner²)
    U = rng.uniform(0.0, 1.0, n_solitons)
    r = np.sqrt(U * (r_outer**2 - r_inner**2) + r_inner**2)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_solitons)

    axes = [i for i in range(3) if i != plane_axis]
    a0, a1 = axes[0], axes[1]

    positions = np.full((n_solitons, 3), center, dtype=np.float64)
    positions[:, a0] = center + r * np.cos(theta)
    positions[:, a1] = center + r * np.sin(theta)
    return positions


def disk_velocities(
    positions: NDArray,
    chi0: float = 19.0,
    kappa: float = 1.0 / 63.0,
    c: float = 1.0,
    center: tuple[float, float, float] | None = None,
    plane_axis: int = 2,
    v_scale: float = 0.05,
) -> NDArray[np.float64]:
    """Tangential circular velocities for solitons in a galactic disk.

    Uses a flat-rotation-curve approximation: every soliton orbits at the
    same characteristic speed ``v_scale * sqrt(G_eff * chi0) / c``.  The
    direction is tangential to the radius vector in the disk plane.

    Parameters
    ----------
    positions : ndarray, shape (n, 3)
        Soliton centres from :func:`disk_positions`.
    chi0 : float
        Background chi value (default 19).
    kappa : float
        LFM gravitational coupling (default 1/63).
    c : float
        Wave speed (default 1.0).
    center : tuple or None
        Disk centre.  Defaults to the mean of *positions*.
    plane_axis : int
        Axis normal to disk plane.
    v_scale : float
        Scalar multiplier on the characteristic velocity.  Tune this so that
        the resulting velocities match the expected flat curve for a given
        mass profile.

    Returns
    -------
    ndarray, shape (n, 3)
        Velocity vectors [[vx, vy, vz], ...].
    """
    n = len(positions)
    if center is None:
        ctr = positions.mean(axis=0)
    else:
        ctr = np.array(center, dtype=np.float64)

    axes = [i for i in range(3) if i != plane_axis]
    a0, a1 = axes[0], axes[1]

    # Characteristic flat-curve speed
    G_eff = c**4 / (kappa * chi0**2)
    v_flat = v_scale * np.sqrt(G_eff * chi0) / c

    velocities = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        dx = positions[i, a0] - ctr[a0]
        dy = positions[i, a1] - ctr[a1]
        r = np.hypot(dx, dy)
        if r < 1e-10:
            continue
        # Tangential direction: (-dy, dx) / r  (counter-clockwise)
        velocities[i, a0] = -v_flat * dy / r
        velocities[i, a1] =  v_flat * dx / r
    return velocities


def initialize_disk(
    sim,
    n_solitons: int = 200,
    r_inner: float = 5.0,
    r_outer: float | None = None,
    amplitude: float | None = None,
    sigma: float | None = None,
    plane_axis: int = 2,
    add_velocities: bool = True,
    v_scale: float = 0.05,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Place a rotating galactic disk of solitons into *sim*.

    Convenience wrapper that calls :func:`disk_positions`, optionally
    :func:`disk_velocities`, then :meth:`Simulation.place_soliton` for each
    soliton.

    Parameters
    ----------
    sim : Simulation
        An initialised (not yet equilibrated) simulation.
    n_solitons : int
        Number of disk solitons to place.
    r_inner : float
        Inner disk radius in grid cells.
    r_outer : float or None
        Outer disk radius (default 75 % of half-grid).
    amplitude : float or None
        Soliton amplitude.  Forwarded to ``place_soliton``; uses its default
        if *None*.
    sigma : float or None
        Soliton width.  Forwarded to ``place_soliton``; uses its default if
        *None*.
    plane_axis : int
        Axis normal to disk plane (0=X, 1=Y, 2=Z).
    add_velocities : bool
        If True, give each soliton a tangential circular velocity.
    v_scale : float
        Scale factor for the tangential velocity (passed to
        :func:`disk_velocities`).
    seed : int or None
        Random seed for position placement.

    Returns
    -------
    ndarray, shape (n_solitons, 3)
        The positions used.
    """
    N = sim.config.grid_size
    center = (N / 2.0, N / 2.0, N / 2.0)

    positions = disk_positions(N, n_solitons, r_inner, r_outer, plane_axis, seed)
    velocities: NDArray | None = None
    if add_velocities:
        velocities = disk_velocities(
            positions,
            chi0=sim.config.chi0,
            kappa=sim.config.kappa,
            c=sim.config.c,
            center=center,
            plane_axis=plane_axis,
            v_scale=v_scale,
        )

    kw: dict = {}
    if amplitude is not None:
        kw["amplitude"] = amplitude
    if sigma is not None:
        kw["sigma"] = sigma

    for i, pos in enumerate(positions):
        vel = tuple(float(v) for v in velocities[i]) if velocities is not None else None
        sim.place_soliton(tuple(float(p) for p in pos), velocity=vel, **kw)

    return positions
