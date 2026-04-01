"""Outgoing spherical light wavefront for Level 1 (complex Ψ) simulations.

Physics
-------
In LFM Level 1, the global U(1) phase symmetry of GOV-01 protects a
massless mode: a uniform global phase Ψ → Ψ·exp(iα) leaves |Ψ|² unchanged,
so GOV-02 is unaffected and χ stays at χ₀.  A small *spatial* phase
perturbation δθ(x,t) on top of a background amplitude A₀ satisfies:

    ∂²(δθ)/∂t² = c²∇²(δθ)          (massless wave equation)

This is the LFM photon.  The perturbation propagates at v_g = 0.9912c
(0.9% below c due to 19-point stencil dispersion — a real lattice prediction,
not a parameter).

Initial conditions
------------------
The outward-only spherical wave requires both Ψ(t=0) *and* Ψ(t=−Δt) to be
set so that the leapfrog starts with zero inward component.  The 1/r-weighted
Gaussian shell

    δΨ(r) = δθ · (R₀/r) · exp(−(r−R₀)²/(2σ²))

is placed at R₀ for t = 0 and at R₀ − c·Δt for t = −Δt, encoding an
outward-propagating d'Alembertian solution f(r−ct)/r.

Usage
-----
Typically called via ``Simulation.place_light_source()``::

    sim = lfm.Simulation(lfm.SimulationConfig(
        grid_size=64,
        field_level=lfm.FieldLevel.COMPLEX,
    ))
    sim.place_light_source((32, 32, 32), R0=12.0, sigma=2.0)
    sim.run(steps=55)

Direct usage::

    from lfm.fields.light import spherical_phase_source
    psi_r, psi_i, psi_r_prev, psi_i_prev = spherical_phase_source(
        N=64, center=(32, 32, 32),
        R0=12.0, sigma=2.0, delta_theta=0.25,
        dt=0.40, c_speed=1.0,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def spherical_phase_source(
    N: int,
    center: tuple[float, float, float],
    R0: float = 12.0,
    sigma: float = 2.0,
    delta_theta: float = 0.25,
    dt: float = 0.40,
    c_speed: float = 1.0,
    charge_phase: float = 0.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Build four field arrays for an outward-only spherical light wavefront.

    Returns ``(psi_r_curr, psi_i_curr, psi_r_prev, psi_i_prev)`` suitable
    for direct assignment to a ``Simulation``'s field buffers.

    The two-timestep initialization (current *and* previous) cancels any
    inward-propagating component so that the wave expands cleanly outward
    with no reflection from the origin.

    Parameters
    ----------
    N : int
        Grid side length (cells).
    center : (cx, cy, cz)
        Source position in grid coordinates.
    R0 : float
        Initial shell radius in grid cells.  Must satisfy
        ``R0 > 3*sigma`` so the shell is spatially distinct from the
        origin (where 1/r diverges).  Typical: 10–15 cells.
    sigma : float
        Shell half-width in grid cells.  Should be ≥ 2 for a
        spectrally narrow pulse.
    delta_theta : float
        Peak phase perturbation (radians).  Keep ≪ 1 for the
        massless-photon approximation to hold (linear regime).
        Typical: 0.1–0.3.
    dt : float
        Leapfrog timestep (lattice units).  Must match
        ``SimulationConfig.dt``.
    c_speed : float
        Wave speed (lattice units).  Must match ``SimulationConfig.c``.
    charge_phase : float
        Global phase rotation of the wavefront (radians).
        ``0`` → electron-like (real part dominant).
        ``π/2`` → imaginary part dominant.
        ``π`` → positron-like (real part reversed).

    Returns
    -------
    psi_r_curr : float32 ndarray, shape (N, N, N)
        Real part of Ψ at t = 0.
    psi_i_curr : float32 ndarray, shape (N, N, N)
        Imaginary part of Ψ at t = 0.
    psi_r_prev : float32 ndarray, shape (N, N, N)
        Real part of Ψ at t = −Δt.
    psi_i_prev : float32 ndarray, shape (N, N, N)
        Imaginary part of Ψ at t = −Δt.

    Notes
    -----
    The 1/r weighting is regularised at the origin: cells with
    r < 0.5 are set to zero (they're inside the numerical stencil
    footprint and would otherwise diverge).
    """
    x = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    cx, cy, cz = center
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)

    def _shell(r_centre: float) -> NDArray[np.float64]:
        """1/r-weighted Gaussian centred at r_centre."""
        r_safe = np.where(R > 0.5, R, np.inf)
        return delta_theta * R0 / r_safe * np.exp(-((R - r_centre) ** 2) / (2.0 * sigma**2))

    shell_t0 = _shell(R0)  # t = 0
    shell_tm1 = _shell(R0 - c_speed * dt)  # t = −Δt (shifted inward → outward IC)

    cos_p = np.cos(charge_phase)
    sin_p = np.sin(charge_phase)

    return (
        (shell_t0 * cos_p).astype(np.float32),
        (shell_t0 * sin_p).astype(np.float32),
        (shell_tm1 * cos_p).astype(np.float32),
        (shell_tm1 * sin_p).astype(np.float32),
    )
