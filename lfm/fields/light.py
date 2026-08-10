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

from lfm.constants import CHI0
from lfm.core.stencils import laplacian_19pt

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _runtime_float_dtype(*arrays: object) -> type[np.float32] | type[np.float64]:
    """Use float64 only when a caller explicitly supplies float64 arrays."""
    for arr in arrays:
        dtype = getattr(arr, "dtype", None)
        if dtype is not None and np.dtype(dtype) == np.dtype(np.float64):
            return np.float64
    return np.float32


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


def planar_r1_light_packet(
    N: int,
    center: tuple[float, float, float],
    sigma: tuple[float, float, float],
    carrier_k: float,
    amplitude: float = 0.20,
    dt: float = 0.32,
    c_speed: float = 1.0,
    axis: int = 0,
    dtype: type[np.float32] | type[np.float64] = np.float32,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return a localized R1 complex packet moving in the +axis direction."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    out_dtype = np.dtype(dtype).type
    if out_dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be np.float32 or np.float64")

    coords = np.arange(N, dtype=out_dtype)
    grids = np.meshgrid(coords, coords, coords, indexing="ij")
    center_arr = np.asarray(center, dtype=out_dtype)
    sigma_arr = np.asarray(sigma, dtype=out_dtype)
    if np.any(sigma_arr <= 0.0):
        raise ValueError("sigma values must be positive")

    def _packet(
        packet_center: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        radius_sq = np.zeros((N, N, N), dtype=out_dtype)
        for idx, grid in enumerate(grids):
            radius_sq += ((grid - packet_center[idx]) / sigma_arr[idx]) ** 2
        envelope = amplitude * np.exp(-0.5 * radius_sq)
        phase = carrier_k * (grids[axis] - packet_center[axis])
        return (
            (envelope * np.cos(phase)).astype(out_dtype),
            (envelope * np.sin(phase)).astype(out_dtype),
        )

    prev_center = center_arr.copy()
    prev_center[axis] -= c_speed * dt
    psi_r, psi_i = _packet(center_arr)
    psi_r_prev, psi_i_prev = _packet(prev_center)
    return psi_r, psi_i, psi_r_prev, psi_i_prev


def r1_vacuum_subtracted_potential(
    chi: NDArray[np.floating],
    chi0: float = CHI0,
) -> NDArray[np.floating]:
    """Return chi^2 - chi0^2 for the massless R1 light perturbation."""
    out_dtype = _runtime_float_dtype(chi)
    chi_f = chi.astype(out_dtype, copy=False)
    return (chi_f * chi_f - out_dtype(chi0 * chi0)).astype(out_dtype)


def r1_light_acceleration(
    psi_r: NDArray[np.floating],
    psi_i: NDArray[np.floating],
    chi: NDArray[np.floating] | None = None,
    chi0: float = CHI0,
    c_speed: float = 1.0,
    vacuum_subtracted: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the R1 light acceleration for one leapfrog update.

    With chi=None this is the flat U(1) massless phase/current sector.
    With chi provided and vacuum_subtracted=True the potential is
    chi^2 - chi0^2, so uniform vacuum remains massless while nonuniform
    chi affects the full complex R1 field.
    """
    out_dtype = _runtime_float_dtype(psi_r, psi_i, chi)
    acc_r = (c_speed * c_speed * laplacian_19pt(psi_r)).astype(out_dtype)
    acc_i = (c_speed * c_speed * laplacian_19pt(psi_i)).astype(out_dtype)
    if chi is None:
        return acc_r, acc_i

    chi_f = chi.astype(out_dtype, copy=False)
    if vacuum_subtracted:
        potential = r1_vacuum_subtracted_potential(chi_f, chi0)
    else:
        potential = (chi_f * chi_f).astype(out_dtype)
    return (
        (acc_r - potential * psi_r).astype(out_dtype),
        (acc_i - potential * psi_i).astype(out_dtype),
    )


def r1_light_step(
    psi_r: NDArray[np.floating],
    psi_i: NDArray[np.floating],
    psi_r_prev: NDArray[np.floating],
    psi_i_prev: NDArray[np.floating],
    dt: float,
    chi: NDArray[np.floating] | None = None,
    chi0: float = CHI0,
    c_speed: float = 1.0,
    vacuum_subtracted: bool = True,
    sponge: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Advance one leapfrog step for a massless R1 light packet."""
    out_dtype = _runtime_float_dtype(psi_r, psi_i, psi_r_prev, psi_i_prev, chi)
    acc_r, acc_i = r1_light_acceleration(
        psi_r,
        psi_i,
        chi=chi,
        chi0=chi0,
        c_speed=c_speed,
        vacuum_subtracted=vacuum_subtracted,
    )
    dt2 = out_dtype(dt * dt)
    psi_r_next = (2.0 * psi_r - psi_r_prev + dt2 * acc_r).astype(out_dtype)
    psi_i_next = (2.0 * psi_i - psi_i_prev + dt2 * acc_i).astype(out_dtype)
    psi_r_prev_next = psi_r.astype(out_dtype, copy=True)
    psi_i_prev_next = psi_i.astype(out_dtype, copy=True)

    if sponge is not None:
        sponge_f = sponge.astype(out_dtype, copy=False)
        psi_r_next *= sponge_f
        psi_i_next *= sponge_f
        psi_r_prev_next *= sponge_f
        psi_i_prev_next *= sponge_f

    return psi_r_next, psi_i_next, psi_r_prev_next, psi_i_prev_next
