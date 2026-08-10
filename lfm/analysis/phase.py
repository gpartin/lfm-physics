"""
Phase / Charge Analysis
=======================

Extract electromagnetic properties from complex wave-field phase.

In LFM, charge = phase θ of the complex wave function.
θ = 0 → electron (negative), θ = π → positron (positive).
Same-phase → repel (constructive), opposite-phase → attract (destructive).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.analysis.energy_current import stencil_links
from lfm.core.stencils import laplacian_19pt, laplacian_27pt

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _runtime_float_dtype(*arrays: object) -> type[np.float32] | type[np.float64]:
    for arr in arrays:
        dtype = getattr(arr, "dtype", None)
        if dtype is not None and np.dtype(dtype) == np.dtype(np.float64):
            return np.float64
    return np.float32


def phase_field(
    psi_r: NDArray,
    psi_i: NDArray,
) -> NDArray:
    """Compute the phase θ(x) = atan2(psi_i, psi_r) at each lattice point.

    Parameters
    ----------
    psi_r : ndarray (N, N, N)
        Real part of Ψ.
    psi_i : ndarray (N, N, N)
        Imaginary part of Ψ.

    Returns
    -------
    theta : ndarray
        Phase in [−π, π].
    """
    return np.arctan2(psi_i, psi_r)


def charge_density(
    psi_r: NDArray,
    psi_i: NDArray,
    dt: float = 0.02,
    psi_r_prev: NDArray | None = None,
    psi_i_prev: NDArray | None = None,
) -> NDArray:
    """Compute the Klein-Gordon charge density (Noether current time component).

    ρ_KG = Im(Ψ* · ∂Ψ/∂t)

    If previous-step fields are provided, the time derivative is approximated
    via finite differences: ∂Ψ/∂t ≈ (Ψ − Ψ_prev)/dt.

    Parameters
    ----------
    psi_r, psi_i : ndarray (N, N, N)
        Current real/imaginary parts.
    dt : float
        Timestep.
    psi_r_prev, psi_i_prev : ndarray or None
        Previous-step fields. If None, returns zeros (static field).

    Returns
    -------
    rho : ndarray
        Charge density (positive for particle, negative for antiparticle).
    """
    if psi_r_prev is None or psi_i_prev is None:
        return np.zeros_like(psi_r)
    dpsi_r_dt = (psi_r - psi_r_prev) / dt
    dpsi_i_dt = (psi_i - psi_i_prev) / dt
    # ρ = Im(Ψ* · dΨ/dt) = psi_r * dpsi_i_dt - psi_i * dpsi_r_dt
    return psi_r * dpsi_i_dt - psi_i * dpsi_r_dt


def canonical_charge_density(
    psi_r: NDArray,
    psi_i: NDArray,
    momentum_r: NDArray,
    momentum_i: NDArray,
) -> NDArray:
    """Return the canonical U(1) charge density from phase-space fields.

    The bare complex GOV-01 field has
    ``rho = psi_r * momentum_i - psi_i * momentum_r``. Unlike a finite
    difference estimate, this observable uses the canonical momentum directly.
    """
    return psi_r * momentum_i - psi_i * momentum_r


def oriented_charge_currents(
    psi_r: NDArray,
    psi_i: NDArray,
    *,
    wave_speed: float = 1.0,
    stencil: str = "19",
) -> dict[tuple[int, int, int], NDArray]:
    """Return exact outgoing U(1) current on every oriented stencil link.

    The current is paired with the selected discrete Laplacian. It therefore
    obeys an exact semidiscrete continuity identity for the bare complex
    GOV-01 equation on a periodic lattice.
    """
    if wave_speed <= 0.0 or not np.isfinite(wave_speed):
        raise ValueError("wave_speed must be positive and finite")
    real = np.asarray(psi_r)
    imag = np.asarray(psi_i)
    if real.shape != imag.shape or real.ndim != 3:
        raise ValueError("psi_r and psi_i must have matching shape (N,N,N)")
    c2 = wave_speed**2
    currents: dict[tuple[int, int, int], NDArray] = {}
    for offset, weight in stencil_links(stencil):
        shifted_real = np.roll(real, shift=offset, axis=(0, 1, 2))
        shifted_imag = np.roll(imag, shift=offset, axis=(0, 1, 2))
        currents[offset] = -c2 * weight * (
            real * shifted_imag - imag * shifted_real
        )
    return currents


def charge_current_divergence(
    psi_r: NDArray,
    psi_i: NDArray,
    *,
    wave_speed: float = 1.0,
    stencil: str = "19",
) -> NDArray:
    """Return the sum of exact outgoing U(1) link currents."""
    currents = oriented_charge_currents(
        psi_r,
        psi_i,
        wave_speed=wave_speed,
        stencil=stencil,
    )
    result = np.zeros_like(np.asarray(psi_r), dtype=np.float64)
    for current in currents.values():
        result += current
    return result


def bare_charge_continuity_residual(
    psi_r: NDArray,
    psi_i: NDArray,
    momentum_r: NDArray,
    momentum_i: NDArray,
    chi: NDArray,
    *,
    wave_speed: float = 1.0,
    stencil: str = "19",
) -> NDArray:
    """Return the exact semidiscrete residual ``d_t rho + div J``.

    The local ``chi**2 * Psi`` term cancels from the U(1) charge rate. This
    function tests the identity rather than advancing a new equation.
    """
    arrays = tuple(
        np.asarray(value)
        for value in (psi_r, psi_i, momentum_r, momentum_i, chi)
    )
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all fields must have matching shapes")
    if arrays[0].ndim != 3:
        raise ValueError("all fields must have shape (N,N,N)")
    if stencil == "19":
        laplacian = laplacian_19pt
    elif stencil == "27":
        laplacian = laplacian_27pt
    else:
        raise ValueError("stencil must be '19' or '27'")
    real, imag, _momentum_real, _momentum_imag, chi_values = arrays
    momentum_rate_real = wave_speed**2 * laplacian(real) - chi_values**2 * real
    momentum_rate_imag = wave_speed**2 * laplacian(imag) - chi_values**2 * imag
    charge_rate = real * momentum_rate_imag - imag * momentum_rate_real
    return charge_rate + charge_current_divergence(
        real,
        imag,
        wave_speed=wave_speed,
        stencil=stencil,
    )


def noether_spatial_current(
    psi_r: NDArray,
    psi_i: NDArray,
    axis: int = 0,
) -> NDArray:
    """Compute spatial Noether current along one lattice axis.

    j_axis = Im(conj(Psi) * d_axis Psi). A centered finite difference is
    used on the periodic lattice.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    dpsi_r = 0.5 * (np.roll(psi_r, -1, axis=axis) - np.roll(psi_r, 1, axis=axis))
    dpsi_i = 0.5 * (np.roll(psi_i, -1, axis=axis) - np.roll(psi_i, 1, axis=axis))
    out_dtype = _runtime_float_dtype(psi_r, psi_i)
    return (psi_r * dpsi_i - psi_i * dpsi_r).astype(out_dtype)


def positive_noether_current(
    psi_r: NDArray,
    psi_i: NDArray,
    axis: int = 0,
) -> NDArray:
    """Return only the positive outgoing part of spatial Noether current."""
    out_dtype = _runtime_float_dtype(psi_r, psi_i)
    return np.maximum(noether_spatial_current(psi_r, psi_i, axis=axis), out_dtype(0.0)).astype(
        out_dtype
    )


def phase_current_energy_density(
    psi_r: NDArray,
    psi_i: NDArray,
    psi_r_prev: NDArray,
    psi_i_prev: NDArray,
    dt: float,
    c_speed: float = 1.0,
    amplitude_floor: float = 1.0e-30,
) -> NDArray:
    """Return the phase-current stress-energy component of a complex wave.

    A pure phase photon can carry energy while ``|Psi|^2`` remains nearly
    constant. This observable extracts that missing source from the U(1)
    Noether current:

        rho_phase = 0.5 * (j_0^2 + c^2 |j_space|^2) / |Psi|^2

    where ``j_0 = Im(conj(Psi) d_t Psi)`` and
    ``j_i = Im(conj(Psi) d_i Psi)``. For ``Psi = A exp(i theta)``, this is
    ``0.5 * A^2 * (theta_t^2 + c^2 |grad theta|^2)``. It is invariant under
    global phase rotations and vanishes for a static uniform phase.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if c_speed < 0.0:
        raise ValueError("c_speed must be non-negative")
    out_dtype = _runtime_float_dtype(psi_r, psi_i, psi_r_prev, psi_i_prev)

    psi_r_f = psi_r.astype(out_dtype, copy=False)
    psi_i_f = psi_i.astype(out_dtype, copy=False)
    psi_r_prev_f = psi_r_prev.astype(out_dtype, copy=False)
    psi_i_prev_f = psi_i_prev.astype(out_dtype, copy=False)

    dpsi_r_dt = (psi_r_f - psi_r_prev_f) / out_dtype(dt)
    dpsi_i_dt = (psi_i_f - psi_i_prev_f) / out_dtype(dt)
    j0 = psi_r_f * dpsi_i_dt - psi_i_f * dpsi_r_dt

    jx = noether_spatial_current(psi_r_f, psi_i_f, axis=0)
    jy = noether_spatial_current(psi_r_f, psi_i_f, axis=1)
    jz = noether_spatial_current(psi_r_f, psi_i_f, axis=2)

    amp_sq = psi_r_f * psi_r_f + psi_i_f * psi_i_f
    amp_safe = np.maximum(amp_sq, out_dtype(amplitude_floor))
    c2 = out_dtype(c_speed * c_speed)
    energy = 0.5 * (j0 * j0 + c2 * (jx * jx + jy * jy + jz * jz)) / amp_safe
    return energy.astype(out_dtype)


def phase_coherence(
    psi_r: NDArray,
    psi_i: NDArray,
    mask: NDArray | None = None,
) -> float:
    """Compute a scalar measure of phase coherence in a region.

    Returns the magnitude of the average complex amplitude normalised by
    the average modulus: C = |⟨Ψ⟩| / ⟨|Ψ|⟩.

    C = 1  →  perfectly coherent (all same phase).
    C = 0  →  completely incoherent (random phases cancel).

    Parameters
    ----------
    psi_r, psi_i : ndarray
        Real/imaginary parts of the field.
    mask : ndarray of bool or None
        If provided, only include True voxels.

    Returns
    -------
    coherence : float in [0, 1].
    """
    if mask is not None:
        pr = psi_r[mask]
        pi = psi_i[mask]
    else:
        pr = psi_r.ravel()
        pi = psi_i.ravel()

    modulus_mean = np.mean(np.sqrt(pr**2 + pi**2))
    if modulus_mean < 1e-30:
        return 0.0
    avg_r = np.mean(pr)
    avg_i = np.mean(pi)
    return float(np.sqrt(avg_r**2 + avg_i**2) / modulus_mean)


def coulomb_interaction_energy(
    psi_r: NDArray,
    psi_i: NDArray,
    psi_r_2: NDArray,
    psi_i_2: NDArray,
) -> float:
    """Compute the interference interaction energy between two complex fields.

    E_int = Σ 2·Re(Ψ₁* · Ψ₂) = 2Σ (psi_r₁·psi_r₂ + psi_i₁·psi_i₂)

    Positive → repulsion (same phase). Negative → attraction (opposite phase).

    Parameters
    ----------
    psi_r, psi_i : ndarray
        First field (real/imaginary).
    psi_r_2, psi_i_2 : ndarray
        Second field (real/imaginary).

    Returns
    -------
    energy : float
    """
    cross = 2.0 * np.sum(psi_r * psi_r_2 + psi_i * psi_i_2)
    return float(cross)
