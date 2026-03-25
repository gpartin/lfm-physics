"""
Discrete Klein-Gordon Dispersion Utilities
==========================================

Compute wavenumber, group velocity, phase velocity, and wavelength for
plane waves on the LFM 19-point stencil lattice.

The exact discrete dispersion relation for GOV-01 with the 19-point
Laplacian is::

    (2/Δt² ) (1 − cos(ωΔt)) = −λ̃(k) + χ₀²

where λ̃(k) is the eigenvalue of the 19-point stencil at wavevector k.
For a wave propagating along a single axis (k = (0, 0, K_z)), the
stencil eigenvalue simplifies to::

    λ̃ = (2/3)(cos K_z − 1) + (4/3)(cos K_z − 1) = 2(cos K_z − 1)

Substituting and solving for K_z:

    cos(ωΔt) = 1 − Δt²(χ₀² + 2 − 2 cos K_z)

For Δt = 1 (continuum-like, ω in rad/step): cos K_z = 1 − (ω² − χ₀²)/2

Group velocity (cells per step):

    v_g = dω/dK = sin(K_z) / (Δt · sin(ωΔt)/Δt)  (exact discrete)
        ≈ sin(K_z) / sin(ω·Δt) × Δt              (small Δt limit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Dispersion", "dispersion"]


@dataclass(frozen=True)
class Dispersion:
    """Discrete dispersion results for a monochromatic plane wave.

    All velocities are in *cells per leapfrog step* (multiply by
    ``1 / dt`` to convert to cells per unit time if needed).
    """

    omega: float
    """Drive frequency (rad per unit time)."""

    chi0: float
    """Effective mass parameter (χ₀)."""

    dt: float
    """Leapfrog timestep."""

    k_z: float
    """Axial wavenumber (rad per cell)."""

    wavelength: float
    """Spatial wavelength (cells per cycle)."""

    v_phase: float
    """Phase velocity (cells per step)."""

    v_group: float
    """Group velocity (cells per step)."""


def dispersion(
    *,
    omega: float | None = None,
    wavelength: float | None = None,
    chi0: float = 19.0,
    dt: float = 0.02,
) -> Dispersion:
    """Compute exact discrete dispersion for the 19-point KG stencil.

    Provide **either** ``omega`` (frequency) **or** ``wavelength``
    (desired spatial period in cells), not both.

    Parameters
    ----------
    omega : float or None
        Drive frequency in rad / time-unit.  Must satisfy the
        propagation condition: ``omega > chi0`` (otherwise the wave is
        evanescent).
    wavelength : float or None
        Desired wavelength in grid cells.  Converted to *k_z* first,
        then the matching ``omega`` is computed.
    chi0 : float
        Effective mass of the medium (default 19.0).
    dt : float
        Leapfrog timestep (default 0.02).

    Returns
    -------
    Dispersion
        Frozen dataclass with ``k_z``, ``wavelength``, ``v_phase``,
        ``v_group``, and the input ``omega``, ``chi0``, ``dt``.

    Raises
    ------
    ValueError
        If the wave is evanescent (omega ≤ chi0) or the requested
        parameters violate the Nyquist / CFL limits.

    Examples
    --------
    >>> d = dispersion(omega=2.0, chi0=1.0, dt=0.02)
    >>> round(d.wavelength, 1)
    3.0
    >>> round(d.v_group, 3)
    0.433
    """
    if (omega is None) == (wavelength is None):
        raise ValueError("Provide exactly one of omega= or wavelength=")

    if wavelength is not None:
        # wavelength → k_z → omega
        if wavelength <= 2.0:
            raise ValueError(f"wavelength={wavelength} < 2 cells (Nyquist limit)")
        k_z = 2.0 * math.pi / wavelength
        # From 19-point stencil: ω² = χ₀² + 2(1 − cos K_z)  (Δx=1 units)
        omega_sq = chi0**2 + 2.0 * (1.0 - math.cos(k_z))
        if omega_sq <= 0:
            raise ValueError("Evanescent: computed ω² ≤ 0")
        omega = math.sqrt(omega_sq)
    else:
        assert omega is not None
        # omega → k_z
        if omega <= abs(chi0):
            raise ValueError(
                f"Evanescent: omega={omega} ≤ chi0={chi0}. "
                f"Wave cannot propagate; increase omega or decrease chi0."
            )
        cos_kz = 1.0 - (omega**2 - chi0**2) / 2.0
        if cos_kz < -1.0 or cos_kz > 1.0:
            raise ValueError(
                f"cos(K_z) = {cos_kz:.4f} out of range [-1, 1]. "
                f"omega={omega} is above the lattice Nyquist cutoff."
            )
        k_z = math.acos(cos_kz)

    wavelength_out = 2.0 * math.pi / k_z if k_z > 1e-15 else float("inf")

    # Phase and group velocities (cells per step)
    # v_phase = ω/k  in cells/time → multiply by dt for cells/step
    v_phase = (omega / k_z * dt) if k_z > 1e-15 else float("inf")

    # v_group = dω/dk = sin(k_z) / [sin(ω·dt) / dt]  (exact discrete)
    sin_kz = math.sin(k_z)
    omega_dt = omega * dt
    if abs(math.sin(omega_dt)) > 1e-15:
        v_group = sin_kz * dt / math.sin(omega_dt)
    else:
        # Small-angle limit: sin(ω·dt) ≈ ω·dt
        v_group = sin_kz / omega

    return Dispersion(
        omega=omega,
        chi0=chi0,
        dt=dt,
        k_z=k_z,
        wavelength=wavelength_out,
        v_phase=v_phase,
        v_group=v_group,
    )
