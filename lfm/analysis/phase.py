"""
Phase / Charge Analysis
=======================

Extract electromagnetic properties from complex wave-field phase.

In LFM, charge = phase θ of the complex wave function.
θ = 0 → electron (negative), θ = π → positron (positive).
Same-phase → repel (constructive), opposite-phase → attract (destructive).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
