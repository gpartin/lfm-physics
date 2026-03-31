"""
Spinor Analysis Functions
=========================

Post-processing for two-component spinor simulations (``FieldLevel.COLOR``
with ``n_colors=2``).  Spinor components are stored as:

* ``psi_r[0]``, ``psi_i[0]``: Re(ψ_↑), Im(ψ_↑)
* ``psi_r[1]``, ``psi_i[1]``: Re(ψ_↓), Im(ψ_↓)

All functions accept arrays of shape ``(2, N, N, N)`` as returned by
``Evolver.get_psi_real()`` / ``get_psi_imag()`` for a COLOR-level simulation
with ``n_colors=2``.

Reference: LFM-PAPER-048 (Spinor Representation in the Lattice Field Medium)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spinor_density(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Total spinor probability density |ψ|² = |ψ_↑|² + |ψ_↓|².

    This is the source term for GOV-02 (gravity is spin-blind).

    Parameters
    ----------
    psi_r, psi_i : ndarray, shape (2, N, N, N)

    Returns
    -------
    density : ndarray, shape (N, N, N)
    """
    return (psi_r[0] ** 2 + psi_i[0] ** 2 + psi_r[1] ** 2 + psi_i[1] ** 2).astype(
        np.float32
    )


def spinor_sigma_z(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
) -> float:
    """Volume-averaged z-polarisation ⟨σ_z⟩.

    ⟨σ_z⟩ = (∫|ψ_↑|² dV − ∫|ψ_↓|² dV) / ∫|ψ|² dV

    Returns +1 for pure spin-up, −1 for pure spin-down, 0 for unpolarised.

    Parameters
    ----------
    psi_r, psi_i : ndarray, shape (2, N, N, N)

    Returns
    -------
    sigma_z : float in [−1, +1]
    """
    up_norm = float(np.sum(psi_r[0] ** 2 + psi_i[0] ** 2))
    dn_norm = float(np.sum(psi_r[1] ** 2 + psi_i[1] ** 2))
    total = up_norm + dn_norm
    return (up_norm - dn_norm) / total if total > 0.0 else 0.0


def spinor_sigma_x(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
) -> float:
    """Volume-averaged x-coherence ⟨σ_x⟩.

    ⟨σ_x⟩ = 2 Re[∫ ψ_↑* ψ_↓ dV] / ∫|ψ|² dV

    For a state R_z(φ)|↑⟩ = (e^{−iφ/2}, 0) this equals sin(−φ/2) × ...
    For a state R_y(φ)|↑⟩ = (cos(φ/2), −sin(φ/2)) this equals sin(φ).

    Parameters
    ----------
    psi_r, psi_i : ndarray, shape (2, N, N, N)

    Returns
    -------
    sigma_x : float in [−1, +1]
    """
    # Re[ψ_↑* ψ_↓] = Re[(u_r − i u_i)(d_r + i d_i)] = u_r d_r + u_i d_i
    cross_real = float(np.sum(psi_r[0] * psi_r[1] + psi_i[0] * psi_i[1]))
    total = float(
        np.sum(psi_r[0] ** 2 + psi_i[0] ** 2 + psi_r[1] ** 2 + psi_i[1] ** 2)
    )
    return 2.0 * cross_real / total if total > 0.0 else 0.0


def spinor_interference_energy(
    psi_r_a: NDArray[np.float32],
    psi_i_a: NDArray[np.float32],
    psi_r_b: NDArray[np.float32],
    psi_i_b: NDArray[np.float32],
) -> float:
    """Total energy of the combined spinor state |ψ_A + ψ_B|².

    Used for the 720° interferometry test.  The interference term depends
    on the relative rotation between the two spinors:

    * Scalar phase rotation φ:  |ψ_A + ψ_B|² ∝ 1 + cos(φ)  → period 360°
    * Spinor rotation φ:        |ψ_A + ψ_B|² ∝ 1 + cos(φ/2) → period 720°

    At φ = 360°: scalar gives constructive interference; spinor gives
    *destructive* interference (the 720° smoking gun).

    Parameters
    ----------
    psi_r_a, psi_i_a : ndarray, shape (2, N, N, N)  — state A
    psi_r_b, psi_i_b : ndarray, shape (2, N, N, N)  — state B

    Returns
    -------
    energy : float  (sum of |ψ_A + ψ_B|² over all lattice sites)
    """
    sumr = psi_r_a + psi_r_b
    sumi = psi_i_a + psi_i_b
    return float(np.sum(sumr ** 2 + sumi ** 2))


def spinor_sigma_y(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
) -> float:
    """Volume-averaged y-coherence ⟨σ_y⟩.

    ⟨σ_y⟩ = 2 ∫ (Re(ψ_↑)·Im(ψ_↓) − Im(ψ_↑)·Re(ψ_↓)) dV / ∫|ψ|² dV

    Derivation:
        ⟨σ_y⟩ = ψ†σ_yψ = ψ_↑*(−iψ_↓) + ψ_↓*(iψ_↑)
               = 2·Re[ i · ψ_↓*ψ_↑ ]
               = 2·(u_r d_i − u_i d_r)  where u = ψ_↑, d = ψ_↓

    Reference states:
        * |-y⟩ = R_x(π/2)|↑⟩  → ⟨σ_y⟩ = −1  (R_x rotates Bloch vec to −y pole)
        * |+y⟩                  → ⟨σ_y⟩ = +1

    Parameters
    ----------
    psi_r, psi_i : ndarray, shape (2, N, N, N)

    Returns
    -------
    sigma_y : float in [−1, +1]
    """
    cross = float(np.sum(psi_r[0] * psi_i[1] - psi_i[0] * psi_r[1]))
    total = float(
        np.sum(psi_r[0] ** 2 + psi_i[0] ** 2 + psi_r[1] ** 2 + psi_i[1] ** 2)
    )
    return 2.0 * cross / total if total > 0.0 else 0.0


def spinor_center_of_energy(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
    axis: int = 2,
) -> float:
    """Center-of-energy coordinate along a given axis.

    Used to measure Stern-Gerlach deflection.

    Parameters
    ----------
    psi_r, psi_i : ndarray, shape (2, N, N, N)
    axis : int
        Axis along which to compute CoE (0=x, 1=y, 2=z).

    Returns
    -------
    center : float (in grid units)
    """
    density = spinor_density(psi_r, psi_i)
    N = density.shape[0]
    coords = np.arange(N, dtype=np.float32)
    total = float(np.sum(density))
    if total == 0.0:
        return float(N) / 2.0
    if axis == 0:
        weight = np.einsum("ijk,i->ijk", density, coords)
    elif axis == 1:
        weight = np.einsum("ijk,j->ijk", density, coords)
    else:
        weight = np.einsum("ijk,k->ijk", density, coords)
    return float(np.sum(weight)) / total
