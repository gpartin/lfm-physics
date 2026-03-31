"""
Spinor Field Initializers
=========================

Create two-component spinor (Dirac) initial conditions for spin-1/2 LFM
simulations.

In LFM, a spin-1/2 particle is represented by two complex field components
stored as ``FieldLevel.COLOR`` with ``n_colors=2``:

* Component 0 (color index 0): ψ_↑  (spin-up)
* Component 1 (color index 1): ψ_↓  (spin-down)

Each component obeys the same GOV-01 (Klein-Gordon) equation.  The
shared χ field is sourced by the *total* spinor density
|ψ|² = |ψ_↑|² + |ψ_↓|² via GOV-02 (spin-blind gravity).

The fundamental spinor prediction is 720° rotational periodicity:

* Scalar fields return to original state after 360° phase rotation.
* Spinor fields acquire a sign flip after 360° and return after 720°.
  This is measurable via interference (see ``spinor_interference_energy``
  in ``lfm.analysis.spinor``).

Reference: LFM-PAPER-048 (Spinor Representation in the Lattice Field Medium)
"""

from __future__ import annotations

import numpy as np

from numpy.typing import NDArray


def gaussian_spinor(
    N: int,
    position: tuple[float, float, float],
    amplitude: float,
    sigma: float,
    spin_up: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Gaussian spinor soliton with pure spin-up or spin-down state.

    Creates a Gaussian envelope in one spinor component with the other
    component set to zero.  No spatial vortex winding — suitable for
    the interferometry test (Experiment 1 of Paper 048).

    Parameters
    ----------
    N : int
        Grid size per axis.
    position : (x, y, z)
        Center in grid coordinates.
    amplitude : float
        Peak amplitude of the Gaussian envelope.
    sigma : float
        Gaussian width in grid cells.
    spin_up : bool
        If True, populate ψ_↑ (color 0); if False, populate ψ_↓ (color 1).

    Returns
    -------
    psi_r, psi_i : ndarray of float32, shape (2, N, N, N)
        Real and imaginary parts.  Index 0 = ψ_↑, index 1 = ψ_↓.
        For pure spin-up/down, the imaginary part is zero everywhere.
    """
    x = np.arange(N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    px, py, pz = position
    r2 = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
    envelope = (amplitude * np.exp(-r2 / (2.0 * sigma ** 2))).astype(np.float32)

    psi_r = np.zeros((2, N, N, N), dtype=np.float32)
    psi_i = np.zeros((2, N, N, N), dtype=np.float32)

    idx = 0 if spin_up else 1
    psi_r[idx] = envelope
    return psi_r, psi_i


def vortex_spinor(
    N: int,
    position: tuple[float, float, float],
    amplitude: float,
    sigma: float,
    winding: float = 0.5,
    spin_up: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Spinor vortex with half-integer winding for Stern-Gerlach tests.

    Creates a vortex in the spin-up or spin-down component.  The vortex
    lies in the x-y plane with winding angle φ = ``winding`` × arctan2(y, x).

    * ``winding = 0.5`` (default): half-integer vortex → true spin-1/2.
      ψ_↑ = f(r) × e^{+i θ/2} where θ = arctan2(y − cy, x − cx).
    * ``winding = 1.0``: integer vortex (spin-1 behaviour, as in Paper 047).

    Parameters
    ----------
    N : int
        Grid size per axis.
    position : (x, y, z)
        Center of the vortex in grid coordinates.
    amplitude : float
        Peak amplitude.
    sigma : float
        Gaussian envelope width.
    winding : float
        Vortex winding number.  Use 0.5 for spin-1/2.
    spin_up : bool
        If True, populate ψ_↑; negative winding goes in ψ_↓.

    Returns
    -------
    psi_r, psi_i : ndarray of float32, shape (2, N, N, N)
    """
    x = np.arange(N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    px, py, pz = position

    r2 = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
    envelope = (amplitude * np.exp(-r2 / (2.0 * sigma ** 2))).astype(np.float32)

    # Azimuthal angle in x-y plane around the vortex centre
    theta = np.arctan2(Y - py, X - px).astype(np.float32)
    phase = (winding * theta).astype(np.float32)

    psi_r = np.zeros((2, N, N, N), dtype=np.float32)
    psi_i = np.zeros((2, N, N, N), dtype=np.float32)

    idx = 0 if spin_up else 1
    psi_r[idx] = envelope * np.cos(phase)
    psi_i[idx] = envelope * np.sin(phase)
    return psi_r, psi_i


def apply_rotation_x(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
    phi: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply the spinor rotation R_x(φ) = exp(−i φ σ_x / 2).

    Rotates the spinor by angle φ about the x-axis.  Used to prepare
    states for the 720° interferometry test.

    Transformation (exact, no approximation)::

        ψ_↑′ =  cos(φ/2) ψ_↑  −  i sin(φ/2) ψ_↓
        ψ_↓′ = −i sin(φ/2) ψ_↑ +   cos(φ/2) ψ_↓

    In real/imaginary components (c = cos(φ/2), s = sin(φ/2))::

        Re(ψ_↑′) = c·Re(ψ_↑) + s·Im(ψ_↓)
        Im(ψ_↑′) = c·Im(ψ_↑) − s·Re(ψ_↓)
        Re(ψ_↓′) = s·Im(ψ_↑) + c·Re(ψ_↓)
        Im(ψ_↓′) = −s·Re(ψ_↑) + c·Im(ψ_↓)

    Key property: R_x(2π)|↑⟩ = −|↑⟩  (sign flip, not identity).
                  R_x(4π)|↑⟩ = +|↑⟩  (720° returns to original).

    Parameters
    ----------
    psi_r, psi_i : ndarray, shape (2, N, N, N)
        Spinor field to rotate.
    phi : float
        Rotation angle in radians.

    Returns
    -------
    new_r, new_i : ndarray, shape (2, N, N, N)
        Rotated spinor.
    """
    c = float(np.cos(phi / 2.0))
    s = float(np.sin(phi / 2.0))

    u_r, d_r = psi_r[0], psi_r[1]
    u_i, d_i = psi_i[0], psi_i[1]

    new_r = np.zeros_like(psi_r)
    new_i = np.zeros_like(psi_i)

    new_r[0] = c * u_r + s * d_i
    new_i[0] = c * u_i - s * d_r
    new_r[1] = s * u_i + c * d_r
    new_i[1] = -s * u_r + c * d_i

    return new_r, new_i


def apply_rotation_z(
    psi_r: NDArray[np.float32],
    psi_i: NDArray[np.float32],
    phi: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply the spinor rotation R_z(φ) = exp(−i φ σ_z / 2).

    Rotates the spinor by angle φ about the z-axis.

    Transformation::

        ψ_↑′ = e^{−iφ/2} ψ_↑
        ψ_↓′ = e^{+iφ/2} ψ_↓

    In real/imaginary components::

        Re(ψ_↑′) =  cos(φ/2) Re(ψ_↑) + sin(φ/2) Im(ψ_↑)
        Im(ψ_↑′) = −sin(φ/2) Re(ψ_↑) + cos(φ/2) Im(ψ_↑)
        Re(ψ_↓′) =  cos(φ/2) Re(ψ_↓) − sin(φ/2) Im(ψ_↓)
        Im(ψ_↓′) =  sin(φ/2) Re(ψ_↓) + cos(φ/2) Im(ψ_↓)

    Parameters
    ----------
    psi_r, psi_i : ndarray, shape (2, N, N, N)
    phi : float
        Rotation angle in radians.

    Returns
    -------
    new_r, new_i : ndarray, shape (2, N, N, N)
    """
    c = float(np.cos(phi / 2.0))
    s = float(np.sin(phi / 2.0))

    u_r, d_r = psi_r[0], psi_r[1]
    u_i, d_i = psi_i[0], psi_i[1]

    new_r = np.zeros_like(psi_r)
    new_i = np.zeros_like(psi_i)

    # e^{-iφ/2} ψ_↑ = (c - i s)(u_r + i u_i) = (c u_r + s u_i) + i(c u_i - s u_r)
    new_r[0] = c * u_r + s * u_i
    new_i[0] = c * u_i - s * u_r

    # e^{+iφ/2} ψ_↓ = (c + i s)(d_r + i d_i) = (c d_r - s d_i) + i(c d_i + s d_r)
    new_r[1] = c * d_r - s * d_i
    new_i[1] = c * d_i + s * d_r

    return new_r, new_i
