"""
Gravitational-wave extraction
=============================

Extract gravitational-wave signals from LFM field data using the
*weak-field strain* and *quadrupole radiation* formulae.

In LFM the gravitational-wave content is encoded in perturbations of the χ
field away from its vacuum value χ₀:

.. math::

    h(\\mathbf{x}) = \\frac{\\chi(\\mathbf{x}) - \\chi_0}{\\chi_0}

The standard quadrupole formula (Peters 1964, in natural units) gives the
gravitational-wave luminosity radiated by an accelerating mass distribution
from the second time derivative of the reduced mass quadrupole moment.

References
----------
* Peters 1964, Phys. Rev. 136, B1224
* Alcubierre 2008, *Introduction to 3+1 Numerical Relativity*
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lfm.constants import CHI0


def gravitational_wave_strain(
    chi: NDArray,
    chi0: float = CHI0,
) -> NDArray[np.float64]:
    """Compute the dimensionless gravitational-wave strain field h(x).

    .. math::

        h(\\mathbf{x}) = \\frac{\\chi(\\mathbf{x}) - \\chi_0}{\\chi_0}

    Parameters
    ----------
    chi : ndarray, shape (N, N, N)
        Current χ field.
    chi0 : float
        Vacuum value χ₀ (default :data:`lfm.CHI0` = 19).

    Returns
    -------
    ndarray, shape (N, N, N)
        Dimensionless strain h ∈ (−1, +∞).
    """
    chi_arr = np.asarray(chi, dtype=np.float64)
    return (chi_arr - chi0) / chi0


def gw_quadrupole(
    energy_density: NDArray,
    center: tuple[float, float, float] | None = None,
) -> NDArray[np.float64]:
    """Reduced mass quadrupole moment tensor I_{ij}.

    .. math::

        I_{ij} = \\int \\rho(\\mathbf{x})
                  \\left(x_i x_j - \\frac{\\delta_{ij}}{3} r^2\\right)
                  d^3x

    The traced-reversed (TT-gauge) part determines gravitational wave
    emission via the quadrupole formula.

    Parameters
    ----------
    energy_density : ndarray, shape (N, N, N)
        |Ψ|² energy density (proxy for mass-energy density ρ).
    center : (cx, cy, cz) or None
        Centre of mass for the coordinate system.  If None, uses the
        energy-weighted centre of the grid.

    Returns
    -------
    ndarray, shape (3, 3)
        Symmetric traceless reduced quadrupole tensor I_{ij}.
    """
    rho = np.asarray(energy_density, dtype=np.float64)
    N = rho.shape[0]

    coords = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    if center is None:
        total_mass = float(rho.sum())
        if total_mass == 0.0:
            total_mass = 1.0
        cx = float((rho * X).sum()) / total_mass
        cy = float((rho * Y).sum()) / total_mass
        cz = float((rho * Z).sum()) / total_mass
    else:
        cx, cy, cz = center

    dx = X - cx
    dy = Y - cy
    dz = Z - cz
    r2 = dx**2 + dy**2 + dz**2

    coords_vec = [dx, dy, dz]
    quad = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            delta_ij = 1.0 if i == j else 0.0
            quad[i, j] = float((rho * (coords_vec[i] * coords_vec[j] - delta_ij / 3.0 * r2)).sum())

    return quad


def gw_power(
    snapshots: list[dict],
    field: str = "energy_density",
    center: tuple[float, float, float] | None = None,
    dt: float = 0.02,
) -> dict[str, NDArray[np.float64]]:
    """Gravitational-wave luminosity L_{GW}(t) from snapshot sequence.

    Uses the Peters quadrupole formula:

    .. math::

        L_{GW} = \\frac{1}{5} \\left\\langle
            \\dddot{I}_{ij} \\dddot{I}_{ij}
        \\rangle

    approximated by finite differences of the quadrupole moment computed
    at each snapshot.

    Parameters
    ----------
    snapshots : list of dict
        Sequence from :meth:`lfm.Simulation.run_with_snapshots`.
        Each dict must contain *field*.
    field : str
        Energy-density field key (default ``"energy_density"``).
    center : (cx, cy, cz) or None
        Fixed centre for the coordinate system.  If None, recomputed
        at each frame using the energy-weighted centre.
    dt : float
        Physical time step between consecutive snapshots (simulation
        time per snapshot, not per leapfrog step).

    Returns
    -------
    dict with keys:
        ``t``           — time array (simulation units)
        ``luminosity``  — L_{GW}(t) at each valid frame
        ``I_tensor``    — quadrupole moments, shape (n_frames, 3, 3)
    """
    if len(snapshots) < 3:
        raise ValueError("At least 3 snapshots are required for GW power (need 2nd derivative).")
    if field not in snapshots[0]:
        available = list(snapshots[0].keys())
        raise KeyError(
            f"Field '{field}' not found.  Request it in run_with_snapshots(fields=['{field}']). "
            f"Available: {available}"
        )

    # Compute quadrupole at each frame
    I_series = np.array(
        [gw_quadrupole(snap[field], center=center) for snap in snapshots]
    )  # shape: (n_frames, 3, 3)

    n = len(I_series)
    # Second derivative via central finite differences (valid for interior frames)
    I_ddot = np.zeros_like(I_series)
    for k in range(1, n - 1):
        I_ddot[k] = (I_series[k + 1] - 2.0 * I_series[k] + I_series[k - 1]) / dt**2

    # Third derivative (forward/backward for edges)
    I_dddot = np.zeros_like(I_series)
    for k in range(1, n - 1):
        I_dddot[k] = (I_ddot[k + 1] - I_ddot[k - 1]) / (2.0 * dt) if k + 1 < n else 0.0

    # L_GW = (1/5) Tr(I_dddot · I_dddot) per Peters formula
    luminosity = np.array(
        [float(np.einsum("ij,ij->", I_dddot[k], I_dddot[k])) / 5.0 for k in range(n)]
    )

    # Snapshot times (from "step" key if present, else index * dt)
    t_steps = np.array([snap.get("step", i) for i, snap in enumerate(snapshots)], dtype=np.float64)
    if snapshots[0].get("step", 0) != 0:
        # steps → times: step * internal_dt, but we don't know internal_dt here;
        # just use index order scaled by dt
        t = np.arange(n, dtype=np.float64) * dt
    else:
        t = t_steps * dt

    return {
        "t": t,
        "luminosity": luminosity,
        "I_tensor": I_series,
    }
