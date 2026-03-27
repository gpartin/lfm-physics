"""Color analysis functions.

Provides:
- ``color_variance`` — normalized f_c for color classification (v14)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def color_variance(
    psi_r: NDArray,
    psi_i: NDArray,
    n_colors: int = 3,
) -> dict[str, NDArray | float]:
    """Compute the normalized color variance f_c at each lattice point.

    f_c = [Σ_a |Ψ_a|⁴ / (Σ_a |Ψ_a|²)²] − 1/N_c

    Parameters
    ----------
    psi_r : ndarray (N_c*N³,) or (N_c, N, N, N)
        Real parts of color field components (flat or shaped).
    psi_i : ndarray
        Imaginary parts (same shape as psi_r).
    n_colors : int
        Number of color components (default 3).

    Returns
    -------
    dict with keys:
        'f_c'       — 3D array (N,N,N) of normalized color variance
        'f_c_mean'  — scalar mean
        'f_c_max'   — scalar max
    """
    if psi_r.ndim == 1:
        total = psi_r.shape[0] // n_colors
        N = round(total ** (1 / 3))
        ea_list = []
        for a in range(n_colors):
            s = slice(a * total, (a + 1) * total)
            ea_list.append(psi_r[s] ** 2 + psi_i[s] ** 2)
        ea = np.stack(ea_list, axis=0)  # (n_colors, total)
    else:
        N = psi_r.shape[1]
        total = N**3
        ea = psi_r.reshape(n_colors, -1) ** 2 + psi_i.reshape(n_colors, -1) ** 2

    total_sq = np.sum(ea, axis=0)  # (total,)
    sum_sq = np.sum(ea**2, axis=0)
    safe = total_sq > 1e-30
    denom = np.where(safe, total_sq**2, 1.0)
    f_c = np.where(safe, sum_sq / denom - 1.0 / n_colors, 0.0)
    f_c_3d = f_c.reshape(N, N, N)

    return {
        "f_c": f_c_3d,
        "f_c_mean": float(np.mean(f_c)),
        "f_c_max": float(np.max(f_c)),
    }
