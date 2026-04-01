"""Radial-profile visualisation with optional 1/r reference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from lfm.analysis.observables import radial_profile
from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_radial_profile(
    chi: NDArray,
    center: tuple[int, int, int] | None = None,
    *,
    max_radius: int | None = None,
    reference: str | None = "1/r",
    title: str = "Radial χ Profile",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot χ(r) around *center* with a 1/r reference line.

    Parameters
    ----------
    chi : ndarray (N, N, N)
        The χ field.
    center : tuple or None
        Centre of the profile.  ``None`` → grid centre.
    max_radius : int or None
        Maximum radius in cells.
    reference : '1/r' | None
        Overlay a best-fit 1/r curve.
    title : str
        Plot title.
    ax : Axes or None
        Draw on an existing axes.

    Returns
    -------
    fig, ax
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    N = chi.shape[0]
    if center is None:
        center = (N // 2, N // 2, N // 2)
    if max_radius is None:
        max_radius = N // 2 - 2

    prof = radial_profile(chi, center, max_radius=max_radius)
    r = np.asarray(prof["r"], dtype=float)
    vals = np.asarray(prof["profile"], dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = cast("Figure", ax.figure)

    ax.plot(r, vals, "o-", ms=3, color="#2563eb", lw=1.5, label="χ(r)")

    if reference == "1/r" and len(r) > 2:
        chi0 = float(chi.max())
        delta = chi0 - vals
        # Fit: Δχ ≈ A / r using r >= 2 to avoid core
        mask = r >= 2.0
        if mask.sum() >= 2:
            r_fit = r[mask]
            d_fit = delta[mask]
            A = float(np.mean(d_fit * r_fit))
            ref_vals = chi0 - A / np.clip(r, 1, None)
            ax.plot(r, ref_vals, "--", color="gray", lw=1, alpha=0.7, label=f"1/r fit (A={A:.2f})")

    ax.set_xlabel("Radius (cells)")
    ax.set_ylabel("χ")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
