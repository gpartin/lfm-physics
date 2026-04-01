"""Power-spectrum plot."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from lfm.analysis.spectrum import power_spectrum
from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_power_spectrum(
    field: NDArray,
    *,
    bins: int = 50,
    log: bool = True,
    title: str = "Power Spectrum",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Radially-averaged power spectrum P(k) of a 3-D field.

    Parameters
    ----------
    field : ndarray (N, N, N)
        Scalar field.
    bins : int
        Number of radial k-bins.
    log : bool
        Log-log axes.
    title : str
        Title.
    ax : Axes or None
        Draw on existing axes.

    Returns
    -------
    fig, ax
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    ps = power_spectrum(field, bins=bins)
    k = ps["k"]
    P = ps["power"]

    # Drop bins with no modes
    mask = ps["counts"] > 0
    k, P = k[mask], P[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = cast("Figure", ax.figure)

    if log and np.all(P[P > 0]):
        ax.loglog(k, P, "o-", ms=3, color="#7c3aed", lw=1.4)
    else:
        ax.plot(k, P, "o-", ms=3, color="#7c3aed", lw=1.4)

    ax.set_xlabel("k (modes)")
    ax.set_ylabel("P(k)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig, ax
