"""2-D slice and histogram plots for 3-D lattice fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_slice(
    field: NDArray,
    axis: int = 2,
    index: int | None = None,
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a 2-D slice through a 3-D field.

    Parameters
    ----------
    field : ndarray of shape (N, N, N)
        Scalar field (e.g. ``sim.chi`` or ``sim.energy_density``).
    axis : int
        Axis perpendicular to the slice (0=x, 1=y, 2=z).
    index : int or None
        Index along *axis*.  ``None`` → middle of the grid.
    title : str or None
        Plot title.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Color limits.
    colorbar : bool
        Whether to add a colorbar.
    ax : matplotlib Axes or None
        Draw on an existing axes.  If ``None`` a new figure is created.

    Returns
    -------
    fig, ax
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if field.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {field.shape}")

    if index is None:
        index = field.shape[axis] // 2

    slc: list[Any] = [slice(None)] * 3
    slc[axis] = index
    data = field[tuple(slc)]

    axis_labels = [("Y", "Z"), ("X", "Z"), ("X", "Y")]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(
        data.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    xl, yl = axis_labels[axis]
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if title is not None:
        ax.set_title(title)
    else:
        axis_name = "XYZ"[axis]
        ax.set_title(f"Slice at {axis_name}={index}")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, ax


def plot_three_slices(
    field: NDArray,
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Three orthogonal slices through the centre of a 3-D field.

    Returns a figure with three panels (XY, XZ, YZ).
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, ax in enumerate(axes):
        plot_slice(field, axis=axis, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=True, ax=ax)
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_chi_histogram(
    chi: NDArray,
    *,
    bins: int = 100,
    title: str = "χ Distribution",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of χ values across the lattice.

    Useful for seeing the split between wells (χ < 17) and voids (χ ≈ 19).
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    flat = np.asarray(chi).ravel()
    ax.hist(flat, bins=bins, color="#4a90d9", edgecolor="none", alpha=0.85)
    ax.axvline(19.0, color="gray", ls="--", lw=1, label="χ₀ = 19")
    ax.axvline(17.0, color="red", ls=":", lw=1, label="well threshold (17)")
    ax.set_xlabel("χ")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, ax
