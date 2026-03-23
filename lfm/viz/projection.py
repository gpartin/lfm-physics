"""
Column-density projections
==========================

Reduce a 3-D field to 2-D by summing (or averaging / max-projecting)
along one axis, then visualise the result.  This is the standard
technique used in cosmological N-body and hydrodynamics codes to produce
"projected density" images.

Examples
--------
>>> proj = lfm.project_field(sim.chi, axis=2)          # 2-D array
>>> fig, ax = lfm.plot_projection(sim.chi, axis=2, log=False)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lfm.viz._util import _require_matplotlib


def project_field(
    field: NDArray[np.float32],
    axis: int = 2,
    method: str = "sum",
) -> NDArray[np.float64]:
    """Reduce a 3-D field to 2-D by projecting along *axis*.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        Input 3-D field.
    axis : int
        Axis to collapse (0 = x, 1 = y, 2 = z).
    method : str
        Reduction method: ``"sum"`` (column density), ``"mean"`` (average),
        or ``"max"`` (maximum-intensity projection).

    Returns
    -------
    ndarray, shape (N, N)
        Projected 2-D array.
    """
    arr = np.asarray(field, dtype=np.float64)
    if method == "sum":
        return arr.sum(axis=axis)
    elif method == "mean":
        return arr.mean(axis=axis)
    elif method == "max":
        return arr.max(axis=axis)
    else:
        raise ValueError(f"Unknown method '{method}'.  Use 'sum', 'mean', or 'max'.")


def plot_projection(
    field: NDArray[np.float32],
    axis: int = 2,
    method: str = "sum",
    log: bool = True,
    *,
    title: str | None = None,
    cmap: str = "inferno",
    colorbar: bool = True,
    ax: "Axes | None" = None,  # type: ignore[name-defined]  # noqa: F821
) -> "tuple[Figure, Axes]":  # type: ignore[name-defined]  # noqa: F821
    """Plot a column-density projection of a 3-D field.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        3-D field to project.
    axis : int
        Axis to project along (0 = x, 1 = y, 2 = z).
    method : str
        ``"sum"`` (default), ``"mean"``, or ``"max"``.
    log : bool
        If True, apply ``log10(|projection| + ε)`` to improve dynamic range.
    title : str or None
        Axes title.
    cmap : str
        Colourmap name.
    colorbar : bool
        If True, add a colourbar.
    ax : Axes or None
        Existing axes to draw on.  If None, a new figure is created.

    Returns
    -------
    (Figure, Axes)
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    proj = project_field(field, axis=axis, method=method)

    if log:
        data = np.log10(np.abs(proj) + 1e-12)
        label = f"log₁₀|{method}|"
    else:
        data = proj
        label = method

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    axis_names = ["x", "y", "z"]
    proj_axis = axis_names[axis]
    remaining = [n for n in axis_names if n != proj_axis]

    im = ax.imshow(data.T, origin="lower", cmap=cmap)
    ax.set_xlabel(remaining[0])
    ax.set_ylabel(remaining[1])

    if title is None:
        title = f"{method} projection along {proj_axis}"
    ax.set_title(title)

    if colorbar:
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    return fig, ax
