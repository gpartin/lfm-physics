"""
Space–time diagrams
===================

Visualise how a 1-D slice of a field evolves over time by stacking
successive snapshots into a 2-D heatmap with space on the horizontal
axis and simulation time on the vertical axis.

This is a classic diagnostic for wave propagation, soliton collisions,
parametric resonance, and perturbation spreading.

Examples
--------
>>> snaps = sim.run_with_snapshots(4000, snapshot_every=50)
>>> fig, ax = lfm.spacetime_diagram(snaps)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from numpy.typing import NDArray


def spacetime_diagram(
    snapshots: list[dict],
    field: str = "chi",
    axis: int = 0,
    center: int | None = None,
    *,
    dt: float = 1.0,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    title: str | None = None,
    ax: Axes | None = None,  # type: ignore[name-defined]  # noqa: F821
) -> tuple[Figure, Axes]:  # type: ignore[name-defined]  # noqa: F821
    """Plot a χ(x, t) space–time diagram from a snapshot sequence.

    Extracts a 1-D pencil at the grid mid-planes (or *center*) from each
    snapshot, stacks them row-by-row, and displays as an image where the
    horizontal axis is the spatial coordinate and the vertical axis is time.

    Parameters
    ----------
    snapshots : list of dict
        Sequence from :meth:`lfm.Simulation.run_with_snapshots`.
        Each dict must contain *field*.
    field : str
        Which field to display (default ``"chi"``).
    axis : int
        Spatial axis for the 1-D pencil (0 = x, 1 = y, 2 = z).
    center : int or None
        Fixed index along the two orthogonal axes.  Defaults to midpoint.
    dt : float
        Time per snapshot (in simulation units).  Sets the y-axis scale.
    cmap : str
        Matplotlib colourmap.
    vmin, vmax : float or None
        Colour scale.  Defaults to the data range.
    colorbar : bool
        Whether to add a colourbar.
    title : str or None
        Axes title.
    ax : Axes or None
        Existing axes to draw on; creates a new figure if None.

    Returns
    -------
    (Figure, Axes)
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not snapshots:
        raise ValueError("snapshots list is empty")
    if field not in snapshots[0]:
        available = list(snapshots[0].keys())
        raise KeyError(f"Field '{field}' not in snapshots.  Available: {available}.")

    rows: list[NDArray[np.float32]] = []
    for snap in snapshots:
        arr: NDArray[np.float32] = snap[field]
        n = arr.shape[0]
        c = center if center is not None else n // 2
        # Take pencil along `axis` at the centre of the other two axes
        if axis == 0:
            pencil = arr[:, c, c]
        elif axis == 1:
            pencil = arr[c, :, c]
        else:
            pencil = arr[c, c, :]
        rows.append(pencil)

    data = np.stack(rows, axis=0)  # shape: (n_frames, N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    n_frames, n_space = data.shape
    extent = [0, n_space, 0, n_frames * dt]
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    axis_names = ["x", "y", "z"]
    ax.set_xlabel(f"{axis_names[axis]} (grid cells)")
    ax.set_ylabel("time (simulation units)")

    if title is None:
        title = f"{field} space–time diagram"
    ax.set_title(title)

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax
