"""Trajectory plotting for tracked peaks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfm.analysis.tracker import flatten_trajectories
from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_trajectories(
    trajectories: list[list[dict[str, float]]],
    *,
    projection: str = "xy",
    title: str = "Peak Trajectories",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Scatter-plot of tracked peak positions.

    Parameters
    ----------
    trajectories : list[list[dict]]
        Output of :func:`lfm.analysis.tracker.track_peaks`.
    projection : str
        Which two axes to project onto: ``'xy'``, ``'xz'``, or ``'yz'``.
    title : str
        Plot title.
    ax : Axes or None
        Draw on existing axes.

    Returns
    -------
    fig, ax
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    flat = flatten_trajectories(trajectories)
    axis_map = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
    if projection not in axis_map:
        raise ValueError(f"projection must be one of {list(axis_map)}")

    a, b = axis_map[projection]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    steps = flat["step"]
    if steps.size == 0:
        ax.set_title(f"{title} (no peaks detected)")
        fig.tight_layout()
        return fig, ax

    sc = ax.scatter(
        flat[a], flat[b],
        c=steps,
        cmap="viridis",
        s=15,
        alpha=0.7,
        edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="Step")
    ax.set_xlabel(a.upper() + " (cells)")
    ax.set_ylabel(b.upper() + " (cells)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax
