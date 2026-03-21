"""Sweep result plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_sweep(
    results: list[dict[str, Any]],
    x_param: str,
    y_metric: str,
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one metric against the swept parameter.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`lfm.sweep.sweep`.
    x_param : str
        Key for the x-axis (the varied parameter).
    y_metric : str
        Key for the y-axis (the measured metric).
    title : str or None
        Plot title.
    ax : Axes or None
        Draw on existing axes.

    Returns
    -------
    fig, ax
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    xs = [r[x_param] for r in results]
    ys = [r[y_metric] for r in results]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.plot(xs, ys, "o-", ms=6, lw=1.5, color="#2563eb")
    ax.set_xlabel(x_param.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    if title is None:
        title = f"{y_metric} vs {x_param}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
