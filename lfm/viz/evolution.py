"""Time-evolution plots from ``sim.history``."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_evolution(
    history: list[dict[str, float]],
    metrics: list[str] | None = None,
    *,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Simulation Evolution",
) -> Figure:
    """Plot selected metrics over time from ``sim.history``.

    Parameters
    ----------
    history : list[dict]
        ``sim.history`` — each dict must contain ``"step"`` and the
        requested metric keys.
    metrics : list[str] or None
        Metric keys to plot.  ``None`` → ``["chi_min", "well_fraction",
        "energy_total"]``.
    figsize : tuple
        Figure size.
    title : str
        Super-title.

    Returns
    -------
    fig : matplotlib Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("history is empty — run sim.run() first")

    if metrics is None:
        # Pick sensible defaults from whatever is available
        available = set(history[0].keys()) - {"step"}
        defaults = ["chi_min", "well_fraction", "energy_total"]
        metrics = [m for m in defaults if m in available]
        if not metrics:
            metrics = sorted(available)[:3]

    steps = [h.get("step", i) for i, h in enumerate(history)]
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    for i, key in enumerate(metrics):
        vals = [h.get(key, float("nan")) for h in history]
        ax: Axes = axes[i]
        ax.plot(steps, vals, color=colors[i % len(colors)], lw=1.5)
        ax.set_ylabel(key.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_energy_components(
    history: list[dict[str, float]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Stacked area plot of kinetic / gradient / potential energy.

    Requires ``energy_kinetic``, ``energy_gradient``, ``energy_potential``
    in each history dict (recorded when ``record_metrics=True``).
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("history is empty — run sim.run() first")

    steps = [h.get("step", i) for i, h in enumerate(history)]
    keys = ["energy_kinetic", "energy_gradient", "energy_potential"]
    present = [k for k in keys if k in history[0]]

    if not present:
        raise KeyError(
            f"History dicts don't contain energy components. "
            f"Available keys: {sorted(history[0].keys())}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = cast(Figure, ax.figure)

    import numpy as np

    data = np.array([[h.get(k, 0.0) for h in history] for k in present])
    colors = ["#3b82f6", "#22c55e", "#f59e0b"]
    ax.stackplot(steps, *data, labels=present, colors=colors[: len(present)], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Components")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax
