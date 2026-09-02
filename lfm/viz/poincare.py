"""Publication plots for the Poincare-emergence audit."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from collections.abc import Iterable


def _finish(figure, output: str | Path | None):
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
    return figure


def plot_symmetry_convergence(
    rows: Iterable[dict],
    *,
    output: str | Path | None = None,
):
    """Plot dispersion, rotational, boost, and velocity-addition convergence."""

    _require_matplotlib()
    import matplotlib.pyplot as plt

    records = list(rows)
    metrics = [
        ("dispersion_error_max", "Dispersion defect"),
        ("directional_anisotropy", "Rotation defect"),
        ("boosted_mass_shell_residual_max", "Boost mass-shell defect"),
        ("velocity_addition_error_max_over_c", "Velocity-addition defect"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), sharex=True)
    colors = {"19": "#1f5a99", "27": "#c25a2c"}
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        for stencil in ("19", "27"):
            subset = sorted(
                (row for row in records if row["stencil"] == stencil),
                key=lambda row: row["spacing"],
            )
            spacing = np.asarray([row["spacing"] for row in subset])
            values = np.asarray([max(row[metric], 1.0e-18) for row in subset])
            axis.loglog(
                spacing,
                values,
                "o-",
                color=colors[stencil],
                label=f"{stencil}-point",
                linewidth=1.6,
                markersize=4,
            )
        axis.set_title(label)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_ylabel("dimensionless error")
    for axis in axes[-1, :]:
        axis.set_xlabel("lattice spacing h (fixed physical k)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Recovery of continuum Poincare kinematics")
    figure.tight_layout()
    return _finish(figure, output)


def plot_directional_dispersion(
    rows: Iterable[dict],
    *,
    output: str | Path | None = None,
):
    """Plot group-speed defects along axis, face, and body directions."""

    _require_matplotlib()
    import matplotlib.pyplot as plt

    records = list(rows)
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    colors = {"axis": "#1f5a99", "face": "#4f9c73", "body": "#c25a2c"}
    for axis, stencil in zip(axes, ("19", "27"), strict=True):
        for direction in ("axis", "face", "body"):
            subset = sorted(
                (
                    row
                    for row in records
                    if row["stencil"] == stencil and row["direction"] == direction
                ),
                key=lambda row: row["kh"],
            )
            axis.plot(
                [row["kh"] for row in subset],
                [row["group_speed_over_c"] - 1.0 for row in subset],
                "o-",
                color=colors[direction],
                label=direction,
                linewidth=1.5,
                markersize=3.5,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(f"{stencil}-point stencil")
        axis.set_xlabel("dimensionless wavenumber kh")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("radial group speed / c - 1")
    axes[0].legend(frameon=False)
    figure.suptitle("Finite-spacing light-cone deformation")
    figure.tight_layout()
    return _finish(figure, output)


def plot_packet_convergence(
    rows: Iterable[dict],
    *,
    output: str | Path | None = None,
):
    """Plot Gaussian-packet propagation error over grid refinements."""

    _require_matplotlib()
    import matplotlib.pyplot as plt

    records = list(rows)
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    colors = {"axis": "#1f5a99", "face": "#4f9c73", "body": "#c25a2c"}
    for axis, stencil in zip(axes, ("19", "27"), strict=True):
        for direction in ("axis", "face", "body"):
            subset = sorted(
                (
                    row
                    for row in records
                    if row["stencil"] == stencil and row["direction"] == direction
                ),
                key=lambda row: row["spacing"],
            )
            axis.loglog(
                [row["spacing"] for row in subset],
                [max(row["radial_error"], 1.0e-16) for row in subset],
                "o-",
                color=colors[direction],
                label=direction,
                linewidth=1.5,
                markersize=4,
            )
        axis.set_title(f"{stencil}-point stencil")
        axis.set_xlabel("lattice spacing h")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("packet centroid-speed error")
    axes[0].legend(frameon=False)
    figure.suptitle("Three-dimensional Gaussian-packet convergence")
    figure.tight_layout()
    return _finish(figure, output)
