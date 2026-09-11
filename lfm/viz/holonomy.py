"""Fixed-camera 3D snapshot GIFs for lattice evolution diagnostics.

The word "holonomy" here means the recorded path through simulation states for
visual inspection. It is not a claim of gauge, Berry, topological, or physical
holonomy. The output is an external lattice visualization only.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from lfm.viz._util import _require_matplotlib


@dataclass(frozen=True)
class LatticeHolonomySnapshot:
    """One saved state for a fixed-camera lattice GIF."""

    step: int
    physical_time: float
    dx: float
    energy: np.ndarray
    chi: np.ndarray
    label: str = ""


def _physical_coordinates(n: int, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = (np.arange(n, dtype=np.float64) - 0.5 * float(n)) * float(dx)
    return np.meshgrid(axis, axis, axis, indexing="ij")


def _select_top_points(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    max_points: int,
    floor_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    positive = arr > 0.0
    if not finite.any() or not positive.any():
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty
    peak = float(np.nanmax(arr[finite]))
    mask = finite & positive & (arr >= floor_fraction * max(peak, 1.0e-30))
    indices = np.flatnonzero(mask.ravel())
    if indices.size > max_points:
        ranked = np.argsort(arr.ravel()[indices])[-max_points:]
        indices = indices[ranked]
    return (
        x.ravel()[indices],
        y.ravel()[indices],
        z.ravel()[indices],
        arr.ravel()[indices],
    )


def _draw_box(ax: Any, limit: float) -> None:
    half = float(limit)
    corners = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float64,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color="0.45",
            linewidth=0.6,
            alpha=0.45,
        )


def _render_frame(
    snapshot: LatticeHolonomySnapshot,
    *,
    title: str,
    chi0: float,
    camera_elev: float,
    camera_azim: float,
    max_energy_points: int,
    max_chi_points: int,
) -> np.ndarray:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    energy = np.asarray(snapshot.energy, dtype=np.float64)
    chi = np.asarray(snapshot.chi, dtype=np.float64)
    n = int(energy.shape[0])
    dx = float(snapshot.dx)
    x, y, z = _physical_coordinates(n, dx)
    dep = np.maximum(float(chi0) - chi, 0.0)
    neg = np.maximum(-chi, 0.0)

    energy_x, energy_y, energy_z, energy_v = _select_top_points(
        energy,
        x,
        y,
        z,
        max_points=max_energy_points,
        floor_fraction=0.18,
    )
    chi_x, chi_y, chi_z, chi_v = _select_top_points(
        dep,
        x,
        y,
        z,
        max_points=max_chi_points,
        floor_fraction=0.34,
    )
    neg_x, neg_y, neg_z, neg_v = _select_top_points(
        neg,
        x,
        y,
        z,
        max_points=max_chi_points,
        floor_fraction=0.05,
    )

    fig = plt.figure(figsize=(5.8, 4.9))
    ax = fig.add_subplot(111, projection="3d")
    limit = 0.5 * n * dx
    _draw_box(ax, limit)

    if chi_v.size:
        sizes = 8.0 + 38.0 * chi_v / max(float(np.nanmax(chi_v)), 1.0e-30)
        ax.scatter(
            chi_x,
            chi_y,
            chi_z,
            c=chi_v,
            cmap="Blues",
            s=sizes,
            alpha=0.22,
            depthshade=False,
            linewidths=0.0,
        )
    if neg_v.size:
        sizes = 20.0 + 90.0 * neg_v / max(float(np.nanmax(neg_v)), 1.0e-30)
        ax.scatter(
            neg_x,
            neg_y,
            neg_z,
            c=neg_v,
            cmap="cool",
            s=sizes,
            alpha=0.82,
            depthshade=False,
            linewidths=0.0,
            marker="x",
        )
    if energy_v.size:
        sizes = 9.0 + 60.0 * energy_v / max(float(np.nanmax(energy_v)), 1.0e-30)
        ax.scatter(
            energy_x,
            energy_y,
            energy_z,
            c=energy_v,
            cmap="inferno",
            s=sizes,
            alpha=0.84,
            depthshade=False,
            linewidths=0.0,
        )

    ax.view_init(elev=camera_elev, azim=camera_azim)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    label_suffix = " | " + snapshot.label if snapshot.label else ""
    ax.set_title(
        f"{title}\nstep {snapshot.step}  t={snapshot.physical_time:.2f}{label_suffix}",
        fontsize=10,
    )
    ax.text2D(
        0.02,
        0.02,
        "orange: E energy   blue: chi depression   x: chi below zero",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=90, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return np.asarray(PILImage.open(buffer).convert("RGB"))


def render_lattice_holonomy_gif(
    snapshots: list[LatticeHolonomySnapshot],
    output_path: str | Path,
    *,
    title: str,
    chi0: float,
    fps: int = 4,
    camera_elev: float = 23.0,
    camera_azim: float = 42.0,
    max_energy_points: int = 1000,
    max_chi_points: int = 1500,
) -> dict[str, Any]:
    """Render selected live states into a compact fixed-camera 3D GIF."""

    if not snapshots:
        return {
            "gif_written": False,
            "gif_path": "",
            "gif_frames": 0,
            "reason": "no snapshots",
        }
    _require_matplotlib()
    from PIL import Image as PILImage

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        _render_frame(
            snapshot,
            title=title,
            chi0=chi0,
            camera_elev=camera_elev,
            camera_azim=camera_azim,
            max_energy_points=max_energy_points,
            max_chi_points=max_chi_points,
        )
        for snapshot in snapshots
    ]
    images = [PILImage.fromarray(frame) for frame in frames]
    duration_ms = int(round(1000.0 / max(int(fps), 1)))
    images[0].save(
        target,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return {
        "gif_written": True,
        "gif_path": str(target.resolve()),
        "gif_frames": len(images),
        "gif_fps": int(fps),
        "camera_elev": float(camera_elev),
        "camera_azim": float(camera_azim),
        "visual_claim_boundary": "external fixed-camera lattice visualization only",
    }
