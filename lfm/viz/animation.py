"""
Animations
==========

Create animated visualisations from field snapshot sequences produced by
:meth:`lfm.Simulation.run_with_snapshots`.

Requires *matplotlib* and, for saving:

* GIF export — `Pillow`  (``pip install Pillow``)
* MP4 / WebM export — `FFmpeg` on ``PATH``

Examples
--------
>>> snaps = sim.run_with_snapshots(5000, snapshot_every=100)
>>> anim = lfm.animate_slice(snaps, save_path="chi_evo.gif")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation


def animate_slice(
    snapshots: list[dict],
    field: str = "chi",
    axis: int = 2,
    index: int | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    interval_ms: int = 100,
    title: str | None = None,
    save_path: str | None = None,
    fps: int = 10,
) -> FuncAnimation:
    """Animate a central 2-D slice through a 3-D field over time.

    Parameters
    ----------
    snapshots : list of dict
        Sequence returned by :meth:`lfm.Simulation.run_with_snapshots`.
        Each dict must contain the *field* key.
    field : str
        Which field to animate: ``"chi"``, ``"psi_real"``, or
        ``"energy_density"``.
    axis : int
        Axis to slice along (0 = x, 1 = y, 2 = z).
    index : int or None
        Slice index.  Defaults to the midpoint.
    vmin, vmax : float or None
        Colour scale limits.  If None the global min/max across all frames
        is used, giving a consistent colour scale.
    cmap : str
        Matplotlib colourmap name.
    interval_ms : int
        Delay between frames in milliseconds.
    title : str or None
        Figure title.  ``"{step}"`` is replaced with the current step.
    save_path : str or None
        If given, save to this file (GIF or MP4 depending on extension).
    fps : int
        Frames per second when saving.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation  # type: ignore[import-untyped]

    if not snapshots:
        raise ValueError("snapshots list is empty")
    if field not in snapshots[0]:
        available = list(snapshots[0].keys())
        raise KeyError(
            f"Field '{field}' not in snapshots.  Available: {available}. "
            "Pass fields=['{field}'] to run_with_snapshots()."
        )

    # Extract 2-D slices from each snapshot
    slices = []
    for snap in snapshots:
        arr = snap[field]
        idx = index if index is not None else arr.shape[axis] // 2
        sl = np.take(arr, idx, axis=axis)
        slices.append(sl)

    # Consistent colour limits across all frames
    if vmin is None:
        vmin = float(min(s.min() for s in slices))
    if vmax is None:
        vmax = float(max(s.max() for s in slices))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        slices[0].T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axis_labels = ["x", "y", "z"]
    removed = axis_labels[axis]
    remaining = [lbl for lbl in axis_labels if lbl != removed]
    ax.set_xlabel(remaining[0])
    ax.set_ylabel(remaining[1])

    _base_title = title if title is not None else f"{field} (slice {axis}={index})"
    title_obj = ax.set_title(_base_title)

    def _update(i: int) -> tuple:
        im.set_data(slices[i].T)
        step = snapshots[i].get("step", i)
        title_obj.set_text(
            _base_title.replace("{step}", str(step))
            if "{step}" in _base_title
            else f"{_base_title}  step={step}"
        )
        return im, title_obj

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(slices),
        interval=interval_ms,
        blit=True,
    )

    if save_path is not None:
        _save_animation(anim, save_path, fps=fps)

    return anim


def animate_three_slices(
    snapshots: list[dict],
    field: str = "chi",
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    interval_ms: int = 100,
    save_path: str | None = None,
    fps: int = 10,
) -> FuncAnimation:
    """Animate three orthogonal mid-plane slices (xy, xz, yz) side-by-side.

    Parameters
    ----------
    snapshots : list of dict
        Sequence from :meth:`lfm.Simulation.run_with_snapshots`.
    field : str
        Field key to visualise.
    vmin, vmax : float or None
        Global colour scale limits (consistent across all panels and frames).
    cmap : str
        Matplotlib colourmap.
    interval_ms : int
        Delay between frames in milliseconds.
    save_path : str or None
        Optional save path (GIF or MP4).
    fps : int
        Frames per second when saving.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation  # type: ignore[import-untyped]

    if not snapshots:
        raise ValueError("snapshots list is empty")
    if field not in snapshots[0]:
        available = list(snapshots[0].keys())
        raise KeyError(f"Field '{field}' not in snapshots.  Available: {available}.")

    def _extract(snap: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = snap[field]
        N = arr.shape[0]
        xy = arr[:, :, N // 2]
        xz = arr[:, N // 2, :]
        yz = arr[N // 2, :, :]
        return xy, xz, yz

    all_frames = [_extract(s) for s in snapshots]

    if vmin is None:
        vmin = min(min(a.min(), b.min(), c.min()) for a, b, c in all_frames)
    if vmax is None:
        vmax = max(max(a.max(), b.max(), c.max()) for a, b, c in all_frames)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["xy  (z-mid)", "xz  (y-mid)", "yz  (x-mid)"]
    ims = []
    for ax, data, ttl in zip(axes, all_frames[0], titles, strict=False):
        im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
        ax.set_title(ttl)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)
    step_text = fig.suptitle(f"{field}  step=0")

    def _update(i: int) -> tuple:
        for im, data in zip(ims, all_frames[i], strict=False):
            im.set_data(data.T)
        step = snapshots[i].get("step", i)
        step_text.set_text(f"{field}  step={step}")
        return (*ims, step_text)

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(all_frames),
        interval=interval_ms,
        blit=True,
    )

    if save_path is not None:
        _save_animation(anim, save_path, fps=fps)

    return anim


def _save_animation(anim: FuncAnimation, path: str, fps: int = 10) -> None:
    """Save animation to GIF or MP4 depending on file extension."""
    from pathlib import Path as _Path

    ext = _Path(path).suffix.lower()
    if ext == ".gif":
        try:
            anim.save(path, writer="pillow", fps=fps)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Could not save GIF to '{path}'.  Install Pillow: pip install Pillow"
            ) from exc
    elif ext in (".mp4", ".webm", ".avi"):
        try:
            anim.save(path, writer="ffmpeg", fps=fps)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Could not save video to '{path}'.  Ensure FFmpeg is installed and on PATH."
            ) from exc
    else:
        anim.save(path, fps=fps)
