"""
Quantum-experiment visualization for LFM double-slit simulations.
==================================================================

Functions
---------
plot_interference_pattern   — 2-D heatmap + fringe profile side-by-side
animate_double_slit         — live-updating 2-D animation of the wave
render_3d_volume            — 3-D volume rendering (pyvista GPU or
                               matplotlib fallback)
volume_render_available     — True if pyvista is importable
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.figure import Figure

__all__ = [
    "plot_interference_pattern",
    "animate_double_slit",
    "render_3d_volume",
    "volume_render_available",
]


# ── Capability probe ───────────────────────────────────────────────────────


def volume_render_available() -> bool:
    """Return ``True`` if pyvista is installed (enables GPU volume rendering)."""
    try:
        import pyvista  # noqa: F401

        return True
    except ImportError:
        return False


# ── Interference pattern plots ────────────────────────────────────────────


def plot_interference_pattern(
    pattern: NDArray,
    *,
    title: str = "Interference Pattern",
    colormap: str = "inferno",
    log_scale: bool = False,
    show_profile: bool = True,
    profile_axis: int = 0,
    figsize: tuple[float, float] = (12, 5),
    save_path: str | None = None,
) -> "Figure":
    """Plot the time-integrated interference pattern from a :class:`~lfm.experiment.DetectorScreen`.

    Shows a 2-D intensity heatmap (left panel) and the marginal fringe
    profile summed along *profile_axis* (right panel).

    Parameters
    ----------
    pattern : ndarray, shape (N, N)
        Time-integrated energy density on the detector screen.
    title : str
        Overall figure title.
    colormap : str
        Matplotlib colormap name (default ``"inferno"``).
    log_scale : bool
        Use logarithmic colour scale for the heatmap.
    show_profile : bool
        If True, add the 1-D line profile panel.
    profile_axis : int
        Which axis to sum along for the 1-D line profile (0 or 1).
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if pattern.ndim != 2:
        raise ValueError(f"Expected 2-D pattern, got shape {pattern.shape}")

    ncols = 2 if show_profile else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    ax_heat = axes[0]
    data = pattern.astype(float)
    if log_scale:
        vmin = data[data > 0].min() if (data > 0).any() else 1e-10
        norm = LogNorm(vmin=vmin, vmax=data.max())
        im = ax_heat.imshow(data.T, origin="lower", cmap=colormap, norm=norm)
    else:
        im = ax_heat.imshow(data.T, origin="lower", cmap=colormap)

    fig.colorbar(im, ax=ax_heat, label="Intensity (a.u.)")
    ax_heat.set_title("Detector Screen")
    ax_heat.set_xlabel("x  (cells)")
    ax_heat.set_ylabel("y  (cells)")
    ax_heat.set_aspect("equal")

    if show_profile:
        ax_line = axes[1]
        profile = data.sum(axis=profile_axis)
        ax_line.plot(profile, np.arange(len(profile)), color="#f97316", lw=1.5)
        ax_line.set_title("Fringe Profile")
        ax_line.set_xlabel("Intensity")
        ax_line.set_ylabel("Position (cells)")
        ax_line.set_ylim(0, len(profile) - 1)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Double-slit animation ─────────────────────────────────────────────────


def animate_double_slit(
    snapshots: list[dict],
    *,
    barrier_axis: int = 2,
    barrier_position: int | None = None,
    detector_position: int | None = None,
    field: str = "energy_density",
    slice_axis: int = 0,
    colormap: str = "inferno",
    fps: int = 12,
    figsize: tuple[float, float] = (14, 5),
    save_path: str | None = None,
) -> "FuncAnimation":
    """Animate a double-slit run as three panels: slice view, pattern build-up, 1-D profile.

    Left panel
        A 2-D slice through the centre of the grid showing the wave
        propagating toward and through the barrier.  The barrier and
        detector positions are marked with dashed lines.

    Middle panel
        The *accumulated* interference pattern on the detector screen
        up to the current frame.

    Right panel
        The 1-D fringe profile (marginal of the accumulated pattern),
        updated each frame.

    Parameters
    ----------
    snapshots : list of dict
        Output of :meth:`~lfm.Simulation.run_with_snapshots`.  Each dict
        must contain the *field* key (default ``"energy_density"``).
    barrier_axis : int
        Propagation axis (same as ``Barrier.axis``).
    barrier_position : int or None
        Barrier location along *barrier_axis* for annotation.
    detector_position : int or None
        Detector screen position for annotation.
    field : str
        Which field key to read from each snapshot.
    slice_axis : int
        The axis to *fix* when extracting the 2-D slice (0 = take yz-plane
        at x=N//2).
    colormap : str
        Matplotlib colormap name.
    fps : int
        Frames per second.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save an animated gif/mp4 to this path.

    Returns
    -------
    anim : matplotlib FuncAnimation
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not snapshots:
        raise ValueError("snapshots list is empty")
    if field not in snapshots[0]:
        raise ValueError(
            f"Field '{field}' not found in snapshots.  "
            f"Available: {list(snapshots[0].keys())}"
        )

    N = snapshots[0][field].shape[0]
    mid = N // 2

    def _slice(arr: NDArray) -> NDArray:
        """Extract 2-D mid-plane slice, NOT along barrier_axis."""
        idx = [slice(None), slice(None), slice(None)]
        idx[slice_axis] = mid
        return arr[tuple(idx)]

    def _detector_slice(arr: NDArray) -> NDArray:
        """Extract the detector plane from a 3-D field."""
        if detector_position is None:
            pos = int(N * 0.80)
        else:
            pos = detector_position
        idx = [slice(None), slice(None), slice(None)]
        idx[barrier_axis] = pos
        return arr[tuple(idx)]

    # Accumulate interference pattern
    accumulated = np.zeros((N, N), dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_wave, ax_pat, ax_prof = axes

    # --- Wave panel ---
    first_slice = _slice(snapshots[0][field])
    im_wave = ax_wave.imshow(
        first_slice.T, origin="lower", cmap=colormap, animated=True
    )
    fig.colorbar(im_wave, ax=ax_wave, fraction=0.046, pad=0.04)
    ax_wave.set_title("Wave propagation")
    ax_wave.set_xlabel("Transverse (cells)")
    ax_wave.set_ylabel("Propagation (cells)")
    # Annotate barrier line
    if barrier_position is not None:
        _axes_other = [ax for ax in [0, 1, 2] if ax != slice_axis]
        if barrier_axis in _axes_other:
            ax_wave.axvline(barrier_position, color="cyan", lw=1, ls="--", label="barrier")
    if detector_position is not None:
        _axes_other = [ax for ax in [0, 1, 2] if ax != slice_axis]
        if barrier_axis in _axes_other:
            ax_wave.axvline(detector_position, color="lime", lw=1, ls="--", label="screen")
    ax_wave.legend(fontsize=7, loc="upper left")

    # --- Pattern panel ---
    first_det = _detector_slice(snapshots[0][field])
    accumulated += first_det.astype(np.float64)
    im_pat = ax_pat.imshow(
        accumulated.T, origin="lower", cmap=colormap, animated=True
    )
    fig.colorbar(im_pat, ax=ax_pat, fraction=0.046, pad=0.04)
    ax_pat.set_title("Accumulated pattern")
    ax_pat.set_xlabel("x  (cells)")
    ax_pat.set_ylabel("y  (cells)")

    # --- Profile panel ---
    profile = accumulated.sum(axis=0)
    (line_prof,) = ax_prof.plot(profile, np.arange(len(profile)), "#f97316", lw=1.5)
    ax_prof.set_title("Fringe profile")
    ax_prof.set_xlabel("Intensity")
    ax_prof.set_ylabel("Position (cells)")
    ax_prof.set_ylim(0, N - 1)
    ax_prof.set_xlim(0, profile.max() * 1.5 + 1)

    fig.suptitle("LFM Double-Slit Experiment", fontsize=13, fontweight="bold")
    fig.tight_layout()

    step_text = ax_wave.set_title("Wave  (step 0)")

    def _update(frame_idx: int):
        snap = snapshots[frame_idx]
        arr = snap[field]

        # Update wave slice
        sl = _slice(arr)
        im_wave.set_data(sl.T)
        vmax = float(sl.max()) or 1.0
        im_wave.set_clim(0, vmax)
        step_text.set_text(f"Wave  (step {snap.get('step', frame_idx)})")

        # Accumulate detector pattern
        nonlocal accumulated
        accumulated += _detector_slice(arr).astype(np.float64)
        im_pat.set_data(accumulated.T)
        pat_vmax = float(accumulated.max()) or 1.0
        im_pat.set_clim(0, pat_vmax)

        # Update profile
        prof = accumulated.sum(axis=0)
        line_prof.set_xdata(prof)
        ax_prof.set_xlim(0, prof.max() * 1.1 + 1)

        return im_wave, im_pat, line_prof

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(snapshots),
        interval=int(1000 / fps),
        blit=True,
    )

    if save_path:
        _save_animation(anim, save_path, fps)

    return anim


# ── 3-D volume rendering ──────────────────────────────────────────────────


def render_3d_volume(
    field: NDArray,
    *,
    threshold: float | None = None,
    backend: str = "auto",
    colormap: str = "inferno",
    opacity: float | str = "sigmoid",
    iso_alpha: float = 0.6,
    title: str | None = None,
    figsize: tuple[float, float] = (9, 8),
    save_path: str | None = None,
) -> "Figure | None":
    """3-D volume rendering of a scalar field.

    Automatically uses **pyvista** (GPU-accelerated) when available and
    falls back to a matplotlib voxel plot otherwise.

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        3-D scalar field to render (e.g. ``sim.energy_density`` or
        ``sim.chi``).
    threshold : float or None
        If given, also draw an isosurface at this value.  Defaults to
        ``field.mean() + 2*field.std()``.
    backend : str
        ``"auto"`` — pyvista if available, else matplotlib;
        ``"pyvista"`` — force pyvista (raises ImportError if missing);
        ``"matplotlib"`` — always use matplotlib.
    colormap : str
        Colormap name.
    opacity : float or str
        Pyvista opacity transfer function (e.g. ``"sigmoid"``) or a
        global alpha float for the matplotlib voxel plot.
    iso_alpha : float
        Isosurface transparency for matplotlib path (0 = transparent).
    title : str or None
        Window/figure title.
    figsize : tuple
        Matplotlib figure size (ignored in pyvista path).
    save_path : str or None
        Save rendered image to this path.

    Returns
    -------
    fig : matplotlib Figure or None
        Returns a Figure in the matplotlib path.  In the pyvista path
        the interactive window is shown; returns ``None``.
    """
    if field.ndim != 3:
        raise ValueError(f"Expected 3-D field, got shape {field.shape}")

    use_pyvista = False
    if backend == "pyvista":
        use_pyvista = True
    elif backend == "auto":
        use_pyvista = volume_render_available()
    elif backend != "matplotlib":
        raise ValueError(
            f"Unknown backend '{backend}'.  Choose 'auto', 'pyvista', or 'matplotlib'."
        )

    if use_pyvista:
        return _render_pyvista(
            field,
            threshold=threshold,
            colormap=colormap,
            opacity=opacity,
            title=title,
            save_path=save_path,
        )
    else:
        return _render_matplotlib(
            field,
            threshold=threshold,
            colormap=colormap,
            iso_alpha=float(opacity) if isinstance(opacity, (int, float)) else 0.4,
            title=title,
            figsize=figsize,
            save_path=save_path,
        )


def _render_pyvista(
    field: NDArray,
    threshold: float | None,
    colormap: str,
    opacity,
    title: str | None,
    save_path: str | None,
) -> None:
    """GPU volume rendering via pyvista (VTK + OpenGL)."""
    import pyvista as pv  # type: ignore[import]

    N = field.shape[0]
    grid = pv.ImageData()
    grid.dimensions = (N + 1, N + 1, N + 1)
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    # cell data
    grid.cell_data["field"] = field.ravel(order="F").astype(float)

    plotter = pv.Plotter(title=title or "LFM Field Volume")
    plotter.add_volume(grid, scalars="field", cmap=colormap, opacity=opacity)

    if threshold is not None:
        surf = grid.contour([threshold], scalars="field")
        if surf.n_points > 0:
            plotter.add_mesh(surf, color="white", opacity=0.5, label="isosurface")

    if save_path:
        plotter.show(auto_close=False)
        plotter.screenshot(save_path)
        plotter.close()
    else:
        plotter.show()

    return None


def _render_matplotlib(
    field: NDArray,
    threshold: float | None,
    colormap: str,
    iso_alpha: float,
    title: str | None,
    figsize: tuple[float, float],
    save_path: str | None,
) -> "Figure":
    """CPU voxel rendering via matplotlib (automatic downsampling)."""
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    N = field.shape[0]
    step = max(1, N // 48)  # downsample to ≤ 48³ for performance

    small = field[::step, ::step, ::step].astype(float)
    if threshold is None:
        thr = float(small.mean() + 2.0 * small.std())
    else:
        thr = float(threshold)

    mask = small >= thr
    norm = Normalize(vmin=float(small.min()), vmax=float(small.max()))
    cmap = plt.get_cmap(colormap)
    colors = cmap(norm(small))
    colors[..., 3] = iso_alpha  # set alpha uniformly

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if mask.any():
        ax.voxels(mask, facecolors=colors, edgecolor="none", linewidth=0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or "LFM Field Volume")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Helpers ────────────────────────────────────────────────────────────────


def _save_animation(anim, path: str, fps: int) -> None:
    """Save animation to GIF or MP4 depending on extension."""
    import matplotlib

    suffix = path.split(".")[-1].lower()
    if suffix == "gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
        anim.save(path, writer=writer)
    else:
        writer = matplotlib.animation.FFMpegWriter(fps=fps)
        anim.save(path, writer=writer)
