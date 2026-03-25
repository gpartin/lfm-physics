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
    "animate_double_slit_3d",
    "animate_3d_slices",
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
            f"Field '{field}' not found in snapshots.  Available: {list(snapshots[0].keys())}"
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
    im_wave = ax_wave.imshow(first_slice.T, origin="lower", cmap=colormap, animated=True)
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
    im_pat = ax_pat.imshow(accumulated.T, origin="lower", cmap=colormap, animated=True)
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


# ── 3-D double-slit movie (world-class) ────────────────────────────────────


def animate_double_slit_3d(
    snapshots: list[dict],
    *,
    barrier_axis: int = 2,
    slit_axis: int | None = None,
    barrier_position: int | None = None,
    detector_position: int | None = None,
    source_position: int | None = None,
    slit_centers: list[int] | None = None,
    slit_width: int = 4,
    field: str = "energy_density",
    colormap: str = "inferno",
    fps: int = 15,
    intensity_floor: float = 0.005,
    max_points: int = 15000,
    camera_elev: float = 20.0,
    camera_azim: float = -65.0,
    camera_rotate: bool = False,
    camera_rotate_speed: float = 0.5,
    barrier_thickness: int = 2,
    max_frames: int = 120,
    figsize: tuple[float, float] = (16, 10),
    title: str = "LFM Double-Slit Experiment",
    save_path: str | None = None,
) -> "FuncAnimation":
    """Create a 3-D animated movie of the double-slit experiment.

    The movie shows the full 3-D wave field as a volumetric point cloud
    with physical geometry overlays (barrier wall with slit openings,
    detector screen, source plane).  A dark background with hot-colormap
    wave rendering creates a dramatic, publication-quality result.

    Layout
    ------
    * **Top 70 %** — 3-D perspective view (Axes3D) with:

      - Wave field as colour/alpha-mapped scatter points
      - Barrier wall (semi-transparent blue) with slit holes (green edges)
      - Detector screen (semi-transparent green plane)
      - Source position (blue dashed cross-hair)
      - Phase indicator (approaching / passing through / building pattern)

    * **Bottom 30 %** — three panels:

      - Propagation cross-section (slit-axis vs propagation mid-plane)
      - Accumulated detector pattern (2-D heatmap on detector face)
      - 1-D fringe profile (intensity vs slit position)

    Parameters
    ----------
    snapshots : list of dict
        Output of :meth:`~lfm.Simulation.run_with_snapshots`.
    barrier_axis : int
        Propagation axis (0, 1, or 2).  Default 2 (z).
    slit_axis : int or None
        Axis along which slit centres vary.  Default ``(barrier_axis+1)%3``.
    barrier_position, detector_position, source_position : int or None
        Positions along the propagation axis.  Default N//2, 3N//4, N//4.
    slit_centers : list of int or None
        Centre positions of each slit along *slit_axis*.
        Default ``[N//2 - N//8, N//2 + N//8]``.
    slit_width : int
        Width of each slit in cells.
    field : str
        Snapshot field key (default ``"energy_density"``).
    colormap : str
        Matplotlib colormap for the wave field.
    fps : int
        Output frames per second.
    intensity_floor : float
        Fraction of global max below which cells are invisible (0–1).
    max_points : int
        Maximum scatter points per frame (intensity-weighted sampling).
    camera_elev, camera_azim : float
        3-D camera elevation and azimuth (degrees).
    camera_rotate : bool
        Slowly rotate the camera azimuth during animation (default
        ``False``).  Best left off so the viewer can study the layout.
    camera_rotate_speed : float
        Azimuth degrees per frame when *camera_rotate* is enabled.
    barrier_thickness : int
        Depth of the barrier slab in cells.  Renders as a solid 3-D
        block instead of a flat plane (default 2).
    max_frames : int
        Cap on animation frames (evenly sub-sampled from snapshots).
    figsize : tuple
        Figure size in inches.
    title : str
        Figure super-title.
    save_path : str or None
        Save movie to this path (``.mp4`` recommended; ``.gif`` also works).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # ── Validate inputs ────────────────────────────────────────────────
    if not snapshots:
        raise ValueError("snapshots list is empty")
    if field not in snapshots[0]:
        raise ValueError(
            f"Field '{field}' not in snapshots. Available: {list(snapshots[0].keys())}"
        )
    first = snapshots[0][field]
    if first.ndim != 3:
        raise ValueError(f"Expected 3-D field, got shape {first.shape}")

    N = first.shape[0]

    # ── Geometry defaults ──────────────────────────────────────────────
    prop = barrier_axis
    s_ax = slit_axis if slit_axis is not None else (prop + 1) % 3
    o_ax = 3 - prop - s_ax  # the "other" transverse axis

    bpos = barrier_position if barrier_position is not None else N // 2
    dpos = detector_position if detector_position is not None else (N * 3) // 4
    spos = source_position if source_position is not None else N // 4
    sctrs = (
        list(sorted(slit_centers))
        if slit_centers is not None
        else [
            N // 2 - N // 8,
            N // 2 + N // 8,
        ]
    )
    sw = slit_width
    mid = N // 2

    # ── Display coordinate mapping ─────────────────────────────────────
    # Matplotlib 3-D always puts Z vertical.  We want:
    #   display X (horizontal, left→right) = propagation axis
    #   display Z (vertical)               = slit axis
    #   display Y (depth, into screen)     = other transverse axis
    #
    # All geometry and scatter coordinates are built in *array* space
    # (prop, s_ax, o_ax) then mapped to display (X, Y, Z) via _disp().

    def _disp_pt(arr_pt):
        """Map a single array-space [3] point to display (X, Y, Z)."""
        return arr_pt[prop], arr_pt[o_ax], arr_pt[s_ax]

    def _disp_coords(coords):
        """Map (N, 3) array-space coords to display columns (X, Y, Z)."""
        return coords[:, prop], coords[:, o_ax], coords[:, s_ax]

    # ── Sub-sample frames (front-loaded for wavefront capture) ─────────
    if len(snapshots) > max_frames:
        # Quadratic spacing: dense at the start to capture the wavefront
        # propagating across the grid, sparser at steady-state.
        t = np.linspace(0.0, 1.0, max_frames)
        raw_idx = np.round((t**2) * (len(snapshots) - 1)).astype(int)
        frame_idx = list(dict.fromkeys(raw_idx.tolist()))
    else:
        frame_idx = list(range(len(snapshots)))
    n_frames = len(frame_idx)

    # ── Source-exclusion zone ──────────────────────────────────────────
    # The continuous source accumulates energy that dwarfs the propagating
    # wave.  Exclude a thin slab around the source plane from both the
    # scatter and the normalisation so the wavefront drives the colour map.
    _src_lo = max(0, spos - 2)
    _src_hi = min(N - 1, spos + 2)
    _src_mask = np.ones((N,) * 3, dtype=bool)
    _idx_exc = [slice(None)] * 3
    _idx_exc[prop] = slice(_src_lo, _src_hi + 1)
    _src_mask[tuple(_idx_exc)] = False

    # Reference max (detects completely empty frames)
    global_max = max((float(snapshots[i][field].max()) for i in frame_idx), default=1.0)
    if global_max == 0:
        global_max = 1.0

    # Signed-field detection: psi_real / psi_imag have +/- values
    _is_signed = field.startswith("psi")
    cmap = plt.get_cmap(colormap)
    rng = np.random.default_rng(42)

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="#080810")
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[3, 1],
        hspace=0.20,
        wspace=0.25,
        left=0.04,
        right=0.97,
        top=0.91,
        bottom=0.05,
    )

    ax_3d = fig.add_subplot(gs[0, :], projection="3d", computed_zorder=False)
    ax_prop = fig.add_subplot(gs[1, 0])
    ax_det = fig.add_subplot(gs[1, 1])
    ax_prof = fig.add_subplot(gs[1, 2])

    # ── Dark theme ─────────────────────────────────────────────────────
    for ax in (ax_prop, ax_det, ax_prof):
        ax.set_facecolor("#0c0c18")
        ax.tick_params(colors="#888", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.title.set_color("#ccc")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")

    ax_3d.set_facecolor("#080810")
    for pane in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("#1a1a2e")
        pane._axinfo["grid"]["color"] = "#1a1a2e"
    ax_3d.tick_params(colors="#555", labelsize=6, pad=0)

    # ── 3-D axes config ────────────────────────────────────────────────
    ax_3d.set_xlim(0, N)
    ax_3d.set_ylim(0, N)
    ax_3d.set_zlim(0, N)
    ax_3d.view_init(elev=camera_elev, azim=camera_azim)

    # Display axes: X=propagation, Y=depth, Z=slit (vertical)
    ax_3d.set_xlabel(
        "Source → Detector",
        color="#6699cc",
        fontsize=8,
        labelpad=4,
    )
    ax_3d.set_ylabel(
        "Depth",
        color="#777",
        fontsize=8,
        labelpad=4,
    )
    ax_3d.set_zlabel(
        "Slit axis",
        color="#777",
        fontsize=8,
        labelpad=4,
    )

    # ── Draw static geometry ───────────────────────────────────────────
    # Barrier solid sections
    slit_ranges = [(sc - sw // 2, sc - sw // 2 + sw) for sc in sctrs]
    solid_ranges: list[tuple[int, int]] = []
    prev = 0
    for lo, hi in slit_ranges:
        if lo > prev:
            solid_ranges.append((prev, lo))
        prev = hi
    if prev < N:
        solid_ranges.append((prev, N))

    _bt = max(1, barrier_thickness)
    for s_lo, s_hi in solid_ranges:
        # Build a solid 3-D block for each barrier section.
        # Array-space corners, then map to display.
        corners_arr = []
        for pv in (bpos, bpos + _bt):
            for sv in (s_lo, s_hi):
                for ov in (0, N):
                    pt = np.zeros(3)
                    pt[prop] = pv
                    pt[s_ax] = sv
                    pt[o_ax] = ov
                    corners_arr.append(_disp_pt(pt))
        # 8 corners: index by (p, s, o) bits → 0..7
        c = np.array(corners_arr)  # (8, 3) display coords
        box_faces = [
            [c[0], c[1], c[3], c[2]],  # s=lo face
            [c[4], c[5], c[7], c[6]],  # s=hi face
            [c[0], c[1], c[5], c[4]],  # o=0 face
            [c[2], c[3], c[7], c[6]],  # o=N face
            [c[0], c[2], c[6], c[4]],  # p=bpos face
            [c[1], c[3], c[7], c[5]],  # p=bpos+bt face
        ]
        poly = Poly3DCollection(
            box_faces,
            alpha=0.22,
            facecolor="#2244aa",
            edgecolor="#4466cc",
            linewidth=0.6,
        )
        ax_3d.add_collection3d(poly)

    # Slit edges (bright green) — 3-D rectangles around slit openings
    for sc in sctrs:
        for edge in (sc - sw // 2, sc - sw // 2 + sw):
            pts = []
            for pv in (bpos, bpos + _bt):
                for ov in (0, N):
                    pt = np.zeros(3)
                    pt[prop] = pv
                    pt[s_ax] = edge
                    pt[o_ax] = ov
                    pts.append(np.array(_disp_pt(pt)))
            # Draw slit edge lines along all 4 edges of the rectangle
            for a, b in [(0, 1), (2, 3), (0, 2), (1, 3)]:
                ax_3d.plot(
                    [pts[a][0], pts[b][0]],
                    [pts[a][1], pts[b][1]],
                    [pts[a][2], pts[b][2]],
                    color="#22ff66",
                    lw=1.8,
                    alpha=0.7,
                )

    # Detector screen (semi-transparent green)
    det_arr = []
    for a, b in [(0, 0), (N, 0), (N, N), (0, N)]:
        pt = np.zeros(3)
        pt[prop] = dpos
        pt[s_ax] = a
        pt[o_ax] = b
        det_arr.append(np.array(_disp_pt(pt)))
    ax_3d.add_collection3d(
        Poly3DCollection(
            [det_arr],
            alpha=0.08,
            facecolor="#33cc55",
            edgecolor="#33cc55",
            linewidth=0.3,
        )
    )

    # 3-D detector fringe scatter — projects accumulated pattern onto
    # the detector plane so fringes are visible IN the 3D scene.
    _det_scatter = [None]  # mutable ref for _update
    _det_grid_s, _det_grid_o = np.meshgrid(
        np.arange(N, dtype=np.float32),
        np.arange(N, dtype=np.float32),
        indexing="ij",
    )  # (N, N) grid coords on slit_ax × other_ax
    _det_cmap = plt.get_cmap("hot")

    # Source cross-hair (blue dashed)
    for cross_ax in (s_ax, o_ax):
        p1, p2 = np.zeros(3), np.zeros(3)
        p1[prop] = spos
        p2[prop] = spos
        p1[cross_ax] = 0
        p2[cross_ax] = N
        other_c = o_ax if cross_ax == s_ax else s_ax
        p1[other_c] = mid
        p2[other_c] = mid
        d1, d2 = _disp_pt(p1), _disp_pt(p2)
        ax_3d.plot(
            [d1[0], d2[0]],
            [d1[1], d2[1]],
            [d1[2], d2[2]],
            color="#4488ff",
            lw=0.8,
            ls="--",
            alpha=0.35,
        )

    # 3-D text labels at key positions
    lbl_arr = np.zeros(3)
    lbl_arr[prop] = bpos
    lbl_arr[s_ax] = N + 1
    lbl_arr[o_ax] = mid
    lx, ly, lz = _disp_pt(lbl_arr)
    ax_3d.text(lx, ly, lz, "barrier", color="#5588cc", fontsize=7, ha="left")
    lbl2_arr = np.zeros(3)
    lbl2_arr[prop] = dpos
    lbl2_arr[s_ax] = N + 1
    lbl2_arr[o_ax] = mid
    l2x, l2y, l2z = _disp_pt(lbl2_arr)
    ax_3d.text(l2x, l2y, l2z, "screen", color="#33cc55", fontsize=7, ha="left")
    # Source label
    lbl3_arr = np.zeros(3)
    lbl3_arr[prop] = spos
    lbl3_arr[s_ax] = N + 1
    lbl3_arr[o_ax] = mid
    l3x, l3y, l3z = _disp_pt(lbl3_arr)
    ax_3d.text(l3x, l3y, l3z, "source", color="#4488ff", fontsize=7, ha="left")

    # ── Initialise 2-D panels ──────────────────────────────────────────
    # Propagation slice: fix other_ax at mid → (slit_ax, prop_ax) plane
    def _prop_slice(arr: NDArray) -> NDArray:
        idx = [slice(None)] * 3
        idx[o_ax] = mid
        sl = arr[tuple(idx)]
        # Remaining dims: axes in ascending order excluding o_ax
        remaining = [i for i in range(3) if i != o_ax]
        # We want imshow vertical=prop, horizontal=slit
        # imshow raw: vertical=dim0, horizontal=dim1
        slit_dim = remaining.index(s_ax)
        if slit_dim == 0:
            return sl.T  # slit is dim0 → transpose to put prop as dim0 (vertical)
        return sl

    def _detector_face(arr: NDArray) -> NDArray:
        idx = [slice(None)] * 3
        idx[prop] = min(dpos, N - 1)
        return arr[tuple(idx)]

    # Accumulated pattern
    accumulated = np.zeros((N, N), dtype=np.float64)

    ps0 = _prop_slice(first)
    _ps0_lim = max(float(np.abs(ps0).max()), 1e-10)
    im_prop = ax_prop.imshow(
        ps0,
        origin="lower",
        cmap=colormap,
        vmin=-_ps0_lim if _is_signed else 0,
        vmax=_ps0_lim,
        aspect="auto",
    )
    ax_prop.set_title("Propagation cross-section", fontsize=9, pad=4)
    ax_prop.set_xlabel("Slit axis (cells)", fontsize=7)
    ax_prop.set_ylabel("Propagation (cells)", fontsize=7)
    ax_prop.axhline(bpos, color="#5588cc", lw=0.8, ls="--", alpha=0.5)
    ax_prop.axhline(dpos, color="#33cc55", lw=0.8, ls="--", alpha=0.5)
    ax_prop.axhline(spos, color="#4488ff", lw=0.6, ls=":", alpha=0.3)

    df0 = _detector_face(first).astype(np.float64)
    # Accumulate energy (ψ²) for signed fields — a detector measures hits,
    # not field amplitude.  For unsigned fields (energy_density), accumulate
    # as-is since it is already positive-definite.
    accumulated += df0**2 if _is_signed else df0
    _det0_lim = max(float(accumulated.max()), 1e-10)
    im_det = ax_det.imshow(
        accumulated.T,
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=_det0_lim,
        aspect="equal",
    )
    ax_det.set_title("Detector (accumulated)", fontsize=9, pad=4)
    ax_det.set_xlabel("Slit axis (cells)", fontsize=7)
    ax_det.set_ylabel("Transverse (cells)", fontsize=7)

    # Fringe profile: sum detector face along the non-slit transverse dim
    remaining_det = [i for i in range(3) if i != prop]
    slit_dim_det = remaining_det.index(s_ax)
    prof_sum_axis = 1 - slit_dim_det  # sum the other dimension

    profile0 = accumulated.sum(axis=prof_sum_axis)
    (line_prof,) = ax_prof.plot(
        profile0,
        np.arange(len(profile0)),
        "#ff8844",
        lw=1.3,
    )
    ax_prof.set_facecolor("#0c0c18")
    ax_prof.set_title("Fringe profile", fontsize=9, pad=4)
    ax_prof.set_xlabel("Intensity", fontsize=7)
    ax_prof.set_ylabel("Slit axis (cells)", fontsize=7)
    ax_prof.set_ylim(0, N - 1)
    ax_prof.set_xlim(0, max(1.0, profile0.max()) * 1.3)

    # ── Text overlays ──────────────────────────────────────────────────
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#eee")
    step_text = fig.text(
        0.5,
        0.935,
        "Step 0",
        ha="center",
        fontsize=10,
        color="#aaa",
    )
    phase_text = fig.text(
        0.97,
        0.935,
        "",
        ha="right",
        fontsize=9,
        color="#66aaff",
    )

    # Points / frame counter (lower-left)
    info_text = fig.text(0.04, 0.935, "", ha="left", fontsize=8, color="#555")

    # ── Scatter reference ──────────────────────────────────────────────
    _scatter = [None]

    # ── Update function ────────────────────────────────────────────────
    def _update(frame_num: int):
        snap_i = frame_idx[frame_num]
        snap = snapshots[snap_i]
        arr = snap[field]
        step = snap.get("step", snap_i)

        # -- Remove old scatter --
        if _scatter[0] is not None:
            _scatter[0].remove()
            _scatter[0] = None

        # -- Two-zone normalisation ────────────────────────────────────
        # Pre-barrier and post-barrier regions differ in energy by ~1000x.
        # Normalise each zone independently so the emerging wavefront
        # through the slits is as visible as the incoming wave.
        visible = arr * _src_mask.astype(arr.dtype)

        # Build zone masks: pre (source-excl to barrier) and post (barrier+)
        _pre_m = np.zeros_like(_src_mask)
        _post_m = np.zeros_like(_src_mask)
        _pre_idx = [slice(None)] * 3
        _pre_idx[prop] = slice(_src_hi + 1, bpos)
        _pre_m[tuple(_pre_idx)] = True
        _post_idx = [slice(None)] * 3
        _post_idx[prop] = slice(bpos + _bt, None)
        _post_m[tuple(_post_idx)] = True

        all_coords = []
        all_nv = []
        all_sign_norm = []  # for diverging cmap on signed fields
        for zone_mask in (_pre_m, _post_m):
            zone = visible * zone_mask.astype(arr.dtype)
            abs_zone = np.abs(zone) if _is_signed else zone
            zmax = float(abs_zone.max())
            if zmax < 1e-10:
                continue
            zt = intensity_floor * zmax
            zm = abs_zone > zt
            zcoords = np.argwhere(zm)
            zvals = zone[zm]  # keep sign for color
            abs_zvals = np.abs(zvals) if _is_signed else zvals
            n_z = len(zcoords)
            half_budget = max_points // 2
            if n_z > half_budget:
                zp = abs_zvals.astype(np.float64)
                zp /= zp.sum()
                ch = rng.choice(n_z, half_budget, replace=False, p=zp)
                zcoords = zcoords[ch]
                zvals = zvals[ch]
                abs_zvals = np.abs(zvals) if _is_signed else zvals
            # Per-zone log normalisation (magnitude)
            zraw = np.clip(abs_zvals / zmax, 0.0, 1.0)
            znv = np.log1p(zraw * 99.0) / np.log1p(99.0)
            znv = np.clip(znv, 0.0, 1.0)
            all_coords.append(zcoords)
            all_nv.append(znv)
            if _is_signed:
                # Map signed value to [0,1]: -max→0, 0→0.5, +max→1
                sn = np.clip(zvals / zmax * 0.5 + 0.5, 0.0, 1.0)
                all_sign_norm.append(sn)

        if all_coords:
            coords = np.concatenate(all_coords)
            nv = np.concatenate(all_nv)
            n_pts = len(coords)

            if _is_signed:
                sign_norm = np.concatenate(all_sign_norm)
                colors = cmap(sign_norm)  # diverging: neg→blue, pos→red
            else:
                colors = cmap(nv)
            colors = np.clip(colors, 0.0, 1.0)
            colors[:, 3] = 0.15 + 0.85 * nv
            sizes = 6.0 + 50.0 * nv

            dx, dy, dz = _disp_coords(coords)
            _scatter[0] = ax_3d.scatter(
                dx,
                dy,
                dz,
                c=colors,
                s=sizes,
                depthshade=True,
                linewidths=0,
            )
        else:
            n_pts = 0
            # No meaningful energy visible — update 2-D panels but skip scatter
            ps = _prop_slice(arr)
            im_prop.set_data(ps)
            if _is_signed:
                vlim = max(float(np.abs(ps).max()), 1e-10)
                im_prop.set_clim(-vlim, vlim)
            else:
                im_prop.set_clim(0, max(float(ps.max()), 1e-10))
            step_text.set_text(f"Step {step}")
            info_text.set_text(f"0 pts   frame {frame_num + 1}/{n_frames}")
            return []

        # -- Phase indicator --
        energy_per_layer = np.zeros(N, dtype=np.float64)
        for k in range(N):
            idx = [slice(None)] * 3
            idx[prop] = k
            layer = arr[tuple(idx)]
            energy_per_layer[k] = float(np.abs(layer).sum() if _is_signed else layer.sum())
        emax = energy_per_layer.max()
        if emax > 0:
            sig = energy_per_layer > 0.01 * emax
            wavefront = int(np.where(sig)[0][-1]) if sig.any() else 0
        else:
            wavefront = 0

        if wavefront < bpos:
            phase_text.set_text("\u25b6 Approaching barrier")
            phase_text.set_color("#66aaff")
        elif wavefront < dpos:
            phase_text.set_text("\u25b6 Passing through slits")
            phase_text.set_color("#ffaa44")
        else:
            phase_text.set_text("\u25b6 Building interference pattern")
            phase_text.set_color("#44ff88")

        # Camera rotation for true 3-D depth perception
        if camera_rotate:
            new_azim = camera_azim + frame_num * camera_rotate_speed
            ax_3d.view_init(elev=camera_elev, azim=new_azim)

        step_text.set_text(f"Step {step}")
        info_text.set_text(f"{n_pts:,} pts   frame {frame_num + 1}/{n_frames}")

        # -- Propagation slice --
        ps = _prop_slice(arr)
        im_prop.set_data(ps)
        if _is_signed:
            vlim = max(float(np.abs(ps).max()), 1e-10)
            im_prop.set_clim(-vlim, vlim)
        else:
            im_prop.set_clim(0, max(float(ps.max()), 1e-10))

        # -- Detector face --
        nonlocal accumulated
        det_face = _detector_face(arr).astype(np.float64)
        accumulated += det_face**2 if _is_signed else det_face
        im_det.set_data(accumulated.T)
        im_det.set_clim(0, max(float(accumulated.max()), 1e-10))

        # -- Fringe profile --
        prof = accumulated.sum(axis=prof_sum_axis)
        line_prof.set_xdata(prof)
        ax_prof.set_xlim(0, max(1.0, prof.max()) * 1.2)

        # -- 3-D detector fringe scatter --------------------------------
        # Project accumulated intensity onto detector plane in the 3D scene.
        if _det_scatter[0] is not None:
            _det_scatter[0].remove()
            _det_scatter[0] = None
        abs_acc = np.abs(accumulated) if _is_signed else accumulated
        dmax = float(abs_acc.max())
        if dmax > 1e-10:
            dn = abs_acc / dmax  # normalised [0,1]
            dt = 0.05  # only show pixels above 5% of max
            dm = dn > dt
            if dm.any():
                ds_vals = _det_grid_s[dm]
                do_vals = _det_grid_o[dm]
                dn_vals = dn[dm]
                # Build 3D coords on the detector plane (array space)
                d_coords = np.zeros((len(ds_vals), 3), dtype=np.float32)
                d_coords[:, prop] = float(dpos)
                d_coords[:, s_ax] = ds_vals
                d_coords[:, o_ax] = do_vals
                # Display coords via same mapping as wave scatter
                dx, dy, dz = _disp_coords(d_coords)
                dcols = _det_cmap(dn_vals)
                dcols[:, 3] = 0.3 + 0.7 * dn_vals  # alpha: brighter = more opaque
                _det_scatter[0] = ax_3d.scatter(
                    dx,
                    dy,
                    dz,
                    c=dcols,
                    s=12 + 30 * dn_vals,
                    depthshade=False,
                    linewidths=0,
                    zorder=5,
                )

        return []

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=int(1000 / fps),
        blit=False,
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


# ── 3-D orthogonal-slice animation ──────────────────────────────────────────


def animate_3d_slices(
    snapshots: list[dict],
    *,
    field: str = "energy_density",
    colormap: str = "inferno",
    fps: int = 12,
    figsize: tuple[float, float] = (13, 4.5),
    save_path: str | None = None,
) -> "FuncAnimation":
    """Animate three orthogonal mid-plane slices of a 3-D scalar field.

    Displays three panels side-by-side:

    * **XY slice** — ``z = N//2`` (constant *z* plane)
    * **XZ slice** — ``y = N//2`` (constant *y* plane)
    * **YZ slice** — ``x = N//2`` (constant *x* plane)

    Each panel is updated every frame so the viewer can follow the full
    3-D wave evolution without any pyvista dependency.

    This function pairs naturally with :func:`~lfm.io.save_snapshots` /
    :func:`~lfm.io.load_snapshots` for offline replay::

        from lfm.io import load_snapshots
        from lfm.viz import animate_3d_slices

        snaps = load_snapshots("wave_run.npz")
        anim  = animate_3d_slices(snaps, fps=15)
        anim.save("wave_slices.gif", writer="pillow", fps=15)

    Parameters
    ----------
    snapshots : list of dict
        Output of :meth:`~lfm.Simulation.run_with_snapshots` or
        :func:`~lfm.io.load_snapshots`.  Each dict must contain *field*.
    field : str
        Field key to visualise (default ``"energy_density"``).
    colormap : str
        Matplotlib colormap name.
    fps : int
        Animation frame rate.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save to this path (GIF or MP4 by extension).

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
            f"Field '{field}' not found in snapshots.  Available: {list(snapshots[0].keys())}"
        )

    first = snapshots[0][field]
    if first.ndim != 3:
        raise ValueError(
            f"animate_3d_slices requires 3-D fields, got shape {first.shape}. "
            "Use animate_double_slit for 2-D fields."
        )

    N = first.shape[0]
    mid = N // 2

    titles = ["XY plane  (z = N//2)", "XZ plane  (y = N//2)", "YZ plane  (x = N//2)"]

    def _slices(arr: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        return arr[:, :, mid], arr[:, mid, :], arr[mid, :, :]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("LFM 3-D Field Evolution", fontsize=13, fontweight="bold")

    s0, s1, s2 = _slices(first)
    slices_init = [s0, s1, s2]

    images = []
    for ax, sl, ttl in zip(axes, slices_init, titles):
        vmax = float(sl.max()) or 1.0
        im = ax.imshow(sl.T, origin="lower", cmap=colormap, vmin=0.0, vmax=vmax, animated=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("cells")
        ax.set_ylabel("cells")
        ax.set_aspect("equal")
        images.append(im)

    step_label = fig.text(
        0.5,
        0.01,
        f"step {snapshots[0].get('step', 0)}",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    def _update(frame_idx: int):
        snap = snapshots[frame_idx]
        arr = snap[field]
        s0_, s1_, s2_ = _slices(arr)
        for im, sl in zip(images, [s0_, s1_, s2_]):
            im.set_data(sl.T)
            vmax = float(sl.max()) or 1.0
            im.set_clim(0, vmax)
        step_label.set_text(f"step {snap.get('step', frame_idx)}")
        return (*images, step_label)

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
