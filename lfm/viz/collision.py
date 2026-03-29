"""
Collision experiment visualization for LFM simulations.
=======================================================

Produces a 3-D animated movie of a particle collision showing the full
volumetric wave field, chi-well structure, and diagnostic panels.

The layout is similar to :func:`~lfm.viz.quantum.animate_double_slit_3d`
but tailored for head-on collisions:

* **Top 70 %** — 3-D scatter volume of ψ (diverging cmap ± wave field)
* **Bottom 30 %** — three panels:

  - Chi cross-section along collision axis (shows wells merging)
  - Energy density cross-section (shows annihilation + radiation)
  - Radial energy profile from grid centre

Usage::

    from lfm.viz.collision import animate_collision_3d
    animate_collision_3d(snapshots, save_path="collision.mp4")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

__all__ = ["animate_collision_3d"]


def _require_matplotlib() -> None:
    import importlib

    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(
            "matplotlib is required for collision visualization.  "
            "Install with: pip install matplotlib"
        )


def _save_mp4(anim: "FuncAnimation", path: str, fps: int) -> None:
    """Save animation — same pattern as the working double-slit code."""
    import matplotlib.animation

    # Try imageio-ffmpeg bundled binary first, then fall back to system
    try:
        import imageio_ffmpeg
        import matplotlib as mpl

        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    writer = matplotlib.animation.FFMpegWriter(fps=fps)
    anim.save(path, writer=writer)


def animate_collision_3d(
    snapshots: list[dict],
    *,
    collision_axis: int = 2,
    pos_a: tuple[int, ...] | None = None,
    pos_b: tuple[int, ...] | None = None,
    field: str = "energy_density",
    colormap: str = "inferno",
    fps: int = 20,
    intensity_floor: float = 0.005,
    max_points: int = 18000,
    camera_elev: float = 25.0,
    camera_azim: float = -55.0,
    camera_rotate: bool = False,
    camera_rotate_speed: float = 0.8,
    max_frames: int = 150,
    figsize: tuple[float, float] = (16, 10),
    title: str = "LFM Particle Collision",
    save_path: str | None = None,
) -> "FuncAnimation":
    """Create a 3-D animated movie of a particle collision.

    Parameters
    ----------
    snapshots : list of dict
        Output from the collision movie pass, each containing at least
        the *field* key as a 3-D array.
    collision_axis : int
        Axis along which the particles approach (0, 1, or 2). Default 2.
    pos_a, pos_b : tuple of int or None
        Initial positions of particles A and B. Used only for labels.
    field : str
        Snapshot field key (default ``"psi_real"``).
    colormap : str
        Diverging colormap for signed fields (default ``"RdBu_r"``).
    fps : int
        Output frames per second.
    intensity_floor : float
        Fraction of global peak under which points are invisible.
    max_points : int
        Maximum scatter points per frame.
    camera_elev, camera_azim : float
        Initial 3-D camera angles.
    camera_rotate : bool
        Slowly rotate camera during animation.
    camera_rotate_speed : float
        Degrees of azimuth rotation per frame.
    max_frames : int
        Maximum animation frames (sub-sampled from snapshots).
    figsize : tuple
        Figure size in inches.
    title : str
        Figure super-title.
    save_path : str or None
        File path to write (``.mp4`` recommended).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    _require_matplotlib()
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # ── Validate ───────────────────────────────────────────────────────
    if not snapshots:
        raise ValueError("snapshots list is empty")
    if field not in snapshots[0]:
        avail = list(snapshots[0].keys())
        raise ValueError(f"Field '{field}' not found.  Available: {avail}")
    first = snapshots[0][field]
    if first.ndim != 3:
        raise ValueError(f"Expected 3-D field, got shape {first.shape}")

    N = first.shape[0]
    mid = N // 2
    prop = collision_axis
    # Two transverse axes
    t1 = (prop + 1) % 3
    t2 = (prop + 2) % 3
    _is_signed = field.startswith("psi")

    # ── Sub-sample frames ──────────────────────────────────────────────
    if len(snapshots) > max_frames:
        # Quadratic spacing: dense early (approach + collision), sparser late
        t = np.linspace(0.0, 1.0, max_frames)
        raw_idx = np.round((t ** 2) * (len(snapshots) - 1)).astype(int)
        frame_idx = list(dict.fromkeys(raw_idx.tolist()))
    else:
        frame_idx = list(range(len(snapshots)))
    n_frames = len(frame_idx)

    # Global max for normalisation reference
    global_max = max(
        (float(np.abs(snapshots[i][field]).max()) for i in frame_idx),
        default=1.0,
    )
    if global_max == 0:
        global_max = 1.0

    cmap = plt.get_cmap(colormap)
    rng = np.random.default_rng(42)

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="#080810")
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[3, 1],
        hspace=0.22,
        wspace=0.28,
        left=0.04, right=0.97,
        top=0.91, bottom=0.05,
    )

    ax_3d = fig.add_subplot(gs[0, :], projection="3d", computed_zorder=False)
    ax_chi = fig.add_subplot(gs[1, 0])
    ax_ed = fig.add_subplot(gs[1, 1])
    ax_rad = fig.add_subplot(gs[1, 2])

    # ── Dark theme ─────────────────────────────────────────────────────
    for ax in (ax_chi, ax_ed, ax_rad):
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

    ax_3d.set_xlim(0, N)
    ax_3d.set_ylim(0, N)
    ax_3d.set_zlim(0, N)
    ax_3d.view_init(elev=camera_elev, azim=camera_azim)
    ax_3d.set_xlabel("Collision axis", color="#cc6644", fontsize=8, labelpad=4)
    ax_3d.set_ylabel("Transverse", color="#777", fontsize=8, labelpad=4)
    ax_3d.set_zlabel("Transverse", color="#777", fontsize=8, labelpad=4)

    # ── Display coordinate mapping ─────────────────────────────────────
    def _disp_coords(coords: np.ndarray):
        """Map (K, 3) array-space coords → display (X, Y, Z)."""
        return coords[:, prop], coords[:, t1], coords[:, t2]

    # ── Draw initial particle positions ────────────────────────────────
    if pos_a is not None:
        ax_3d.scatter(
            [pos_a[prop]], [pos_a[t1]], [pos_a[t2]],
            marker="x", s=120, c="#4488ff", linewidths=2, zorder=10,
        )
        ax_3d.text(pos_a[prop], pos_a[t1], pos_a[t2] + 3, "A",
                   color="#4488ff", fontsize=8, ha="center")
    if pos_b is not None:
        ax_3d.scatter(
            [pos_b[prop]], [pos_b[t1]], [pos_b[t2]],
            marker="x", s=120, c="#ff4444", linewidths=2, zorder=10,
        )
        ax_3d.text(pos_b[prop], pos_b[t1], pos_b[t2] + 3, "B",
                   color="#ff4444", fontsize=8, ha="center")

    # ── Initialise 2-D panels ──────────────────────────────────────────
    # Chi cross-section along collision axis through centre
    chi_line_data = np.full(N, 19.0)
    (line_chi,) = ax_chi.plot(
        range(N), chi_line_data, color="#3b82f6", lw=1.5,
    )
    ax_chi.set_xlim(0, N)
    ax_chi.set_ylim(0, 20)
    ax_chi.axhline(19, color="#555", ls="--", lw=0.6)
    ax_chi.set_title("χ along collision axis", fontsize=9, pad=4)
    ax_chi.set_xlabel("Position (cells)", fontsize=7)
    ax_chi.set_ylabel("χ", fontsize=7)

    # Energy density mid-plane slice
    ed_init = np.abs(first) if _is_signed else first
    slc = [slice(None)] * 3
    slc[t2] = mid
    ed_plane = ed_init[tuple(slc)]
    im_ed = ax_ed.imshow(
        ed_plane.T, origin="lower", cmap="inferno", aspect="auto",
        vmin=0, vmax=max(float(ed_plane.max()), 1e-10),
    )
    ax_ed.set_title("|Ψ|² cross-section", fontsize=9, pad=4)
    ax_ed.set_xlabel("Collision axis", fontsize=7)
    ax_ed.set_ylabel("Transverse", fontsize=7)

    # Radial energy profile from centre
    max_r = int(N * 0.45)
    (line_rad,) = ax_rad.plot(
        range(max_r), np.zeros(max_r), color="#22c55e", lw=1.4,
    )
    ax_rad.set_xlim(0, max_r)
    ax_rad.set_ylim(0, 1)
    ax_rad.set_title("Radial energy profile", fontsize=9, pad=4)
    ax_rad.set_xlabel("Radius from centre (cells)", fontsize=7)
    ax_rad.set_ylabel("Shell energy", fontsize=7)

    # ── Text overlays ──────────────────────────────────────────────────
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#eee")
    step_text = fig.text(0.5, 0.935, "Step 0", ha="center", fontsize=10, color="#aaa")
    phase_text = fig.text(0.97, 0.935, "", ha="right", fontsize=9, color="#ff8844")
    info_text = fig.text(0.04, 0.935, "", ha="left", fontsize=8, color="#555")

    # ── Scatter reference ──────────────────────────────────────────────
    _scatter = [None]

    # Pre-compute radial bin indices for vectorised profile
    _dist = np.sqrt(
        (np.arange(N, dtype=np.float32).reshape(N, 1, 1) - mid) ** 2
        + (np.arange(N, dtype=np.float32).reshape(1, N, 1) - mid) ** 2
        + (np.arange(N, dtype=np.float32).reshape(1, 1, N) - mid) ** 2
    )
    _dist_bins = np.clip(_dist.astype(np.intp), 0, max_r)

    # ── Update function ────────────────────────────────────────────────
    def _update(frame_num: int):
        snap_i = frame_idx[frame_num]
        snap = snapshots[snap_i]
        arr = snap[field]
        step = snap.get("step", snap_i)

        # Try to convert cupy → numpy
        try:
            import cupy
            if isinstance(arr, cupy.ndarray):
                arr = cupy.asnumpy(arr)
        except ImportError:
            pass

        # ── Remove old scatter ─────────────────────────────────────────
        if _scatter[0] is not None:
            _scatter[0].remove()
            _scatter[0] = None

        # ── Build scatter points ───────────────────────────────────────
        abs_arr = np.abs(arr) if _is_signed else arr
        fmax = float(abs_arr.max())
        if fmax < 1e-10:
            step_text.set_text(f"Step {step}")
            info_text.set_text(f"0 pts   frame {frame_num + 1}/{n_frames}")
            return []

        thresh = intensity_floor * fmax
        mask = abs_arr > thresh
        coords = np.argwhere(mask)
        vals = arr[mask]
        abs_vals = abs_arr[mask]
        n_cand = len(coords)

        if n_cand > max_points:
            probs = abs_vals.astype(np.float64)
            probs /= probs.sum()
            ch = rng.choice(n_cand, max_points, replace=False, p=probs)
            coords = coords[ch]
            vals = vals[ch]
            abs_vals = abs_vals[ch]

        # Colour mapping (match double-slit: clip before alpha)
        if _is_signed:
            sign_norm = np.clip(vals / fmax * 0.5 + 0.5, 0.0, 1.0)
            colors = cmap(sign_norm)
        else:
            nv = np.clip(abs_vals / fmax, 0.0, 1.0)
            nv = np.log1p(nv * 99.0) / np.log1p(99.0)
            colors = cmap(nv)

        # Alpha from intensity
        nv2 = np.clip(abs_vals / fmax, 0.0, 1.0)
        nv2 = np.log1p(nv2 * 99.0) / np.log1p(99.0)
        colors[:, 3] = 0.15 + 0.85 * nv2
        sizes = 4.0 + 40.0 * nv2

        # Final clip — float32→float64 arithmetic can exceed 1.0 by ~1e-10
        colors = np.clip(colors, 0.0, 1.0)

        dx, dy, dz = _disp_coords(coords)
        _scatter[0] = ax_3d.scatter(
            dx, dy, dz,
            c=colors, s=sizes, depthshade=True, linewidths=0,
        )

        # Camera rotation
        if camera_rotate:
            ax_3d.view_init(
                elev=camera_elev,
                azim=camera_azim + frame_num * camera_rotate_speed,
            )

        # ── Phase indicator ────────────────────────────────────────────
        # Estimate collision progress from chi cross-section
        chi_snap = snap.get("chi")
        if chi_snap is not None:
            try:
                import cupy as _cp
                if isinstance(chi_snap, _cp.ndarray):
                    chi_snap = _cp.asnumpy(chi_snap)
            except ImportError:
                pass

        frac = frame_num / max(1, n_frames - 1)
        if frac < 0.3:
            phase_text.set_text("\u25b6 Approach")
            phase_text.set_color("#4488ff")
        elif frac < 0.55:
            phase_text.set_text("\u26a1 Collision!")
            phase_text.set_color("#ff4444")
        else:
            phase_text.set_text("\u2728 Aftermath")
            phase_text.set_color("#22cc88")

        step_text.set_text(f"Step {step}")
        info_text.set_text(
            f"{len(coords):,} pts   frame {frame_num + 1}/{n_frames}"
        )

        # ── Chi cross-section panel ────────────────────────────────────
        if chi_snap is not None:
            chi_idx = [mid] * 3  # start at centre
            chi_line = np.empty(N, dtype=np.float64)
            for k in range(N):
                idx: list[Any] = [mid, mid, mid]
                idx[prop] = k
                chi_line[k] = float(chi_snap[tuple(idx)])
            line_chi.set_ydata(chi_line)
            chi_lo = max(0.0, float(chi_line.min()) - 1)
            chi_hi = min(25.0, float(chi_line.max()) + 1)
            ax_chi.set_ylim(chi_lo, chi_hi)

        # ── Energy density cross-section ───────────────────────────────
        ed_arr = abs_arr if _is_signed else arr
        # Re-fetch full field (we may have sub-sampled for scatter)
        full_abs = np.abs(snap[field]) if _is_signed else snap[field]
        try:
            import cupy as _cp2
            if isinstance(full_abs, _cp2.ndarray):
                full_abs = _cp2.asnumpy(full_abs)
        except ImportError:
            pass
        ed_slc = [slice(None)] * 3
        ed_slc[t2] = mid
        ed_plane = full_abs[tuple(ed_slc)]
        im_ed.set_data(ed_plane.T)
        im_ed.set_clim(0, max(float(ed_plane.max()), 1e-10))

        # ── Radial energy profile (vectorised) ─────────────────────────
        full_ed = full_abs ** 2 if _is_signed else full_abs
        bins = _dist_bins
        radial_e = np.bincount(
            bins.ravel(), weights=full_ed.ravel(), minlength=max_r + 1,
        )[:max_r].astype(np.float64)
        line_rad.set_ydata(radial_e)
        ax_rad.set_ylim(0, max(float(radial_e.max()), 1e-10) * 1.1)

        return []

    # ── Create animation ───────────────────────────────────────────────
    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=int(1000 / fps),
        blit=False,
    )

    if save_path:
        _save_mp4(anim, save_path, fps)

    return anim
