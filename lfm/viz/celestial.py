"""3-D animated movie for LFM celestial body simulations.

``animate_celestial_3d`` runs the simulation forward step-by-step,
captures a compact chi-well point cloud per frame, pre-renders each
frame with matplotlib Axes3D, and saves the result as MP4 or GIF.

Memory strategy
---------------
Full 3-D chi arrays are freed immediately after extracting the voxel
point cloud (max_points sparse samples).  Stored per-frame data is
only ~ 100 kB/frame regardless of grid size.

Example
-------
::

    import lfm
    from lfm import solar_system, place_bodies
    from lfm.viz.celestial import animate_celestial_3d

    cfg = lfm.SimulationConfig(grid_size=128, field_level=lfm.FieldLevel.REAL,
                                boundary_type=lfm.BoundaryType.FROZEN)
    sim = lfm.Simulation(cfg)
    bodies      = solar_system()
    body_omegas = place_bodies(sim, bodies)
    animate_celestial_3d(sim, bodies, body_omegas,
                         n_frames=60, save_path="solar_system_3d.mp4")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Circle

from lfm.constants import CHI0

if TYPE_CHECKING:
    from lfm.scenarios.celestial import CelestialBody


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _render_frame_3d(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    depths: np.ndarray,
    bodies: list[CelestialBody],
    body_positions: dict[str, tuple[float, float, float]],
    t: float,
    step: int,
    chi_min: float,
    N: int,
    cx: float,
    cy: float,
    cz: float,
    dchi_vmax: float,
    frame_idx: int,
    title: str,
    camera_elev: float,
    camera_azim_start: float,
    camera_rotate_speed: float,
    figsize: tuple[float, float],
    dpi: int,
    cmap,
    show_chi: bool = True,
    chi_reveal_frame: int = 0,
    n_frames: int = 1,
) -> np.ndarray:
    """Pre-render one frame to an RGB numpy array."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, facecolor="#050510", dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#050510")

    # Transparent panes, dim grid
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#1a1a2e")
        axis._axinfo["grid"]["color"] = "#151520"
    ax.tick_params(colors="#333", labelsize=5, pad=-2)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_zlim(0, N)
    ax.set_xlabel("x", color="#444", fontsize=6, labelpad=1)
    ax.set_ylabel("y", color="#444", fontsize=6, labelpad=1)
    ax.set_zlabel("z", color="#444", fontsize=6, labelpad=1)

    az = camera_azim_start + frame_idx * camera_rotate_speed
    ax.view_init(elev=camera_elev, azim=az)

    # ── Chi-well point cloud (hidden during orbit 1) ────────────────────────
    # depthshade=False avoids matplotlib's 3-D depth re-sort which causes
    # body markers to blink as the camera rotates.  Depth is encoded via alpha.
    fade = min(1.0, (frame_idx - chi_reveal_frame + 1) / 5.0) if show_chi else 0.0
    if show_chi and len(xs) > 0:
        norm_d = np.clip(depths / max(dchi_vmax, 1e-6), 0.0, 1.0)
        colors = cmap(norm_d * 0.85 + 0.1)
        colors[:, 3] = np.clip(norm_d * 0.6 + 0.05, 0.0, 1.0) * fade
        ax.scatter(
            xs,
            ys,
            zs,
            c=colors,
            s=2,
            linewidths=0,
            depthshade=False,
            rasterized=True,
            zorder=2,
        )

    # ── Body markers drawn as 2-D overlay ────────────────────────────────────
    # matplotlib Axes3D depth-sorts ALL artists by centroid depth; when the chi
    # cloud's centroid is closer to the camera the Sun gets occluded and blinks.
    # Fix: call canvas.draw() first so the 3-D projection matrix is finalised,
    # then project body positions to figure-fraction coords and paint them as
    # 2-D figure-level artists — these are ALWAYS on top, no depth-fight.
    fig.canvas.draw()  # lock in the 3-D projection
    _draw_body_overlays(fig, ax, bodies, body_positions, N, dpi, figsize)

    # ── Info overlay ─────────────────────────────────────────────────────────
    fig.text(0.03, 0.97, title, color="white", fontsize=9, va="top", fontweight="bold")
    orbit_label = (
        "orbit 1  (χ hidden)"
        if not show_chi
        else (
            "χ revealed — orbit 2+  (dark-matter halo visible)"
            if frame_idx == chi_reveal_frame
            else "orbit 2+  (χ active)"
        )
    )
    fig.text(
        0.03,
        0.92,
        f"t = {t:.1f}   step {step:,}\nχ_min = {chi_min:+.4f}   χ₀ = {int(CHI0)}\n{orbit_label}",
        color="#9999bb",
        fontsize=7,
        va="top",
        fontfamily="monospace",
    )

    # ── Rasterise to RGB ──────────────────────────────────────────────────────
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    rgb = arr[:, :, :3].copy()
    plt.close(fig)
    return rgb


# ──────────────────────────────────────────────────────────────────────────────
# 2-D body overlay helper (called AFTER 3-D canvas.draw)
# ──────────────────────────────────────────────────────────────────────────────


def _draw_body_overlays(
    fig,
    ax,
    bodies,
    body_positions: dict[str, tuple[float, float, float]],
    N: int,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    """Project each body to 2-D figure coordinates and draw as figure-level artists.

    Body positions are the χ-field minima extracted by GOV-02 — no analytical
    orbit formula.  This completely sidesteps matplotlib's 3-D depth-sort; the
    markers are painted directly onto the figure canvas in normalised figure
    coordinates and are guaranteed on top of everything in the 3-D axes.
    """
    from mpl_toolkits.mplot3d import proj3d

    proj_mat = ax.get_proj()  # 4×4 homogeneous projection matrix (valid now)
    fw_px = fig.get_figwidth() * dpi
    fh_px = fig.get_figheight() * dpi

    # Transparent 2-D overlay axes that covers the whole figure
    ax2d = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax2d.set_xlim(0.0, 1.0)
    ax2d.set_ylim(0.0, 1.0)
    ax2d.set_axis_off()

    for b in bodies:
        bx, by, bz = body_positions[b.name]

        # Project 3-D world coords → normalised device coords (NDC, ~[-1,1]²)
        x_ndc, y_ndc, _ = proj3d.proj_transform(bx, by, bz, proj_mat)

        # NDC → display pixels via the Axes3D data transform
        disp = ax.transData.transform((x_ndc, y_ndc))

        # Display pixels → figure fraction
        xf = float(disp[0]) / fw_px
        yf = float(disp[1]) / fh_px

        # Marker radius in figure-fraction units — keep it small (dot, not disc)
        r_fill = max(b.dot_size**0.5 * 0.0006, 0.003)
        r_glow = r_fill * 1.8

        # Glow halo
        ax2d.add_patch(
            Circle(
                (xf, yf),
                r_glow,
                transform=ax2d.transAxes,
                color=b.ring_color,
                alpha=0.15,
                linewidth=0,
                zorder=90,
            )
        )
        # Filled body
        ax2d.add_patch(
            Circle(
                (xf, yf),
                r_fill,
                transform=ax2d.transAxes,
                facecolor=b.color,
                edgecolor=b.ring_color,
                linewidth=1.2,
                zorder=91,
            )
        )
        # Label — just above the marker
        ax2d.text(
            xf,
            yf + r_fill * 1.8,
            b.name,
            transform=ax2d.transAxes,
            color=b.ring_color,
            fontsize=5.5,
            ha="center",
            va="bottom",
            zorder=92,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def animate_celestial_3d(
    sim,
    bodies: list[CelestialBody],
    body_omegas: dict[str, float],
    *,
    n_frames: int = 80,
    steps_per_frame: int = 150,
    center: tuple[float, float, float] | None = None,
    fps: int = 10,
    well_threshold: float = 0.25,
    max_points: int = 8000,
    camera_elev: float = 25.0,
    camera_azim_start: float = -30.0,
    camera_rotate_speed: float = 0.0,
    title: str = "LFM Celestial Bodies",
    save_path: str | None = None,
    figsize: tuple[float, float] = (12.8, 7.2),
    dpi: int = 100,
    chi_reveal_frame: int | None = None,
    verbose: bool = True,
) -> str | None:
    """Create a rotating 3-D animated movie of a celestial-body LFM simulation.

    The function advances the simulation forward, captures sparse chi-well
    point clouds per frame, pre-renders each frame with matplotlib Axes3D,
    and saves the result as MP4 (via FFMpeg / imageio) or GIF (fallback).

    Parameters
    ----------
    sim
        A fully equilibrated ``lfm.Simulation`` instance, ready to run.
    bodies : list of CelestialBody
        Bodies already placed in *sim* by ``place_bodies()``.
    body_omegas : dict[str, float]
        Dict returned by ``place_bodies()``: ``{body.name: omega}`` where
        omega is angular velocity in rad / (lattice time unit).
    n_frames : int
        Number of rendered frames.
    steps_per_frame : int
        GOV-01/02 evolution steps between consecutive frames.
    center : (cx, cy, cz) or None
        Grid centre coordinates.  Defaults to (N//2, N//2, N//2).
    fps : int
        Output frame rate.
    well_threshold : float
        Show voxels where χ < χ₀ – well_threshold.
    max_points : int
        Maximum scatter points per frame (downsampled if needed).
    camera_elev : float
        3-D camera elevation angle in degrees.
    camera_azim_start : float
        Starting azimuth angle in degrees.
    camera_rotate_speed : float
        Azimuth rotation per rendered frame, in degrees.
    title : str
        Figure super-title.
    save_path : str or None
        Output path.  ``.mp4`` → tries FFMpeg/imageio; ``.gif`` → tries
        imageio/Pillow.  If None, a `<title>.mp4`` path is derived.
    figsize : (width, height)
        Figure size in inches.
    dpi : int
        Render resolution.
    chi_reveal_frame : int or None
        Frame index at which the chi-well cloud becomes visible.  Before
        this frame only body markers are shown ("orbit 1 bare"); from this
        frame onward the carved chi halo fades in over 5 frames.
        If None (default), auto-calculated as the frame when the outermost
        orbiter completes its first full orbit, so chi appears **exactly**
        as orbit 2 begins.  Pass 0 to show chi from the start.
    verbose : bool
        Print per-frame progress.

    Returns
    -------
    saved_path : str or None
        Path of the saved file, or None if saving failed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = sim.chi.shape[0]
    if center is None:
        cx = cy = cz = float(N // 2)
    else:
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    dt = sim.config.dt
    rng = np.random.default_rng(42)

    # ── Auto chi_reveal_frame: show chi after FASTEST planet's first orbit ──
    # Using the fastest (innermost) orbiter gives the most visible "orbit 1 →
    # orbit 2" comparison.  You see Mercury (or the innermost body) complete
    # one bare orbit, chi fades in revealing the carved dark-matter halo, then
    # orbit 2 is visibly tighter/faster — same mechanism as flat rotation curves.
    if chi_reveal_frame is None:
        omegas = [abs(body_omegas[b.name]) for b in bodies if b.orbital_radius > 0]
        if omegas:
            omega_max = max(omegas)  # fastest body (innermost)
            T_fastest = (2 * np.pi / omega_max) if omega_max > 0 else 0.0
            steps_one_orbit = T_fastest / dt
            chi_reveal_frame = int(np.ceil(steps_one_orbit / steps_per_frame))
            chi_reveal_frame = min(chi_reveal_frame, n_frames // 3)  # chi visible for >=2/3 of anim
            chi_reveal_frame = min(chi_reveal_frame, n_frames - 5)  # need room for orbit 2
            if verbose:
                fastest_name = max(
                    (b for b in bodies if b.orbital_radius > 0),
                    key=lambda b: abs(body_omegas[b.name]),
                ).name
                print(
                    f"χ reveal at frame {chi_reveal_frame}/{n_frames} "
                    f"(after {fastest_name}'s first orbit: "
                    f"{steps_one_orbit:.0f} steps = {T_fastest:.1f} t)"
                )
        else:
            chi_reveal_frame = 0

    # ── Step 1: evolve + collect compact point clouds ─────────────────────────
    if verbose:
        print(f"Evolving {n_frames} frames × {steps_per_frame} steps…")

    # Initialise search centers from t=0 placement positions (initial conditions only).
    # After each frame the center is updated to the chi-field-derived position.
    body_search_centers: dict[str, tuple[int, int, int]] = {}
    for b in bodies:
        if b.orbital_radius <= 0:
            body_search_centers[b.name] = (int(cx), int(cy), int(cz))
        else:
            body_search_centers[b.name] = (
                int(cx + b.orbital_radius * np.cos(b.orbital_phase)),
                int(cy + b.orbital_radius * np.sin(b.orbital_phase)),
                int(cz),
            )

    frame_clouds: list[tuple] = []
    for f in range(n_frames):
        sim.run(steps_per_frame, record_metrics=False)
        step = (f + 1) * steps_per_frame
        t = step * dt

        chi3 = np.array(sim.chi, dtype=np.float32)  # GPU→CPU; freed below
        chi_min = float(chi3.min())

        # χ volume: every voxel where χ < χ₀ − well_threshold gets a dot.
        # Alpha is proportional to depth so χ = 19 is invisible (transparent
        # vacuum) and deep wells are bright.
        mask = chi3 < (CHI0 - well_threshold)
        if mask.any():
            ix, iy, iz = np.where(mask)
            deps = (CHI0 - chi3[mask]).astype(np.float32)
            if len(ix) > max_points:
                sel = rng.choice(len(ix), max_points, replace=False)  # uniform
                ix, iy, iz, deps = ix[sel], iy[sel], iz[sel], deps[sel]
        else:
            ix = iy = iz = np.empty(0, dtype=np.intp)
            deps = np.empty(0, dtype=np.float32)

        # Find each body's position via |Ψ|² (energy density) maximum in a
        # tight search box around the previous-frame position.
        # energy_density = Ψ_real² + Ψ_imag²; works for both REAL and COMPLEX
        # solitons.  For the phase-stabilised complex case |Ψ|² is constant
        # (no false zeros when psi_real crosses zero in its internal oscillation).
        # For orbiting bodies, blank out the central body's region first so its
        # large |Ψ|² peak doesn't win over a nearby planet's smaller peak.
        psi2 = np.array(sim.energy_density, dtype=np.float32)

        # Build a version of psi2 with the central body masked out (for planets).
        # Use a SPHERICAL mask (not a box) so the Sun's field is fully excluded
        # out to the same radius in all directions including diagonals.
        central_body = next((b for b in bodies if b.orbital_radius <= 0), None)
        psi2_nomask = psi2
        if central_body is not None:
            cx_s, cy_s, cz_s = body_search_centers[central_body.name]
            # Use 4σ sphere to ensure the Sun Gaussian tail is fully zeroed.
            # A box of side 2*rex would leak at corners (r=rex*√3 > rex).
            rex = int(central_body.sigma * 4) + 1
            xs0, xs1 = max(0, cx_s - rex), min(N, cx_s + rex + 1)
            ys0, ys1 = max(0, cy_s - rex), min(N, cy_s + rex + 1)
            zs0, zs1 = max(0, cz_s - rex), min(N, cz_s + rex + 1)
            psi2_planet = psi2.copy()
            # Spherical mask: zero all voxels within rex cells of the Sun center
            xi = np.arange(xs0, xs1) - cx_s
            yi = np.arange(ys0, ys1) - cy_s
            zi = np.arange(zs0, zs1) - cz_s
            xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
            sphere_mask = (xx**2 + yy**2 + zz**2) <= rex**2
            psi2_planet[xs0:xs1, ys0:ys1, zs0:zs1][sphere_mask] = 0.0
        else:
            psi2_planet = psi2

        # Low-pass filter the entire psi2 image once per frame.
        # Moving solitons are encoded as carrier waves (k ≈ χ₀·v/c ≈ 2.3 rad/cell
        # for Mercury).  The instantaneous psi_real² oscillates at twice the
        # carrier frequency: the peak is zero wherever cos(k·Δr)=0, which can
        # place argmax at the sidelobe rather than the soliton centre.
        # Blurring with σ ≈ soliton-sigma damps the carrier
        # (exp(-k²·σ²/2) ≈ 0.002 for k=2.3, σ=1.5) while keeping the envelope
        # peak at the soliton centre (the blurred envelope is just wider).
        from scipy.ndimage import gaussian_filter as _gf  # noqa: PLC0415

        min_planet_sigma = min((b.sigma for b in bodies if b.orbital_radius > 0), default=1.5)
        blur_sigma = max(min_planet_sigma, 1.0)
        psi2_smooth = _gf(psi2_nomask, sigma=blur_sigma)
        psi2_planet_smooth = _gf(psi2_planet, sigma=blur_sigma)

        body_field_pos: dict[str, tuple[float, float, float]] = {}
        for b in bodies:
            ex, ey, ez = body_search_centers[b.name]
            field = psi2_smooth if b.orbital_radius <= 0 else psi2_planet_smooth
            if b.orbital_radius <= 0:
                sr = max(int(b.sigma * 5), 8)
            else:
                move = abs(body_omegas[b.name]) * b.orbital_radius * steps_per_frame * dt
                sr = max(int(move * 1.5 + b.sigma * 3), 5)
            x0, x1 = max(0, ex - sr), min(N, ex + sr + 1)
            y0, y1 = max(0, ey - sr), min(N, ey + sr + 1)
            z0, z1 = max(0, ez - sr), min(N, ez + sr + 1)
            region = field[x0:x1, y0:y1, z0:z1]
            if region.size > 0 and region.max() > 0:
                idx = np.unravel_index(np.argmax(region), region.shape)
                fp = (float(x0 + idx[0]), float(y0 + idx[1]), float(z0 + idx[2]))
            else:
                fp = (float(ex), float(ey), float(ez))
            body_field_pos[b.name] = fp
            body_search_centers[b.name] = (int(fp[0]), int(fp[1]), int(fp[2]))

        del psi2, psi2_planet, psi2_smooth, psi2_planet_smooth

        frame_clouds.append(
            (
                ix.astype(np.float32),
                iy.astype(np.float32),
                iz.astype(np.float32),
                deps.astype(np.float32),
                step,
                t,
                chi_min,
                body_field_pos,
            )
        )
        del chi3  # free the big array immediately

        if verbose:
            print(
                f"\r  {f + 1:3d}/{n_frames}  step={step:,}  t={t:.1f}  "
                f"χ_min={chi_min:.3f}  pts={len(ix)}",
                end="",
                flush=True,
            )
    if verbose:
        print()

    # Global depth colour scales
    all_depths = np.concatenate([fc[3] for fc in frame_clouds if len(fc[3]) > 0])
    dchi_vmax = float(np.percentile(all_depths, 99)) if len(all_depths) > 0 else 5.0
    # plasma: dark purple (shallow) → yellow (deep well).  χ = 19 is transparent.
    cmap = plt.get_cmap("plasma")

    # ── Step 2: pre-render frames ─────────────────────────────────────────────
    if verbose:
        print(f"Rendering {n_frames} frames at {figsize[0] * dpi:.0f}×{figsize[1] * dpi:.0f} px…")

    rendered: list[np.ndarray] = []
    for f_idx, (xs, ys, zs, deps, step, t, chi_min, body_field_pos) in enumerate(frame_clouds):
        show_chi = f_idx >= chi_reveal_frame
        rgb = _render_frame_3d(
            xs,
            ys,
            zs,
            deps,
            bodies,
            body_field_pos,
            t,
            step,
            chi_min,
            N,
            cx,
            cy,
            cz,
            dchi_vmax,
            f_idx,
            title,
            camera_elev,
            camera_azim_start,
            camera_rotate_speed,
            figsize,
            dpi,
            cmap,
            show_chi=show_chi,
            chi_reveal_frame=chi_reveal_frame,
            n_frames=n_frames,
        )
        rendered.append(rgb)
        if verbose:
            print(f"\r  rendered {f_idx + 1}/{n_frames}", end="", flush=True)
    if verbose:
        print()

    # ── Step 3: save ──────────────────────────────────────────────────────────
    if save_path is None:
        slug = title.lower().replace(" ", "_")[:30]
        save_path = f"{slug}.mp4"

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    saved = None
    if p.suffix.lower() == ".mp4":
        saved = _save_mp4(rendered, str(p), fps)
    if saved is None:
        gif_p = p.with_suffix(".gif")
        saved = _save_gif(rendered, str(gif_p), fps)

    if verbose:
        if saved:
            print(f"Movie saved: {saved}")
        else:
            print("WARNING: could not save movie (no writer available).")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ══════════════════════════════════════════════════════════════════════════════


def _ensure_even(frame: np.ndarray) -> np.ndarray:
    """Crop to even dimensions (required by some video codecs)."""
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def _save_mp4(frames: list[np.ndarray], path: str, fps: int) -> str | None:
    """Try to save RGB frames as H.264 MP4."""
    # Approach 1: imageio with ffmpeg
    try:
        import imageio

        frames_even = [_ensure_even(f) for f in frames]
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"],
        )
        for fr in frames_even:
            writer.append_data(fr)
        writer.close()
        return path
    except Exception:
        pass

    # Approach 2: subprocess ffmpeg + PIL for individual PNGs
    try:
        import subprocess
        import tempfile

        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            for i, fr in enumerate(frames):
                Image.fromarray(_ensure_even(fr)).save(f"{td}/{i:06d}.png")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    f"{td}/%06d.png",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    path,
                ],
                capture_output=True,
                timeout=180,
            )
            if result.returncode == 0:
                return path
    except Exception:
        pass

    return None


def _save_gif(frames: list[np.ndarray], path: str, fps: int) -> str | None:
    """Save RGB frames as GIF via imageio or Pillow."""
    try:
        import imageio

        imageio.mimsave(path, frames, fps=fps)
        return path
    except Exception:
        pass

    try:
        from PIL import Image

        pil_frames = [Image.fromarray(fr) for fr in frames]
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=True,
            loop=0,
            duration=duration_ms,
        )
        return path
    except Exception:
        pass

    return None
