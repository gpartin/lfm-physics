"""Demo rendering for macroscopic LIMIT-02 two-body trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lfm.constants import CHI0
from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from lfm.fields.macroscopic import Limit02BodyProfile


def _trajectory_arrays(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    keys = (
        "time",
        "heavy_x",
        "heavy_y",
        "light_x",
        "light_y",
        "separation",
        "bearing_rad",
    )
    return {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key in keys}


def _translated_profile_slice(
    profile: Limit02BodyProfile,
    position: tuple[float, float, float],
) -> np.ndarray:
    from scipy.ndimage import shift

    z_index = int(round(profile.center[2])) % profile.grid_size
    base = np.asarray(profile.chi_delta[:, :, z_index], dtype=np.float64)
    offsets = (
        float(position[0]) - profile.center[0],
        float(position[1]) - profile.center[1],
    )
    return shift(
        base,
        shift=offsets,
        order=1,
        mode="wrap",
        prefilter=False,
    )


def combined_limit02_chi_slice(
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    heavy_position: tuple[float, float, float],
    light_position: tuple[float, float, float],
    *,
    chi0: float = CHI0,
) -> np.ndarray:
    """Return the translated and superposed equatorial LIMIT-02 chi slice."""
    if heavy.grid_size != light.grid_size:
        raise ValueError("body profiles must use the same grid")
    return (
        float(chi0)
        + _translated_profile_slice(heavy, heavy_position)
        + _translated_profile_slice(light, light_position)
    )


def plot_limit02_orbit_demo(
    rows: list[dict[str, Any]],
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    output_path: str | Path,
) -> Path:
    """Render a static orbit, chi substrate, and separation summary."""
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if len(rows) < 2:
        raise ValueError("at least two trajectory rows are required")
    arrays = _trajectory_arrays(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_heavy = (
        float(rows[-1]["heavy_x"]),
        float(rows[-1]["heavy_y"]),
        float(rows[-1]["heavy_z"]),
    )
    final_light = (
        float(rows[-1]["light_x"]),
        float(rows[-1]["light_y"]),
        float(rows[-1]["light_z"]),
    )
    chi_slice = combined_limit02_chi_slice(
        heavy,
        light,
        final_heavy,
        final_light,
    )
    angle = float(
        np.degrees(np.unwrap(arrays["bearing_rad"])[-1] - np.unwrap(arrays["bearing_rad"])[0])
    )

    figure = plt.figure(figsize=(13, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.15, 1.0))
    orbit_axis = figure.add_subplot(grid[:, 0])
    chi_axis = figure.add_subplot(grid[0, 1])
    separation_axis = figure.add_subplot(grid[1, 1])

    orbit_axis.plot(
        arrays["heavy_x"],
        arrays["heavy_y"],
        color="#2f7ed8",
        linewidth=2.0,
        label="Larger sphere",
    )
    orbit_axis.plot(
        arrays["light_x"],
        arrays["light_y"],
        color="#f2a93b",
        linewidth=1.2,
        label="Smaller sphere",
    )
    orbit_axis.add_patch(
        Circle(
            final_heavy[:2],
            heavy.radius,
            facecolor="#2f7ed8",
            edgecolor="#12365d",
            alpha=0.75,
        )
    )
    orbit_axis.add_patch(
        Circle(
            final_light[:2],
            light.radius,
            facecolor="#f2a93b",
            edgecolor="#754b0c",
            alpha=0.9,
        )
    )
    orbit_axis.set_aspect("equal")
    orbit_axis.set_xlabel("x (lattice cells)")
    orbit_axis.set_ylabel("y (lattice cells)")
    orbit_axis.set_title("Barycentric trajectories and true-scale bodies")
    orbit_axis.grid(alpha=0.2)
    orbit_axis.legend(loc="best")

    image = chi_axis.imshow(
        chi_slice.T,
        origin="lower",
        extent=(0, heavy.grid_size, 0, heavy.grid_size),
        cmap="magma",
        aspect="equal",
    )
    chi_axis.add_patch(
        Circle(
            final_heavy[:2],
            heavy.radius,
            facecolor="none",
            edgecolor="cyan",
            linewidth=1.3,
        )
    )
    chi_axis.add_patch(
        Circle(
            final_light[:2],
            light.radius,
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
        )
    )
    chi_axis.set_title("Translated LIMIT-02 chi substrate")
    chi_axis.set_xlabel("x (lattice cells)")
    chi_axis.set_ylabel("y (lattice cells)")
    figure.colorbar(image, ax=chi_axis, label="chi")

    separation_axis.plot(
        arrays["time"],
        arrays["separation"],
        color="#5a8f29",
        linewidth=1.5,
    )
    separation_axis.axhline(
        heavy.radius + light.radius,
        color="#a33a3a",
        linewidth=1.0,
        linestyle="--",
        label="Surface contact",
    )
    separation_axis.set_xlabel("LFM time")
    separation_axis.set_ylabel("Centre separation (cells)")
    separation_axis.set_title(
        f"Mass ratio {heavy.mass / light.mass:.4f}:1 | net angle {angle:.1f} deg"
    )
    separation_axis.grid(alpha=0.2)
    separation_axis.legend(loc="best")

    figure.suptitle(
        "LFM macroscopic LIMIT-02 two-sphere orbit",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def animate_limit02_orbit_demo(
    rows: list[dict[str, Any]],
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    output_path: str | Path,
    *,
    max_frames: int = 240,
    fps: int = 30,
) -> Path:
    """Render an orbit animation beside the translated live chi slice."""
    _require_matplotlib()
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if len(rows) < 2:
        raise ValueError("at least two trajectory rows are required")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = _trajectory_arrays(rows)
    frame_count = min(max_frames, len(rows))
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, frame_count).astype(int))

    initial_heavy = (
        float(rows[0]["heavy_x"]),
        float(rows[0]["heavy_y"]),
        float(rows[0]["heavy_z"]),
    )
    initial_light = (
        float(rows[0]["light_x"]),
        float(rows[0]["light_y"]),
        float(rows[0]["light_z"]),
    )
    initial_chi = combined_limit02_chi_slice(
        heavy,
        light,
        initial_heavy,
        initial_light,
    )
    chi_min = float(CHI0 + np.min(heavy.chi_delta) + np.min(light.chi_delta))
    chi_max = float(CHI0 + np.max(heavy.chi_delta) + np.max(light.chi_delta))

    figure, (orbit_axis, chi_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5.8),
        constrained_layout=True,
    )
    orbit_axis.set_aspect("equal")
    orbit_axis.set_xlim(18, 110)
    orbit_axis.set_ylim(18, 110)
    orbit_axis.set_xlabel("x (lattice cells)")
    orbit_axis.set_ylabel("y (lattice cells)")
    orbit_axis.set_title("Two density spheres")
    orbit_axis.grid(alpha=0.2)
    (heavy_trail,) = orbit_axis.plot([], [], color="#2f7ed8", linewidth=2.0)
    (light_trail,) = orbit_axis.plot([], [], color="#f2a93b", linewidth=1.2)
    heavy_circle = Circle(
        initial_heavy[:2],
        heavy.radius,
        facecolor="#2f7ed8",
        edgecolor="#12365d",
        alpha=0.8,
    )
    light_circle = Circle(
        initial_light[:2],
        light.radius,
        facecolor="#f2a93b",
        edgecolor="#754b0c",
        alpha=0.95,
    )
    orbit_axis.add_patch(heavy_circle)
    orbit_axis.add_patch(light_circle)
    status = orbit_axis.text(
        0.02,
        0.98,
        "",
        transform=orbit_axis.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    image = chi_axis.imshow(
        initial_chi.T,
        origin="lower",
        extent=(0, heavy.grid_size, 0, heavy.grid_size),
        cmap="magma",
        vmin=chi_min,
        vmax=chi_max,
        aspect="equal",
    )
    chi_axis.set_xlim(18, 110)
    chi_axis.set_ylim(18, 110)
    chi_axis.set_xlabel("x (lattice cells)")
    chi_axis.set_ylabel("y (lattice cells)")
    chi_axis.set_title("Live LIMIT-02 chi substrate")
    chi_heavy_circle = Circle(
        initial_heavy[:2],
        heavy.radius,
        facecolor="none",
        edgecolor="cyan",
        linewidth=1.3,
    )
    chi_light_circle = Circle(
        initial_light[:2],
        light.radius,
        facecolor="none",
        edgecolor="white",
        linewidth=1.0,
    )
    chi_axis.add_patch(chi_heavy_circle)
    chi_axis.add_patch(chi_light_circle)
    figure.colorbar(image, ax=chi_axis, label="chi")
    figure.suptitle(f"LFM LIMIT-02 orbit | mass ratio {heavy.mass / light.mass:.4f}:1")

    unwrapped = np.unwrap(arrays["bearing_rad"])

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        heavy_position = (
            float(rows[index]["heavy_x"]),
            float(rows[index]["heavy_y"]),
            float(rows[index]["heavy_z"]),
        )
        light_position = (
            float(rows[index]["light_x"]),
            float(rows[index]["light_y"]),
            float(rows[index]["light_z"]),
        )
        heavy_trail.set_data(
            arrays["heavy_x"][: index + 1],
            arrays["heavy_y"][: index + 1],
        )
        light_trail.set_data(
            arrays["light_x"][: index + 1],
            arrays["light_y"][: index + 1],
        )
        heavy_circle.center = heavy_position[:2]
        light_circle.center = light_position[:2]
        chi_heavy_circle.center = heavy_position[:2]
        chi_light_circle.center = light_position[:2]
        image.set_data(
            combined_limit02_chi_slice(
                heavy,
                light,
                heavy_position,
                light_position,
            ).T
        )
        angle = float(np.degrees(unwrapped[index] - unwrapped[0]))
        status.set_text(
            f"time {arrays['time'][index]:.1f}\n"
            f"separation {arrays['separation'][index]:.2f}\n"
            f"angle {angle:.1f} deg"
        )
        return (
            heavy_trail,
            light_trail,
            heavy_circle,
            light_circle,
            chi_heavy_circle,
            chi_light_circle,
            image,
            status,
        )

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )
    if output.suffix.lower() == ".mp4" and animation.writers.is_available("ffmpeg"):
        movie.save(
            output,
            writer=animation.FFMpegWriter(
                fps=fps,
                bitrate=2200,
                metadata={"title": "LFM LIMIT-02 two-sphere orbit"},
            ),
            dpi=120,
        )
    else:
        output = output.with_suffix(".gif")
        movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(figure)
    return output


def _limit02_surface_data(
    chi_slice: np.ndarray,
    *,
    lower: int = 20,
    upper: int = 108,
    stride: int = 4,
    vertical_scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a display-sampled lattice surface and its unscaled chi values."""
    coordinates = np.arange(lower, upper + 1, stride, dtype=np.intp)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
    chi_values = chi_slice[np.ix_(coordinates, coordinates)].T
    z_grid = (chi_values - CHI0) * float(vertical_scale)
    return x_grid, y_grid, z_grid, chi_values


def _sphere_mesh(
    center: tuple[float, float, float],
    radius: float,
    *,
    longitude_samples: int = 24,
    latitude_samples: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude = np.linspace(0.0, 2.0 * np.pi, longitude_samples)
    latitude = np.linspace(0.0, np.pi, latitude_samples)
    x_mesh = center[0] + radius * np.outer(
        np.cos(longitude),
        np.sin(latitude),
    )
    y_mesh = center[1] + radius * np.outer(
        np.sin(longitude),
        np.sin(latitude),
    )
    z_mesh = center[2] + radius * np.outer(
        np.ones_like(longitude),
        np.cos(latitude),
    )
    return x_mesh, y_mesh, z_mesh


def _configure_limit02_3d_axis(axis) -> None:
    axis.set_facecolor("#071019")
    axis.set_xlim(20, 108)
    axis.set_ylim(20, 108)
    axis.set_zlim(-7, 15)
    axis.set_box_aspect((88, 88, 22))
    axis.view_init(elev=31, azim=-57)
    axis.set_xlabel("lattice x", color="#a6b6c8", labelpad=8)
    axis.set_ylabel("lattice y", color="#a6b6c8", labelpad=8)
    axis.set_zlabel("")
    axis.tick_params(colors="#7890a4", labelsize=7, pad=0)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.pane.fill = False
        coordinate_axis.pane.set_edgecolor("#25384a")
        coordinate_axis._axinfo["grid"]["color"] = "#183047"


def _draw_limit02_3d_scene(
    axis,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    index: int,
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    *,
    color_map,
    color_norm,
    audit_status: str,
    audit_residual: float,
    vertical_scale: float = 2.5,
) -> None:
    axis.clear()
    _configure_limit02_3d_axis(axis)
    row = rows[index]
    heavy_position = (
        float(row["heavy_x"]),
        float(row["heavy_y"]),
        float(row["heavy_z"]),
    )
    light_position = (
        float(row["light_x"]),
        float(row["light_y"]),
        float(row["light_z"]),
    )
    chi_slice = combined_limit02_chi_slice(
        heavy,
        light,
        heavy_position,
        light_position,
    )
    x_grid, y_grid, z_grid, chi_values = _limit02_surface_data(
        chi_slice,
        vertical_scale=vertical_scale,
    )
    chi_depth = np.maximum(CHI0 - chi_values, 0.0)
    face_colors = color_map(color_norm(chi_depth))
    axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        facecolors=face_colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.94,
        zorder=1,
    )
    axis.plot_wireframe(
        x_grid,
        y_grid,
        z_grid + 0.04,
        rstride=1,
        cstride=1,
        color="#d8e5ef",
        linewidth=0.35,
        alpha=0.52,
        zorder=2,
    )

    def local_surface_height(position: tuple[float, float, float]) -> float:
        x_index = int(round(position[0])) % chi_slice.shape[0]
        y_index = int(round(position[1])) % chi_slice.shape[1]
        return float(vertical_scale * (chi_slice[x_index, y_index] - CHI0))

    heavy_surface = local_surface_height(heavy_position)
    light_surface = local_surface_height(light_position)
    heavy_center = (
        heavy_position[0],
        heavy_position[1],
        heavy_surface + heavy.radius,
    )
    light_center = (
        light_position[0],
        light_position[1],
        light_surface + light.radius,
    )
    heavy_mesh = _sphere_mesh(heavy_center, heavy.radius)
    light_mesh = _sphere_mesh(light_center, light.radius)
    axis.plot_surface(
        *heavy_mesh,
        color="#38bdf8",
        edgecolor="#d5f4ff",
        linewidth=0.22,
        antialiased=True,
        shade=True,
        alpha=0.98,
        zorder=6,
    )
    axis.plot_surface(
        *light_mesh,
        color="#fbbf24",
        edgecolor="#fff1a8",
        linewidth=0.20,
        antialiased=True,
        shade=True,
        alpha=1.0,
        zorder=7,
    )

    heavy_acceleration = np.asarray(
        [
            float(row["heavy_ax"]),
            float(row["heavy_ay"]),
            float(row["heavy_az"]),
        ],
        dtype=np.float64,
    )
    light_acceleration = np.asarray(
        [
            float(row["light_ax"]),
            float(row["light_ay"]),
            float(row["light_az"]),
        ],
        dtype=np.float64,
    )
    for center, acceleration, arrow_color in (
        (heavy_center, heavy_acceleration, "#d5f4ff"),
        (light_center, light_acceleration, "#fff1a8"),
    ):
        magnitude = float(np.linalg.norm(acceleration))
        if magnitude > 0.0:
            direction = acceleration / magnitude
            axis.quiver(
                center[0],
                center[1],
                min(center[2] + 1.0, 13.5),
                direction[0],
                direction[1],
                0.0,
                length=7.0,
                normalize=True,
                color=arrow_color,
                linewidth=1.8,
                arrow_length_ratio=0.28,
                zorder=10,
            )

    trail_height = 13.4
    axis.plot(
        arrays["heavy_x"][: index + 1],
        arrays["heavy_y"][: index + 1],
        np.full(index + 1, trail_height),
        color="#38bdf8",
        linewidth=1.8,
        alpha=0.9,
        zorder=4,
    )
    axis.plot(
        arrays["light_x"][: index + 1],
        arrays["light_y"][: index + 1],
        np.full(index + 1, trail_height),
        color="#fbbf24",
        linewidth=1.2,
        alpha=0.82,
        zorder=5,
    )
    axis.text(
        heavy_center[0],
        heavy_center[1],
        min(14.5, heavy_center[2] + heavy.radius + 0.5),
        "81.3017 mass",
        color="#d5f4ff",
        fontsize=8,
        ha="center",
        zorder=8,
    )
    axis.text(
        light_center[0],
        light_center[1],
        min(14.5, light_center[2] + light.radius + 0.7),
        "1 mass",
        color="#fff1a8",
        fontsize=8,
        ha="center",
        zorder=9,
    )

    unwrapped = np.unwrap(arrays["bearing_rad"])
    angle = float(np.degrees(unwrapped[index] - unwrapped[0]))
    axis.text2D(
        0.52,
        0.97,
        f"RED-TEAM {audit_status} | no GM/r^2 | no Kepler | no Einstein solver",
        transform=axis.transAxes,
        color="#90f5c1" if audit_status == "PASS" else "#fca5a5",
        fontsize=8,
        va="top",
        ha="center",
        bbox={
            "facecolor": "#0b1722",
            "edgecolor": "#315268",
            "alpha": 0.88,
            "boxstyle": "round,pad=0.35",
        },
    )
    axis.text2D(
        0.02,
        0.925,
        (
            f"time {arrays['time'][index]:.0f} | "
            f"separation {arrays['separation'][index]:.2f} | "
            f"angle {angle:.1f} deg | "
            f"arrows = -(c^2/chi0) grad19(delta_chi)"
        ),
        transform=axis.transAxes,
        color="#edf6ff",
        fontsize=9,
        va="top",
        ha="left",
    )
    heavy_acceleration_norm = float(np.linalg.norm(heavy_acceleration))
    light_acceleration_norm = float(np.linalg.norm(light_acceleration))
    maximum_depth = float(np.max(chi_depth))
    pipeline_boxes = (
        (
            0.01,
            f"SOURCE rho\nM_H:M_L = {heavy.mass / light.mass:.4f}:1",
            "#38bdf8",
        ),
        (
            0.255,
            f"GOV-02 -> LIMIT-02\nD19 dchi =\nk(rho-<rho>)\ndepth max = {maximum_depth:.3f}",
            "#c084fc",
        ),
        (
            0.50,
            "GOV-01 -> WKB\n"
            "a = -(c^2/chi0)\n"
            "grad19 dchi\n"
            f"|aH|={heavy_acceleration_norm:.2e}\n"
            f"|aL|={light_acceleration_norm:.2e}",
            "#34d399",
        ),
        (
            0.745,
            "CENTER UPDATE\nvelocity-Verlet, dt=0.25\ntrajectory not prescribed",
            "#fbbf24",
        ),
    )
    for x_position, text, edge_color in pipeline_boxes:
        axis.text2D(
            x_position,
            0.015,
            text,
            transform=axis.transAxes,
            color="#edf6ff",
            fontsize=6.8,
            va="bottom",
            ha="left",
            family="monospace",
            bbox={
                "facecolor": "#0b1722",
                "edgecolor": edge_color,
                "alpha": 0.90,
                "boxstyle": "round,pad=0.35",
            },
        )
    for x_position in (0.235, 0.48, 0.725):
        axis.text2D(
            x_position,
            0.065,
            "->",
            transform=axis.transAxes,
            color="#91a9bc",
            fontsize=10,
            va="center",
            ha="center",
        )
    axis.text2D(
        0.02,
        0.885,
        (f"128^3 grid | 19-point stencil | LIMIT-02 residual <= {audit_residual:.2e}"),
        transform=axis.transAxes,
        color="#91a9bc",
        fontsize=7.5,
        va="top",
        ha="left",
    )
    axis.set_title(
        "LFM coupled-equation substrate orbit",
        color="#edf6ff",
        fontsize=15,
        pad=12,
    )


def plot_limit02_orbit_3d_demo(
    rows: list[dict[str, Any]],
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    output_path: str | Path,
    *,
    frame_index: int = -1,
    audit_status: str = "NOT RUN",
    audit_residual: float = float("nan"),
) -> Path:
    """Render a single 3-D lattice, color-coded chi surface, and two bodies."""
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import PowerNorm

    if len(rows) < 2:
        raise ValueError("at least two trajectory rows are required")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = _trajectory_arrays(rows)
    index = frame_index % len(rows)
    depth_max = float(-np.min(heavy.chi_delta) - np.min(light.chi_delta))
    color_norm = PowerNorm(gamma=0.33, vmin=0.0, vmax=depth_max)
    color_map = plt.get_cmap("magma")

    figure = plt.figure(figsize=(12, 7.5), facecolor="#071019")
    axis = figure.add_subplot(
        111,
        projection="3d",
        computed_zorder=False,
    )
    _draw_limit02_3d_scene(
        axis,
        rows,
        arrays,
        index,
        heavy,
        light,
        color_map=color_map,
        color_norm=color_norm,
        audit_status=audit_status,
        audit_residual=audit_residual,
    )
    scalar_map = ScalarMappable(norm=color_norm, cmap=color_map)
    scalar_map.set_array([])
    color_bar = figure.colorbar(
        scalar_map,
        ax=axis,
        shrink=0.62,
        pad=0.015,
        aspect=24,
    )
    color_bar.set_label("chi0 - chi (substrate depth)", color="#c8d8e8")
    color_bar.ax.tick_params(colors="#91a9bc", labelsize=8)
    figure.text(
        0.5,
        0.025,
        (
            "Color = chi0 - chi (PowerNorm gamma=0.33) | mesh = lattice | "
            "height = 2.5x display amplification of chi - chi0"
        ),
        color="#91a9bc",
        fontsize=9,
        ha="center",
    )
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def animate_limit02_orbit_3d_demo(
    rows: list[dict[str, Any]],
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    output_path: str | Path,
    *,
    max_frames: int = 180,
    fps: int = 18,
    audit_status: str = "NOT RUN",
    audit_residual: float = float("nan"),
) -> Path:
    """Animate one 3-D lattice scene with color-coded live chi geometry."""
    _require_matplotlib()
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import PowerNorm

    if len(rows) < 2:
        raise ValueError("at least two trajectory rows are required")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = _trajectory_arrays(rows)
    frame_count = min(max_frames, len(rows))
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, frame_count).astype(int))
    depth_max = float(-np.min(heavy.chi_delta) - np.min(light.chi_delta))
    color_norm = PowerNorm(gamma=0.33, vmin=0.0, vmax=depth_max)
    color_map = plt.get_cmap("magma")

    figure = plt.figure(figsize=(12, 7.5), facecolor="#071019")
    axis = figure.add_subplot(
        111,
        projection="3d",
        computed_zorder=False,
    )
    scalar_map = ScalarMappable(norm=color_norm, cmap=color_map)
    scalar_map.set_array([])
    color_bar = figure.colorbar(
        scalar_map,
        ax=axis,
        shrink=0.62,
        pad=0.015,
        aspect=24,
    )
    color_bar.set_label("chi0 - chi (substrate depth)", color="#c8d8e8")
    color_bar.ax.tick_params(colors="#91a9bc", labelsize=8)
    figure.text(
        0.5,
        0.025,
        (
            "Color = chi0 - chi (PowerNorm gamma=0.33) | mesh = lattice | "
            "height = 2.5x display amplification of chi - chi0"
        ),
        color="#91a9bc",
        fontsize=9,
        ha="center",
    )

    def update(frame_number: int):
        _draw_limit02_3d_scene(
            axis,
            rows,
            arrays,
            int(frame_indices[frame_number]),
            heavy,
            light,
            color_map=color_map,
            color_norm=color_norm,
            audit_status=audit_status,
            audit_residual=audit_residual,
        )
        return ()

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )
    if output.suffix.lower() == ".mp4" and animation.writers.is_available("ffmpeg"):
        movie.save(
            output,
            writer=animation.FFMpegWriter(
                fps=fps,
                bitrate=3200,
                metadata={"title": "LFM LIMIT-02 3-D lattice orbit"},
            ),
            dpi=120,
        )
    else:
        output = output.with_suffix(".gif")
        movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(figure)
    return output


__all__ = [
    "animate_limit02_orbit_3d_demo",
    "animate_limit02_orbit_demo",
    "combined_limit02_chi_slice",
    "plot_limit02_orbit_3d_demo",
    "plot_limit02_orbit_demo",
]
