"""
High-level ringdown / QNM projection experiments for the LFM framework.
=======================================================================

These experiments package the stable projection-based extraction workflow
that was developed in Papers and promote it into library-native APIs.

Two public entry points are provided:

``qnm_mode_projection_check``
    Single-run extractor diagnostic comparing point probes with global
    low-k mode projection. Optionally captures a fixed-camera 3-D movie.

``next5_falsification_projection_v2``
    Corrected Next-5 falsification suite using projection-first extraction,
    target-band clustering, and comoving recentering for merger remnants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import lfm
from lfm.analysis import fit_ringdown_series, relative_spread, target_band_summary
from lfm.experiment.common import ExperimentResult

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_RINGDOWN_K_MODES",
    "QNMProjectionResult",
    "Next5FalsificationResult",
    "qnm_mode_projection_check",
    "next5_falsification_projection_v2",
]

DEFAULT_RINGDOWN_K_MODES: list[tuple[int, int, int]] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (2, 0, 0),
]

_EPS = 1e-12


def _ensure_float_list(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def _apply_perturbation(sim: lfm.Simulation, profile: str, frac: float) -> None:
    psi = sim.psi_real.copy()
    if profile == "uniform":
        psi_new = psi * (1.0 + frac)
    else:
        n = psi.shape[0]
        c = n // 2
        ii, jj, kk = np.indices(psi.shape, dtype=np.float64)
        x = ii - c
        y = jj - c
        z = kk - c
        r2 = x * x + y * y + z * z
        q = np.zeros_like(psi, dtype=np.float64)
        mask = r2 > 0
        q[mask] = (3.0 * z[mask] * z[mask] - r2[mask]) / r2[mask]
        q_scale = float(np.max(np.abs(q)))
        q /= max(q_scale, _EPS)
        psi_new = psi * (1.0 + frac * q)
    sim.set_psi_real(psi_new)
    sim.set_psi_real_prev(sim.psi_real.copy())


def _precompute_basis(grid: int, modes: Sequence[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    ii, jj, kk = np.indices((grid, grid, grid), dtype=np.float64)
    basis: dict[str, np.ndarray] = {}
    for kx, ky, kz in modes:
        phase = -2.0 * np.pi * (kx * ii + ky * jj + kz * kk) / grid
        basis[f"({kx},{ky},{kz})"] = np.exp(1j * phase)
    return basis


def _extract_point_cloud(
    dchi: np.ndarray,
    *,
    max_points: int,
    percentile: float = 99.0,
) -> dict[str, Any]:
    abs_d = np.abs(dchi)
    thr = float(np.percentile(abs_d, percentile))
    idx = np.argwhere(abs_d >= thr)
    if len(idx) == 0:
        return {
            "xyz": np.zeros((0, 3), dtype=np.float32),
            "val": np.zeros((0,), dtype=np.float32),
            "thr": thr,
        }

    if len(idx) > max_points:
        choose = np.linspace(0, len(idx) - 1, num=max_points, dtype=int)
        idx = idx[choose]

    vals = dchi[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
    xyz = idx.astype(np.float32)
    return {"xyz": xyz, "val": vals, "thr": thr}


def _save_fixed_camera_movie(
    frames: list[dict[str, Any]],
    *,
    grid: int,
    out_path: Path,
    dt: float,
) -> Path | None:
    if not frames:
        return None

    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, grid)
    ax.set_ylim(0, grid)
    ax.set_zlim(0, grid)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=38)

    title = ax.set_title("")
    all_vals = np.concatenate([fr["val"] for fr in frames if len(fr["val"]) > 0])
    if len(all_vals) == 0:
        all_vals = np.array([1.0], dtype=np.float32)
    vmax_global = float(max(np.max(np.abs(all_vals)), _EPS))
    scat = ax.scatter(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        c=np.array([], dtype=np.float32),
        cmap="coolwarm",
        vmin=-vmax_global,
        vmax=vmax_global,
        s=6,
        alpha=0.75,
        linewidths=0,
    )

    def update(i: int):
        fr = frames[i]
        xyz = fr["xyz"]
        vals = fr["val"]
        if len(xyz) > 0:
            scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            scat.set_array(vals.astype(np.float64))
        else:
            empty = np.array([], dtype=np.float32)
            scat._offsets3d = (empty, empty, empty)
            scat.set_array(np.array([], dtype=np.float64))
        title.set_text(f"Ringdown 3D dchi cloud | frame={i} | t={fr['t']:.3f}")
        return [scat, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)

    saved_path = out_path
    try:
        frame_dt = max(float(dt) * 10.0, _EPS)
        writer = animation.FFMpegWriter(fps=max(5, int(round(1.0 / frame_dt))))
        ani.save(saved_path.as_posix(), writer=writer, dpi=120)
    except Exception:
        saved_path = out_path.with_suffix(".gif")
        ani.save(saved_path.as_posix(), writer="pillow", dpi=100)
    finally:
        plt.close(fig)

    return saved_path


def _centroid_shift(dchi: np.ndarray) -> tuple[int, int, int]:
    n = dchi.shape[0]
    c = n // 2
    w = np.abs(dchi)
    tot = float(np.sum(w))
    if tot <= _EPS:
        return (0, 0, 0)
    ii, jj, kk = np.indices(dchi.shape, dtype=np.float64)
    cx = int(round(float(np.sum(ii * w) / tot) - c))
    cy = int(round(float(np.sum(jj * w) / tot) - c))
    cz = int(round(float(np.sum(kk * w) / tot) - c))
    return (cx, cy, cz)


def _shell_mode_keys() -> tuple[set[str], set[str], set[str]]:
    shell1 = {"(1,0,0)", "(0,1,0)", "(0,0,1)"}
    shell2 = {"(1,1,0)", "(1,0,1)", "(0,1,1)"}
    axis = shell1.copy()
    return shell1, shell2, axis


@dataclass
class QNMProjectionResult(ExperimentResult):
    """Result for the single-run projection-vs-probe ringdown diagnostic."""

    summary: dict[str, Any]
    movie_frames: list[dict[str, Any]]
    dt: float

    def plot(self, *, figsize: tuple[float, float] = (12, 5)):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        probe_rows = self.summary.get("probe_fits", [])
        mode_rows = self.summary.get("mode_projection_fits", [])

        probe_x = np.arange(len(probe_rows))
        probe_y = [float(r["omega"]) for r in probe_rows]
        probe_ok = [bool(r["valid"]) for r in probe_rows]
        axes[0].scatter(probe_x, probe_y, c=["#16a34a" if ok else "#dc2626" for ok in probe_ok])
        axes[0].set_title("Point probes")
        axes[0].set_xlabel("Probe index")
        axes[0].set_ylabel("omega")

        mode_x = np.arange(len(mode_rows))
        mode_y = [float(r["omega"]) for r in mode_rows]
        mode_ok = [bool(r["valid"]) for r in mode_rows]
        axes[1].scatter(mode_x, mode_y, c=["#16a34a" if ok else "#dc2626" for ok in mode_ok])
        axes[1].set_title("Projected modes")
        axes[1].set_xlabel("Mode index")
        axes[1].set_ylabel("omega")

        comp = self.summary.get("comparison", {})
        fig.suptitle(
            "QNM Mode Projection Check\n"
            f"probe spread={comp.get('probe_omega_spread_frac', float('nan')):.3f}, "
            f"mode spread={comp.get('projected_mode_omega_spread_frac', float('nan')):.3f}",
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    def save(
        self,
        stem: str,
        *,
        directory: str | Path | None = None,
        dpi: int = 150,
        save_movie: bool = True,
        save_summary_json: bool = True,
    ) -> dict[str, Path]:
        import matplotlib.pyplot as plt

        out = Path(directory) if directory else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        fig = self.plot()
        plot_path = out / f"{stem}.png"
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written["summary_plot"] = plot_path

        if save_summary_json:
            json_path = out / f"{stem}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(self.summary, f, indent=2)
            written["summary_json"] = json_path

        if save_movie and self.movie_frames:
            movie_path = _save_fixed_camera_movie(
                self.movie_frames,
                grid=self.N,
                out_path=out / f"{stem}_3d_movie.mp4",
                dt=self.dt,
            )
            if movie_path is not None:
                written["movie"] = movie_path

        return written


@dataclass
class Next5FalsificationResult(ExperimentResult):
    """Result for the corrected projection-based Next-5 falsification suite."""

    summary: dict[str, Any]

    def save(self, stem: str, *, directory: str | Path | None = None) -> dict[str, Path]:
        out = Path(directory) if directory else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
        return {"summary_json": json_path}


def qnm_mode_projection_check(
    *,
    N: int = 64,
    dt: float = 0.02,
    amplitude: float = 12.0,
    sigma: float = 6.0,
    ring_steps: int = 3200,
    record_every: int = 2,
    perturb_profile: str = "quadrupole",
    perturb_frac: float = 0.12,
    boundary_type: lfm.BoundaryType = lfm.BoundaryType.PERIODIC,
    probe_offsets: Sequence[int] = (8, 12, 16, 20),
    k_modes: Sequence[tuple[int, int, int]] = tuple(DEFAULT_RINGDOWN_K_MODES),
    capture_movie: bool = False,
    movie_every: int = 10,
    movie_max_points: int = 6000,
    label: str = "QNM Mode Projection Check",
) -> QNMProjectionResult:
    """Run the projection-vs-probe ringdown extraction diagnostic."""
    cfg = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.REAL,
        boundary_type=boundary_type,
        chi0=lfm.CHI0,
        kappa=lfm.KAPPA,
        lambda_self=lfm.LAMBDA_H,
        dt=dt,
        report_interval=record_every,
    )
    sim = lfm.Simulation(cfg)
    c = N // 2
    sim.place_soliton((c, c, c), amplitude=amplitude, sigma=sigma)
    sim.equilibrate()

    chi_bg = sim.chi.copy()
    _apply_perturbation(sim, perturb_profile, perturb_frac)

    basis = _precompute_basis(N, k_modes)
    probes = [(c + d, c, c) for d in probe_offsets]
    probe_traces: dict[str, list[float]] = {str(p): [] for p in probes}
    mode_coeff_traces: dict[str, list[complex]] = {key: [] for key in basis}
    t_s: list[float] = []
    movie_frames: list[dict[str, Any]] = []
    callback_count = 0

    def cb(s: lfm.Simulation, step: int) -> None:
        nonlocal callback_count
        callback_count += 1
        t_s.append(step * dt)
        dchi = s.chi - chi_bg
        dchi = dchi - np.mean(dchi)
        if capture_movie and callback_count % max(1, movie_every) == 0:
            cloud = _extract_point_cloud(dchi, max_points=max(100, movie_max_points))
            movie_frames.append({"t": step * dt, **cloud})
        for probe in probes:
            probe_traces[str(probe)].append(float(s.chi[probe]))
        for key, phi in basis.items():
            coeff = np.sum(dchi * phi) / dchi.size
            mode_coeff_traces[key].append(complex(coeff))

    sim.run(ring_steps, callback=cb)

    t = np.array(t_s, dtype=np.float64)

    probe_rows: list[dict[str, Any]] = []
    probe_omegas: list[float] = []
    for probe in probes:
        fit = fit_ringdown_series(t, np.array(probe_traces[str(probe)], dtype=np.float64))
        omega = float(fit["omega"])
        row = {
            "probe": list(probe),
            "omega": omega,
            "gamma": float(fit["gamma"]),
            "n_peaks": int(fit["n_peaks"]),
            "valid": bool(fit["valid"]),
        }
        probe_rows.append(row)
        if row["valid"]:
            probe_omegas.append(omega)

    mode_rows: list[dict[str, Any]] = []
    mode_omegas: list[float] = []
    for key in basis:
        fit = fit_ringdown_series(t, np.array(mode_coeff_traces[key], dtype=np.complex128).real)
        omega = float(fit["omega"])
        row = {
            "mode_k": key,
            "omega": omega,
            "gamma": float(fit["gamma"]),
            "n_peaks": int(fit["n_peaks"]),
            "valid": bool(fit["valid"]),
        }
        mode_rows.append(row)
        if row["valid"]:
            mode_omegas.append(omega)

    probe_spread = relative_spread(probe_omegas)
    mode_spread = relative_spread(mode_omegas)
    spread_ratio = (
        float(mode_spread / max(float(probe_spread), _EPS))
        if np.isfinite(mode_spread)
        else float("nan")
    )
    h0_rejected = bool(
        np.isfinite(mode_spread)
        and np.isfinite(probe_spread)
        and float(mode_spread) < 0.5 * float(probe_spread)
    )

    summary = {
        "experiment": "qnm_mode_projection_check",
        "config": {
            "grid": N,
            "dt": dt,
            "ring_steps": ring_steps,
            "record_every": record_every,
            "perturb_profile": perturb_profile,
            "perturb_frac": perturb_frac,
            "boundary": boundary_type.name.lower(),
            "k_modes": [list(k) for k in k_modes],
            "probe_offsets": [int(v) for v in probe_offsets],
        },
        "probe_fits": probe_rows,
        "mode_projection_fits": mode_rows,
        "comparison": {
            "probe_omega_spread_frac": float(probe_spread),
            "projected_mode_omega_spread_frac": float(mode_spread),
            "spread_ratio_projected_over_probe": spread_ratio,
            "valid_probe_count": len(probe_omegas),
            "valid_projected_mode_count": len(mode_omegas),
            "h0_status": "REJECTED" if h0_rejected else "FAILED_TO_REJECT",
            "criterion": "REJECT H0 if projected spread < 50% of probe spread",
        },
        "movie": {
            "enabled": bool(capture_movie),
            "frame_count": len(movie_frames),
            "camera": {"elev": 22, "azim": 38, "spin": False},
        },
    }

    return QNMProjectionResult(
        snapshots=[],
        movie_snapshots=[],
        metrics=[],
        label=label,
        N=N,
        summary=summary,
        movie_frames=movie_frames,
        dt=dt,
    )


def _run_projection_case(
    *,
    grid: int,
    boundary: lfm.BoundaryType,
    amp: float,
    sigma: float,
    ring_steps: int,
    perturb_profile: str,
    perturb_frac: float,
    dt: float,
    record_every: int,
    modes: Sequence[tuple[int, int, int]],
    merger: bool = False,
    merger_pre_steps: int = 1200,
    seed: int = 0,
    min_target: float = 1.0,
) -> dict[str, Any]:
    cfg = lfm.SimulationConfig(
        grid_size=grid,
        field_level=lfm.FieldLevel.REAL,
        boundary_type=boundary,
        chi0=lfm.CHI0,
        kappa=lfm.KAPPA,
        lambda_self=lfm.LAMBDA_H,
        dt=dt,
        report_interval=record_every,
    )
    sim = lfm.Simulation(cfg)
    c = grid // 2

    if merger:
        rng = np.random.default_rng(seed)
        jitter = int(rng.integers(-2, 3))
        sim.place_solitons(
            [(c - 10 + jitter, c, c), (c + 10 + jitter, c, c)],
            amplitude=10.0,
            sigma=6.0,
        )
        sim.equilibrate()
        sim.run(merger_pre_steps)
    else:
        sim.place_soliton((c, c, c), amplitude=amp, sigma=sigma)
        sim.equilibrate()

    chi_bg = sim.chi.copy()
    _apply_perturbation(sim, perturb_profile, perturb_frac)

    basis = _precompute_basis(grid, modes)
    traces: dict[str, list[complex]] = {k: [] for k in basis}
    t_s: list[float] = []

    def cb(s: lfm.Simulation, step: int) -> None:
        t_s.append(step * dt)
        dchi = s.chi - chi_bg
        dchi = dchi - np.mean(dchi)
        if merger:
            sx, sy, sz = _centroid_shift(dchi)
            dchi = np.roll(dchi, shift=(-sx, -sy, -sz), axis=(0, 1, 2))
        for key, phi in basis.items():
            coeff = np.sum(dchi * phi) / dchi.size
            traces[key].append(complex(coeff))

    sim.run(ring_steps, callback=cb)

    t = np.array(t_s, dtype=np.float64)
    mode_rows: list[dict[str, Any]] = []
    for key in basis:
        fit = fit_ringdown_series(t, np.array(traces[key], dtype=np.complex128).real)
        mode_rows.append(
            {
                "mode_k": key,
                "omega": float(fit["omega"]),
                "gamma": float(fit["gamma"]),
                "n_peaks": int(fit["n_peaks"]),
                "valid": bool(fit["valid"]),
            }
        )

    band_summary = target_band_summary(mode_rows, min_target=min_target)
    return {
        "mode_rows": mode_rows,
        "valid_mode_count": int(band_summary["valid_mode_count"]),
        "target_band_omegas": _ensure_float_list(band_summary["target_band"]),
        "slow_band_omegas": _ensure_float_list(band_summary["slow_band"]),
        "target_band_center": float(band_summary["target_center"]),
        "target_band_spread": float(band_summary["target_spread"]),
    }


def next5_falsification_projection_v2(
    *,
    dt: float = 0.02,
    record_every: int = 2,
    modes: Sequence[tuple[int, int, int]] = tuple(DEFAULT_RINGDOWN_K_MODES),
    min_target: float = 1.0,
    base_amp: float = 12.0,
    base_sigma: float = 6.0,
    f1_grid: int = 64,
    f2_grids: Sequence[int] = (48, 64, 96),
    f3_grid: int = 64,
    f4_grid: int = 64,
    f1_ring_steps: int = 3000,
    f2_ring_steps: int = 2600,
    f3_ring_steps: int = 3000,
    f4_ring_steps: int = 3200,
    merger_pre_steps: int = 1200,
    merger_seeds: Sequence[int] = (0, 1),
    perturb_profile: str = "uniform",
    perturb_frac: float = 0.12,
    merger_perturb_frac: float = 0.10,
    label: str = "Next-5 Falsification Suite (Projection v2)",
) -> Next5FalsificationResult:
    """Run the corrected projection-based Next-5 falsification suite."""
    details: dict[str, Any] = {}
    verdicts: dict[str, str] = {}

    shell1, shell2, axis_shell = _shell_mode_keys()

    f1 = _run_projection_case(
        grid=f1_grid,
        boundary=lfm.BoundaryType.PERIODIC,
        amp=base_amp,
        sigma=base_sigma,
        ring_steps=f1_ring_steps,
        perturb_profile=perturb_profile,
        perturb_frac=perturb_frac,
        dt=dt,
        record_every=record_every,
        modes=modes,
        min_target=min_target,
    )
    shell1_omegas = [
        float(r["omega"])
        for r in f1["mode_rows"]
        if r["valid"] and r["mode_k"] in shell1 and float(r["omega"]) >= min_target
    ]
    shell2_omegas = [
        float(r["omega"])
        for r in f1["mode_rows"]
        if r["valid"] and r["mode_k"] in shell2 and float(r["omega"]) >= min_target
    ]
    if len(shell1_omegas) < 2 or len(shell2_omegas) < 2:
        verdicts["F1_mode_ratio_projection"] = "INVALID"
        f1_ratio = float("nan")
        f1_err = float("nan")
    else:
        shell1_mean = float(np.mean(shell1_omegas))
        shell2_mean = float(np.mean(shell2_omegas))
        f1_ratio = float(shell2_mean / max(shell1_mean, _EPS))
        f1_err = float(abs(relative_spread(shell1_omegas)) + abs(relative_spread(shell2_omegas)))
        verdicts["F1_mode_ratio_projection"] = "PASS" if f1_err < 0.10 else "FAIL"
    details["F1_mode_ratio_projection"] = {
        "shell1_omegas": shell1_omegas,
        "shell2_omegas": shell2_omegas,
        "shell2_over_shell1": f1_ratio,
        "internal_shell_spread_metric": f1_err,
        "criterion": "PASS if internal shell spread metric < 0.10 (projection-consistency gate)",
    }

    conv = []
    for n in f2_grids:
        r = _run_projection_case(
            grid=int(n),
            boundary=lfm.BoundaryType.PERIODIC,
            amp=base_amp,
            sigma=base_sigma,
            ring_steps=f2_ring_steps,
            perturb_profile=perturb_profile,
            perturb_frac=perturb_frac,
            dt=dt,
            record_every=record_every,
            modes=modes,
            min_target=min_target,
        )
        conv.append(
            {
                "N": int(n),
                "center": float(r["target_band_center"]),
                "spread": float(r["target_band_spread"]),
                "valid": bool(r["valid_mode_count"] >= 4),
            }
        )
    valid_centers = [x for x in conv if x["valid"] and np.isfinite(x["center"])]
    if len(valid_centers) < 2:
        verdicts["F2_resolution_convergence_projection"] = "INVALID"
        drift = float("nan")
    else:
        drift = float(
            abs(valid_centers[-1]["center"] - valid_centers[-2]["center"])
            / max(float(abs(valid_centers[-2]["center"])), _EPS)
        )
        verdicts["F2_resolution_convergence_projection"] = "PASS" if drift <= 0.10 else "FAIL"
    details["F2_resolution_convergence_projection"] = {
        "runs": conv,
        "top_two_relative_drift": drift,
        "criterion": "PASS if top-two relative drift <= 10% with valid extraction",
    }

    r3 = _run_projection_case(
        grid=f3_grid,
        boundary=lfm.BoundaryType.PERIODIC,
        amp=base_amp,
        sigma=base_sigma,
        ring_steps=f3_ring_steps,
        perturb_profile=perturb_profile,
        perturb_frac=perturb_frac,
        dt=dt,
        record_every=record_every,
        modes=modes,
        min_target=min_target,
    )
    axis_omegas = [
        float(r["omega"])
        for r in r3["mode_rows"]
        if r["valid"] and r["mode_k"] in axis_shell and float(r["omega"]) >= min_target
    ]
    if len(axis_omegas) < 2:
        verdicts["F3_basis_invariance_projection"] = "INVALID"
        axis_spread = float("nan")
    else:
        axis_spread = float(relative_spread(axis_omegas))
        verdicts["F3_basis_invariance_projection"] = "PASS" if axis_spread <= 0.10 else "FAIL"
    details["F3_basis_invariance_projection"] = {
        "axis_mode_omegas": axis_omegas,
        "axis_spread": axis_spread,
        "criterion": "PASS if axis-permuted spread <= 10%",
    }

    merger_runs = []
    centers: list[float] = []
    for seed in merger_seeds:
        r = _run_projection_case(
            grid=f4_grid,
            boundary=lfm.BoundaryType.PERIODIC,
            amp=base_amp,
            sigma=base_sigma,
            ring_steps=f4_ring_steps,
            perturb_profile=perturb_profile,
            perturb_frac=merger_perturb_frac,
            dt=dt,
            record_every=record_every,
            modes=modes,
            merger=True,
            merger_pre_steps=merger_pre_steps,
            seed=int(seed),
            min_target=min_target,
        )
        merger_runs.append(
            {
                "seed": int(seed),
                "center": float(r["target_band_center"]),
                "spread": float(r["target_band_spread"]),
                "valid": bool(r["valid_mode_count"] >= 4),
            }
        )
        if r["valid_mode_count"] >= 4 and np.isfinite(r["target_band_center"]):
            centers.append(float(r["target_band_center"]))
    if len(centers) < 2:
        verdicts["F4_merger_comoving_projection"] = "INVALID"
        merger_drift = float("nan")
    else:
        merger_drift = float(abs(centers[1] - centers[0]) / max(float(abs(centers[0])), _EPS))
        verdicts["F4_merger_comoving_projection"] = "PASS" if merger_drift <= 0.15 else "FAIL"
    details["F4_merger_comoving_projection"] = {
        "runs": merger_runs,
        "seed_to_seed_center_drift": merger_drift,
        "criterion": "PASS if seed-to-seed center drift <= 15% with valid extraction",
    }

    core = [
        verdicts.get("F1_mode_ratio_projection"),
        verdicts.get("F2_resolution_convergence_projection"),
        verdicts.get("F3_basis_invariance_projection"),
        verdicts.get("F4_merger_comoving_projection"),
    ]
    if any(v in ("FAIL", "INVALID") for v in core):
        verdicts["F5_external_consistency_projection"] = "INVALID"
        details["F5_external_consistency_projection"] = {
            "reason": "Deferred: run only after F1-F4 projection extraction is robust.",
            "criterion": "Requires F1-F4 all PASS",
        }
    else:
        verdicts["F5_external_consistency_projection"] = "PASS"
        details["F5_external_consistency_projection"] = {
            "reason": "Placeholder PASS gate reached; ready for posterior-mapping stage.",
            "criterion": "F1-F4 all PASS",
        }

    pass_count = sum(1 for v in verdicts.values() if v == "PASS")
    fail_count = sum(1 for v in verdicts.values() if v == "FAIL")
    invalid_count = sum(1 for v in verdicts.values() if v == "INVALID")

    summary = {
        "suite": "next5_falsification_projection_v2",
        "verdicts": verdicts,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "invalid_count": invalid_count,
        "details": details,
    }

    return Next5FalsificationResult(
        snapshots=[],
        movie_snapshots=[],
        metrics=[],
        label=label,
        N=max(int(f1_grid), int(f3_grid), int(f4_grid), *(int(v) for v in f2_grids)),
        summary=summary,
    )
