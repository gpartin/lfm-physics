"""
High-level particle collision experiment for the LFM framework.
================================================================

Smash two particles together on a 3-D lattice and watch what happens.
All geometry (placement, boost, snapshot cadence) is auto-computed from
the grid size and particle species.

The canonical showcase collision is **proton + antiproton annihilation**:
opposite-phase solitons approach, interfere destructively at contact,
and release their bound energy as outward-propagating radiation — all
from GOV-01 + GOV-02 alone.

Quickstart
----------
>>> from lfm.experiment import collision
>>> r = collision(N=128)                                     # p + p̄
>>> r = collision(N=256, speed=0.1)                          # publication
>>> r = collision("electron", "positron", N=64)              # e⁻ + e⁺
>>> r = collision("proton", "proton", N=128)                 # p + p
>>> r.save("pp_annihilation", directory="outputs/")
>>> r.plot()

Results
-------
``result.snapshots``           — field snapshots (energy_density, chi)
``result.movie_snapshots``     — dense psi_real snapshots for 3-D movie
``result.metrics``             — lightweight scalar time series
``result.annihilation_frac``   — fraction of initial energy radiated away
``result.chi_min_history``     — chi minimum over time
``result.plot()``              — matplotlib summary figure
``result.save("stem")``        — write PNG + MP4 + NPZ
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from lfm.experiment.common import (
    ExperimentConfig,
    ExperimentResult,
    build_sim,
    gpu_snapshot_loop,
    midplane_slice,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    import lfm as _lfm_t

__all__ = [
    "collision",
    "CollisionResult",
]

# ── Default constants ──────────────────────────────────────────────────────
_CHI0: float = 19.0
_DT: float = 0.02
_DEFAULT_SPEED: float = 0.30      # fast head-on collision, visually dramatic
_DEFAULT_AMPLITUDE: float = 8.0   # deep chi-wells, well above structure threshold
_SIGMA_FRAC: float = 0.04        # soliton width as fraction of N
_SEPARATION_FRAC: float = 0.80   # start near opposite walls for dramatic collision
_BOUNDARY_FRAC: float = 0.12     # sponge layer fraction


# ── Geometry ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _CollisionGeometry:
    """Pre-computed positions for one collision experiment."""

    N: int
    axis: int               # collision axis (0=x, 1=y, 2=z)
    pos_a: tuple[int, ...]  # particle A centre
    pos_b: tuple[int, ...]  # particle B centre
    separation: int         # centre-to-centre distance (cells)
    sigma: float            # Gaussian soliton width (cells)
    amplitude: float        # soliton peak amplitude
    speed: float            # approach speed per particle (units of c)
    chi0: float
    total_steps: int
    snap_every: int
    movie_steps: int
    movie_snap_every: int
    metrics_every: int


def _compute_geometry(
    N: int,
    *,
    axis: int = 2,
    speed: float = _DEFAULT_SPEED,
    amplitude: float = _DEFAULT_AMPLITUDE,
    chi0: float = _CHI0,
) -> _CollisionGeometry:
    """Auto-compute collision geometry from grid size."""
    mid = N // 2
    sep = max(8, int(N * _SEPARATION_FRAC))
    sigma = max(3.0, N * _SIGMA_FRAC)

    # Position both particles along collision axis, centred in transverse
    pos_a = [mid] * 3
    pos_b = [mid] * 3
    pos_a[axis] = mid - sep // 2
    pos_b[axis] = mid + sep // 2

    # Steps: enough for particles to travel separation + 50% post-collision
    # observation with some buffer
    travel_cells = sep
    steps_per_cell = 1.0 / max(speed * _DT, 1e-6)  # steps to cross 1 cell
    meet_steps = int(travel_cells * steps_per_cell / 2)  # each travels half
    post_collision = int(meet_steps * 1.0)  # watch aftermath equally long
    total_steps = meet_steps + post_collision

    # Snapshot cadence: ~40 physics snapshots (plot uses only 4;
    # at 256³ each snap is 2×67 MB so 40×134 MB ≈ 5.4 GB — safe)
    snap_every = max(1, total_steps // 40)

    # Movie: ~80 dense frames for smooth 4 s animation at 20 fps
    movie_steps = total_steps
    movie_snap_every = max(1, total_steps // 80)

    # Metrics: lightweight, frequent
    metrics_every = max(1, total_steps // 500)

    return _CollisionGeometry(
        N=N,
        axis=axis,
        pos_a=tuple(pos_a),
        pos_b=tuple(pos_b),
        separation=sep,
        sigma=sigma,
        amplitude=amplitude,
        speed=speed,
        chi0=chi0,
        total_steps=total_steps,
        snap_every=snap_every,
        movie_steps=movie_steps,
        movie_snap_every=movie_snap_every,
        metrics_every=metrics_every,
    )


# ── Result class ───────────────────────────────────────────────────────────


class CollisionResult(ExperimentResult):
    """Result of a :func:`collision` experiment run.

    Extends :class:`~lfm.experiment.common.ExperimentResult` with
    collision-specific derived physics.

    Attributes
    ----------
    geometry : _CollisionGeometry
        All computed positions and step counts.
    particle_a_name, particle_b_name : str
        Particle species names.
    initial_energy : float
        Total |Ψ|² energy at step 0.
    """

    def __init__(
        self,
        snapshots: list[dict],
        movie_snapshots: list[dict],
        metrics: list[dict],
        label: str,
        N: int,
        geometry: "_CollisionGeometry",
        particle_a_name: str,
        particle_b_name: str,
        initial_energy: float,
    ) -> None:
        super().__init__(
            snapshots=snapshots,
            movie_snapshots=movie_snapshots,
            metrics=metrics,
            label=label,
            N=N,
        )
        self.geometry = geometry
        self.particle_a_name = particle_a_name
        self.particle_b_name = particle_b_name
        self.initial_energy = initial_energy

    # ── Derived physics ────────────────────────────────────────────────

    @property
    def chi_min_history(self) -> list[float]:
        """Chi minimum at each metrics sample."""
        return [m["chi_min"] for m in self.metrics]

    @property
    def energy_history(self) -> list[float]:
        """Total energy at each metrics sample."""
        return [m["energy_total"] for m in self.metrics]

    @property
    def annihilation_fraction(self) -> float:
        """Fraction of initial localized energy that has been radiated.

        Computed from early vs late energy in the central region.
        A value near 1.0 means near-complete annihilation.
        """
        if not self.snapshots:
            return 0.0

        N = self.N
        geo = self.geometry
        # Define "collision zone" as central 1/4 of the grid
        lo = N // 4
        hi = 3 * N // 4

        def _central_energy(snap: dict) -> float:
            ed = snap.get("energy_density")
            if ed is None:
                return 0.0
            try:
                import cupy
                if isinstance(ed, cupy.ndarray):
                    ed = cupy.asnumpy(ed)
            except ImportError:
                pass
            return float(ed[lo:hi, lo:hi, lo:hi].sum())

        # Compare first 10% and last 10% of snapshots
        n = len(self.snapshots)
        early = self.snapshots[: max(1, n // 10)]
        late = self.snapshots[max(0, n - n // 10):]

        e_early = np.mean([_central_energy(s) for s in early])
        e_late = np.mean([_central_energy(s) for s in late])

        if e_early <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - e_late / e_early))

    # ── Outputs ────────────────────────────────────────────────────────

    def plot(
        self,
        *,
        figsize: tuple[float, float] = (16, 10),
        title: str | None = None,
    ) -> "Figure":
        """Return a summary figure with 4 panels.

        1. Energy density slices at key moments (before / during / after)
        2. Chi minimum over time
        3. Total energy over time
        4. Radial energy profile at final snapshot
        """
        import matplotlib.pyplot as plt

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle(title or self.label, fontsize=14, fontweight="bold")

        N = self.N
        geo = self.geometry

        # Pick 4 representative snapshots: early, pre-collision, collision, late
        n = len(self.snapshots)
        if n >= 4:
            indices = [0, n // 4, n // 2, n - 1]
        elif n >= 2:
            indices = [0, n - 1]
        else:
            indices = list(range(n))

        n_panels = len(indices)
        gs = GridSpec(2, max(n_panels, 3), figure=fig,
                      hspace=0.35, wspace=0.35,
                      top=0.92, bottom=0.08, left=0.06, right=0.97)

        # ── Row 1: Energy density slices along collision axis ──────────
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(gs[0, i])
            snap = self.snapshots[idx]
            ed = snap.get("energy_density")
            if ed is None:
                continue
            plane = midplane_slice(ed, geo.axis)
            ax.imshow(
                plane.T,
                origin="lower",
                cmap="inferno",
                aspect="equal",
            )
            step = snap.get("step", idx)
            ax.set_title(f"step {step}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("|Ψ|² (collision axis)")

        # ── Row 2: three equal-width panels ───────────────────────────
        # Map 3 bottom panels across the full width using column spans
        ncols = gs.ncols
        span = ncols / 3  # fractional column width per bottom panel

        def _bottom_ax(panel_idx: int):
            c0 = round(panel_idx * span)
            c1 = round((panel_idx + 1) * span)
            return fig.add_subplot(gs[1, c0:c1])

        # ── Row 2, panel 1: chi_min ───────────────────────────────────
        if self.metrics:
            steps_m = [m["step"] for m in self.metrics]

            ax_chi = _bottom_ax(0)
            ax_chi.plot(steps_m, self.chi_min_history, color="#3b82f6", lw=1.5)
            ax_chi.set_xlabel("Step")
            ax_chi.set_ylabel("χ_min")
            ax_chi.set_title("Chi well depth", fontsize=10)
            ax_chi.axhline(geo.chi0, color="gray", ls="--", lw=0.8, label=f"χ₀={geo.chi0}")
            ax_chi.legend(fontsize=8)

            # ── Row 2, panel 2: total energy ──────────────────────────
            ax_e = _bottom_ax(1)
            ax_e.plot(steps_m, self.energy_history, color="#22c55e", lw=1.5)
            ax_e.set_xlabel("Step")
            ax_e.set_ylabel("Total |Ψ|²")
            ax_e.set_title("Energy conservation", fontsize=10)

        # ── Row 2, panel 3: annihilation metric ──────────────────────
        ax_info = _bottom_ax(2)
        ax_info.axis("off")
        info_lines = [
            f"Particles: {self.particle_a_name} + {self.particle_b_name}",
            f"Grid: {N}³",
            f"Speed: {geo.speed:.3f} c",
            f"Separation: {geo.separation} cells",
            f"Total steps: {geo.total_steps:,}",
            f"Annihilation: {self.annihilation_fraction:.1%}",
        ]
        if self.chi_min_history:
            info_lines.append(f"χ_min final: {self.chi_min_history[-1]:.2f}")
        ax_info.text(
            0.1, 0.5, "\n".join(info_lines),
            transform=ax_info.transAxes,
            fontsize=11, family="monospace",
            verticalalignment="center",
        )
        ax_info.set_title("Experiment summary", fontsize=10)

        return fig

    def save(
        self,
        stem: str,
        *,
        directory: str | Path | None = None,
        dpi: int = 150,
        save_movie: bool = True,
        save_snapshots_npz: bool = True,
    ) -> dict[str, Path]:
        """Write all outputs to *directory*.

        Returns
        -------
        dict[str, Path]
            Maps output type to path written.
        """
        import matplotlib.pyplot as plt
        from lfm.viz.collision import animate_collision_3d

        out = Path(directory) if directory else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        # ── Summary PNG ────────────────────────────────────────────────
        fig = self.plot()
        p = out / f"{stem}.png"
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written["summary"] = p

        # ── Snapshots NPZ ──────────────────────────────────────────────
        if save_snapshots_npz and self.snapshots:
            p_s = out / f"{stem}_snapshots.npz"
            self.save_snapshots_npz(p_s)
            written["snapshots"] = p_s

        # ── 3-D movie MP4 ──────────────────────────────────────────────
        if save_movie and self.movie_snapshots:
            p_mp4 = out / f"{stem}_3d_movie.mp4"
            try:
                animate_collision_3d(
                    self.movie_snapshots,
                    collision_axis=self.geometry.axis,
                    pos_a=self.geometry.pos_a,
                    pos_b=self.geometry.pos_b,
                    field="energy_density",
                    colormap="inferno",
                    fps=20,
                    max_frames=150,
                    title=self.label,
                    save_path=str(p_mp4),
                )
                written["movie"] = p_mp4
            except Exception as exc:
                print(f"    (3-D movie skipped: {exc})")

        return written


# ── Physics runs ───────────────────────────────────────────────────────────


def _build_collision_sim(
    geo: _CollisionGeometry,
    particle_a_name: str,
    particle_b_name: str,
) -> "_lfm_t.Simulation":
    """Create and populate the simulation for a collision."""
    import lfm
    from lfm.particles import get_particle

    part_a = get_particle(particle_a_name)
    part_b = get_particle(particle_b_name)

    # Determine field level from particles (at least COMPLEX for phase)
    fl_int = max(part_a.field_level, part_b.field_level, 1)
    fl_map = {0: "real", 1: "complex", 2: "color"}

    cfg = ExperimentConfig(
        N=geo.N,
        chi0=geo.chi0,
        dt=_DT,
        total_steps=geo.total_steps,
        snap_every=geo.snap_every,
        boundary_type="frozen",
        boundary_fraction=0.0,
        field_level=fl_map.get(fl_int, "complex"),
        evolve_chi=True,
    )
    sim = build_sim(cfg)

    # Phase: antimatter has phase=π, matter has phase=0
    phase_a = float(getattr(part_a, "phase", 0.0))
    phase_b = float(getattr(part_b, "phase", 0.0))

    # Velocity vectors: A moves +axis, B moves -axis → head-on
    vel_a = [0.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]
    vel_a[geo.axis] = +geo.speed
    vel_b[geo.axis] = -geo.speed

    sim.place_soliton(
        position=geo.pos_a,
        amplitude=geo.amplitude,
        sigma=geo.sigma,
        phase=phase_a,
        velocity=tuple(vel_a),
    )
    sim.place_soliton(
        position=geo.pos_b,
        amplitude=geo.amplitude,
        sigma=geo.sigma,
        phase=phase_b,
        velocity=tuple(vel_b),
    )

    # No need to call sim.equilibrate() manually — the framework
    # auto-equilibrates on first evolution if solitons were placed.

    return sim


def _run_physics(
    geo: _CollisionGeometry,
    particle_a: str,
    particle_b: str,
    *,
    verbose: bool,
    animate: bool = False,
) -> tuple[list[dict], list[dict], float, list[dict]]:
    """Run physics + optional movie capture in a single pass."""
    sim = _build_collision_sim(geo, particle_a, particle_b)

    # Record initial energy before any evolution
    try:
        import cupy
        xp = cupy if isinstance(sim.energy_density, cupy.ndarray) else np
    except ImportError:
        xp = np
    initial_energy = float(xp.sum(sim.energy_density))

    snaps, metrics, movie_snaps = gpu_snapshot_loop(
        sim,
        total_steps=geo.total_steps,
        snap_every=geo.snap_every,
        fields=["energy_density", "chi"],
        movie_every=geo.movie_snap_every if animate else None,
        movie_fields=["energy_density", "chi"],
        verbose=verbose,
        label=f"{particle_a}+{particle_b}",
        evolve_chi=True,
        metrics_every=geo.metrics_every,
    )

    return snaps, metrics, initial_energy, movie_snaps


def _run_movie_pass(
    geo: _CollisionGeometry,
    particle_a: str,
    particle_b: str,
) -> list[dict]:
    """Run a dense snapshot pass for the 3-D movie."""
    sim = _build_collision_sim(geo, particle_a, particle_b)

    # Capture initial frame
    frames: list[dict] = [{"step": 0, "psi_real": sim.psi_real.copy()}]

    snaps = sim.run_with_snapshots(
        steps=geo.movie_steps,
        snapshot_every=geo.movie_snap_every,
        fields=["psi_real"],
    )
    frames.extend(snaps)
    return frames


# ── Public API ─────────────────────────────────────────────────────────────


def collision(
    particle_a: str = "proton",
    particle_b: str = "antiproton",
    N: int = 128,
    *,
    speed: float = _DEFAULT_SPEED,
    amplitude: float = _DEFAULT_AMPLITUDE,
    axis: int = 2,
    chi0: float = _CHI0,
    animate: bool = True,
    label: str | None = None,
    verbose: bool = True,
) -> CollisionResult:
    """Run a complete particle collision experiment.

    Parameters
    ----------
    particle_a, particle_b : str
        Particle names from the catalog (e.g. ``"proton"``, ``"antiproton"``,
        ``"electron"``, ``"positron"``).
    N : int
        Grid size (N × N × N lattice).
        64 = fast preview, 128 = good quality, 256 = publication.
    speed : float
        Approach speed of each particle (units of c).  Both particles start
        at this speed, moving toward each other.  Must be in (0, 0.5].
    amplitude : float
        Soliton peak amplitude.  Higher = deeper χ-wells = stronger gravity.
    axis : int
        Collision axis (0=x, 1=y, 2=z).
    chi0 : float
        Background χ value (default 19).
    animate : bool
        If True, run a second dense-snapshot pass for the 3-D movie.
    label : str or None
        Human-readable label.  Auto-generated if None.
    verbose : bool
        Print progress.

    Returns
    -------
    CollisionResult
        Result with ``.snapshots``, ``.metrics``, ``.annihilation_fraction``,
        ``.plot()``, ``.save()``.

    Examples
    --------
    >>> from lfm.experiment import collision
    >>> r = collision(N=64)
    >>> r.plot()
    >>> r.save("pp_annihilation", directory="outputs/collision/")

    >>> # Electron-positron at higher speed
    >>> r2 = collision("electron", "positron", N=128, speed=0.12)
    >>> print(f"Annihilation: {r2.annihilation_fraction:.1%}")
    """
    if not 0 < speed <= 0.5:
        raise ValueError(f"speed must be in (0, 0.5], got {speed}")

    geo = _compute_geometry(N, axis=axis, speed=speed, amplitude=amplitude, chi0=chi0)

    if label is None:
        label = f"{particle_a} + {particle_b} collision (N={N}, v={speed:.3f}c)"

    if verbose:
        print(
            f"\n{'─' * 60}\n"
            f"Running: {label}\n"
            f"  Grid {N}³  separation={geo.separation}  σ={geo.sigma:.1f}  "
            f"amp={geo.amplitude}  v={speed:.3f}c\n"
            f"  Axis {geo.axis}  steps={geo.total_steps:,}  "
            f"snap_every={geo.snap_every}  metrics_every={geo.metrics_every}",
            flush=True,
        )

    # ── Single-pass: physics + movie capture ───────────────────────────
    snaps, metrics, initial_energy, movie_snaps = _run_physics(
        geo, particle_a, particle_b, verbose=verbose, animate=animate,
    )

    if verbose and metrics:
        chi_min_final = metrics[-1]["chi_min"]
        e_total_final = metrics[-1]["energy_total"]
        n_movie = len(movie_snaps)
        print(
            f"  done.  χ_min={chi_min_final:.2f}  "
            f"E_total={e_total_final:.1f}  "
            f"snapshots={len(snaps)}  movie_frames={n_movie}",
            flush=True,
        )

    return CollisionResult(
        snapshots=snaps,
        movie_snapshots=movie_snaps,
        metrics=metrics,
        label=label,
        N=N,
        geometry=geo,
        particle_a_name=particle_a,
        particle_b_name=particle_b,
        initial_energy=initial_energy,
    )
