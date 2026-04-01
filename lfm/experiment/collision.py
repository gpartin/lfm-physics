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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lfm.experiment.common import (
    ExperimentConfig,
    ExperimentResult,
    build_sim,
    gpu_snapshot_loop,
    midplane_slice,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

    import lfm as _lfm_t

__all__ = [
    "collision",
    "CollisionResult",
]

# ── Default constants ──────────────────────────────────────────────────────
_CHI0: float = 19.0
_DT: float = 0.005  # small dt for accurate phase-gradient tracking
_DEFAULT_SPEED: float = 0.10  # within Nyquist; max safe ~0.13c for χ₀=19
_DEFAULT_AMPLITUDE: float = 3.0  # shallow chi-wells → solitons actually move
_SIGMA_FRAC: float = 0.04  # soliton width as fraction of N
_MIN_SIGMA: float = 5.0  # floor: sig<5 → Peierls-Nabarro pinning
_SEPARATION_FRAC: float = 0.65  # particles start near edges, fly inward
_BOUNDARY_FRAC: float = 0.10  # absorbing sponge layer fraction


# ── Geometry ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CollisionGeometry:
    """Pre-computed positions for one collision experiment."""

    N: int
    axis: int  # collision axis (0=x, 1=y, 2=z)
    pos_a: tuple[int, ...]  # particle A centre
    pos_b: tuple[int, ...]  # particle B centre
    separation: int  # centre-to-centre distance (cells)
    sigma: float  # Gaussian soliton width (cells)
    amplitude: float  # soliton peak amplitude
    speed: float  # approach speed per particle (units of c)
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
    sigma = max(_MIN_SIGMA, N * _SIGMA_FRAC)

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
        geometry: _CollisionGeometry,
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

        Uses peak central-zone energy (at collision) vs late energy.
        A value near 1.0 means near-complete annihilation.
        """
        if not self.snapshots:
            return 0.0

        N = self.N
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

        energies = [_central_energy(s) for s in self.snapshots]
        e_peak = max(energies) if energies else 0.0

        # Compare peak vs last 10% of snapshots
        n = len(self.snapshots)
        late = energies[max(0, n - n // 10) :]
        e_late = float(np.mean(late)) if late else e_peak

        if e_peak <= 0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - e_late / e_peak)))

    # ── Outputs ────────────────────────────────────────────────────────

    def plot(
        self,
        *,
        figsize: tuple[float, float] = (16, 10),
        title: str | None = None,
    ) -> Figure:
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
        gs = GridSpec(
            2,
            max(n_panels, 3),
            figure=fig,
            hspace=0.35,
            wspace=0.35,
            top=0.92,
            bottom=0.08,
            left=0.06,
            right=0.97,
        )

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
            0.1,
            0.5,
            "\n".join(info_lines),
            transform=ax_info.transAxes,
            fontsize=11,
            family="monospace",
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
    lambda_self: float = 0.0,
    poisson_only: bool = False,
    verbose: bool = False,
) -> _lfm_t.Simulation:
    """Create and populate the simulation for a collision.

    Uses eigenmode relaxation + phase-gradient boost for physically
    correct initial momentum (solitons actually move).

    Algorithm
    ---------
    1. Relax eigenmode ONCE at grid centre via Poisson-relaxation cycling
       (or skip relaxation and use raw Gaussian + Poisson if poisson_only=True)
    2. Roll the converged fields to particle A and B positions
    3. Apply charge phase (matter θ=0, antimatter θ=π)
    4. Boost each with ``boost_fields()`` (complex phase gradient)
    5. Superpose Ψ fields linearly; Poisson-solve combined χ
    6. Inject into Simulation via field setters (bypasses place_soliton)
    """
    import warnings

    from lfm.constants import KAPPA
    from lfm.particles import get_particle
    from lfm.particles.solver import boost_fields, relax_eigenmode

    part_a = get_particle(particle_a_name)
    part_b = get_particle(particle_b_name)

    # Field level: at least COMPLEX for phase-gradient boost
    fl_int = max(part_a.field_level, part_b.field_level, 1)
    fl_map = {0: "real", 1: "complex", 2: "color"}

    # ── Step 1: Build soliton template ─────────────────────────────
    center = geo.N // 2
    if poisson_only:
        # Skip eigenmode relaxation — use raw Gaussian with one-shot
        # Poisson chi (gives deep chi_min ≈ 14-16 for amp=6 at N=64,
        # rather than the shallow 18.865 that the eigenmode converges to).
        if verbose:
            print("  Poisson-only init (no eigenmode relaxation)...", flush=True)
        coords = np.arange(geo.N, dtype=np.float64) - center
        gz, gy, gx = np.meshgrid(coords, coords, coords, indexing="ij")
        r2 = (gx**2 + gy**2 + gz**2).astype(np.float64)
        E_template = (geo.amplitude * np.exp(-r2 / (2.0 * geo.sigma**2))).astype(np.float32)
        kx = np.fft.fftfreq(geo.N) * 2.0 * np.pi
        KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
        K2 = (KX**2 + KY**2 + KZ**2).astype(np.float64)
        K2[0, 0, 0] = 1.0
        e2 = E_template.astype(np.float64) ** 2
        dchi_hat = -KAPPA * np.fft.fftn(e2) / K2
        dchi_hat[0, 0, 0] = 0.0
        chi_template = (geo.chi0 + np.fft.ifftn(dchi_hat).real).astype(np.float32)
        np.clip(chi_template, 0.01, None, out=chi_template)
        dchi_template = chi_template - np.float32(geo.chi0)
        # Use chi_min as approximate eigenvalue (wave frequency inside well)
        eigenvalue = float(max(chi_template.min(), 1.0))
        if verbose:
            print(
                f"  Poisson-only ready: chi_min={chi_template.min():.4f}  "
                f"omega_approx={eigenvalue:.4f}  E_sum={float(e2.sum()):.1f}",
                flush=True,
            )
    else:
        # ── Eigenmode relaxation (default) ─────────────────────────
        if verbose:
            print("  Relaxing eigenmode...", flush=True)
        sol = relax_eigenmode(
            N=geo.N,
            amplitude=geo.amplitude,
            sigma=geo.sigma,
            chi0=geo.chi0,
            verbose=verbose,
        )
        if not sol.converged:
            warnings.warn(
                "Eigenmode relaxation did not converge "
                f"(chi_min={sol.chi_min:.3f}). Using best result.",
                stacklevel=2,
            )
        if verbose:
            print(
                f"  Eigenmode ready: chi_min={sol.chi_min:.3f}  omega={sol.eigenvalue:.4f}",
                flush=True,
            )
        E_template = sol.psi_r
        dchi_template = sol.chi - np.float32(geo.chi0)
        eigenvalue = sol.eigenvalue

    # ── Step 2: Roll soliton template to each particle position ────
    center = geo.N // 2
    E_template = E_template  # already defined above
    dchi_template = dchi_template  # already defined above

    def _roll_to(arr: np.ndarray, pos: tuple) -> np.ndarray:
        out = arr
        for ax in range(3):
            shift = int(pos[ax]) - center
            if shift != 0:
                out = np.roll(out, shift, axis=ax)
        return out

    E_a = _roll_to(E_template, geo.pos_a)
    chi_a = np.float32(geo.chi0) + _roll_to(dchi_template, geo.pos_a)

    E_b = _roll_to(E_template, geo.pos_b)
    chi_b = np.float32(geo.chi0) + _roll_to(dchi_template, geo.pos_b)

    # ── Step 3: Boost each particle ────────────────────────────────
    phase_a = float(getattr(part_a, "phase", 0.0))
    phase_b = float(getattr(part_b, "phase", 0.0))

    vel_a = [0.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]
    vel_a[geo.axis] = +geo.speed
    vel_b[geo.axis] = -geo.speed

    vel_a_t: tuple[float, float, float] = (vel_a[0], vel_a[1], vel_a[2])
    vel_b_t: tuple[float, float, float] = (vel_b[0], vel_b[1], vel_b[2])

    pr_a, pi_a, prp_a, pip_a, _ = boost_fields(
        E_a,
        chi_a,
        vel_a_t,
        dt=_DT,
        omega=eigenvalue,
        chi0=geo.chi0,
    )
    pr_b, pi_b, prp_b, pip_b, _ = boost_fields(
        E_b,
        chi_b,
        vel_b_t,
        dt=_DT,
        omega=eigenvalue,
        chi0=geo.chi0,
    )

    # ── Step 4: Apply charge phase (matter/antimatter) ─────────────
    def _apply_phase(pr, pi, phase):
        if abs(phase) < 1e-10:
            return pr, pi
        c, s = math.cos(phase), math.sin(phase)
        return (pr * c - pi * s).astype(np.float32), (pr * s + pi * c).astype(np.float32)

    pr_a, pi_a = _apply_phase(pr_a, pi_a, phase_a)
    prp_a, pip_a = _apply_phase(prp_a, pip_a, phase_a)
    pr_b, pi_b = _apply_phase(pr_b, pi_b, phase_b)
    prp_b, pip_b = _apply_phase(prp_b, pip_b, phase_b)

    # ── Step 5: Superpose Ψ fields ────────────────────────────────
    psi_r_curr = pr_a + pr_b
    psi_i_curr = pi_a + pi_b
    psi_r_prev = prp_a + prp_b
    psi_i_prev = pip_a + pip_b

    # ── Step 6: Poisson-solve combined χ ───────────────────────────
    kappa = KAPPA
    kx = np.fft.fftfreq(geo.N) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # avoid DC division

    def _poisson_chi(pr, pi):
        e2 = pr.astype(np.float64) ** 2 + pi.astype(np.float64) ** 2
        dchi_hat = -kappa * np.fft.fftn(e2) / K2
        dchi_hat[0, 0, 0] = 0.0
        chi = (geo.chi0 + np.fft.ifftn(dchi_hat).real).astype(np.float32)
        np.clip(chi, 0.1, geo.chi0, out=chi)
        return chi

    chi_curr = _poisson_chi(psi_r_curr, psi_i_curr)
    chi_prev = _poisson_chi(psi_r_prev, psi_i_prev)

    # ── Step 7: Create simulation and inject fields ────────────────
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
        lambda_self=lambda_self,
    )
    sim = build_sim(cfg)

    # Inject pre-computed fields (bypasses place_soliton entirely)
    sim.set_psi_real(psi_r_curr)
    sim.set_psi_imag(psi_i_curr)
    sim.set_psi_real_prev(psi_r_prev)
    sim.set_psi_imag_prev(psi_i_prev)
    sim._evolver.set_chi_current(chi_curr)
    sim._evolver.set_chi_prev(chi_prev)

    # Mark as fully prepared so auto-equilibrate is skipped
    sim._equilibrated = True

    return sim


def _run_physics(
    geo: _CollisionGeometry,
    particle_a: str,
    particle_b: str,
    *,
    verbose: bool,
    animate: bool = False,
    lambda_self: float = 0.0,
    poisson_only: bool = False,
    step_callback: Callable[[_lfm_t.Simulation, int], None] | None = None,
) -> tuple[list[dict], list[dict], float, list[dict]]:
    """Run physics + optional movie capture in a single pass."""
    sim = _build_collision_sim(
        geo,
        particle_a,
        particle_b,
        lambda_self=lambda_self,
        poisson_only=poisson_only,
        verbose=verbose,
    )

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
        step_callback=step_callback,
    )

    return snaps, metrics, initial_energy, movie_snaps


def _run_movie_pass(
    geo: _CollisionGeometry,
    particle_a: str,
    particle_b: str,
    lambda_self: float = 0.0,
) -> list[dict]:
    """Run a dense snapshot pass for the 3-D movie."""
    sim = _build_collision_sim(geo, particle_a, particle_b, lambda_self=lambda_self)

    # Capture initial frame
    frames: list[dict] = [{"step": 0, "energy_density": sim.energy_density.copy()}]

    snaps = sim.run_with_snapshots(
        steps=geo.movie_steps,
        snapshot_every=geo.movie_snap_every,
        fields=["energy_density"],
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
    lambda_self: float = 0.0,
    poisson_only: bool = False,
    step_callback: Callable[[_lfm_t.Simulation, int], None] | None = None,
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
    lambda_self : float
        Mexican-hat self-interaction coefficient λ_H for GOV-02.
        0.0 (default) = gravity-only.  Set to ``lfm.LAMBDA_H`` (≈ 4/31)
        to enable the Higgs potential V(χ) = λ_H(χ²−χ₀²)² — chi rings at
        ω_H ≈ 19.30 after a collision crushes it below χ₀.
    step_callback : callable or None
        Optional ``(sim, step) -> None`` called inside the evolution loop at
        every step.  Use this for dense per-step diagnostics (e.g. chi at the
        collision centre for Higgs-ringing FFT) without re-running the sim.
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

    if poisson_only and verbose:
        print(
            "  poisson_only=True: skipping eigenmode relaxation "
            "(raw Gaussian + one-shot Poisson chi — deep wells, fast setup)",
            flush=True,
        )

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
        geo,
        particle_a,
        particle_b,
        verbose=verbose,
        animate=animate,
        lambda_self=lambda_self,
        poisson_only=poisson_only,
        step_callback=step_callback,
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
