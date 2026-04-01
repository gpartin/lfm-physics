"""
High-level double-slit experiment for the LFM framework.
=========================================================

Runs any variant of the Young / double-slit experiment with a single call.
All geometry (source, barrier, detector placement) is auto-computed from the
grid size and regime (near-field vs far-field).

Quickstart
----------
>>> from lfm.experiment import double_slit
>>> r = double_slit(N=64)                                  # V1: no detector
>>> r = double_slit(N=64, n_slits=1)                       # V2: single slit
>>> r = double_slit(N=64, which_path=0.5)                  # V3: partial
>>> r = double_slit(N=64, which_path=1.0)                  # V4: full which-path
>>> r = double_slit(N=64, mode="packet")                   # V5: wave packet
>>> r = double_slit(N=64, far_field=True)                  # V6: Young's stripes
>>> r = double_slit(N=64, far_field=True, which_path=0.5)  # V7: FF partial
>>> r = double_slit(N=64, far_field=True, which_path=1.0)  # V8: FF full

Results
-------
``result.pattern``         — (N, N) accumulated interference pattern
``result.fringe_spacing``  — fringe spacing in cells (λD/d, from geometry)
``result.fresnel_number``  — Fresnel number N_F = d²/(λD)
``result.visibility``      — fringe contrast (Imax−Imin)/(Imax+Imin)
``result.plot()``          — matplotlib figure (heatmap + 1-D profile)
``result.click_pattern()`` — Monte-Carlo single-particle click events
``result.save("stem")``    — write pattern PNG, clicks PNG, movie MP4, NPZ
``DoubleSlit.compare([r1, r2, ...])``  — side-by-side profile comparison
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from lfm.experiment.barrier import Slit
from lfm.experiment.dispersion import dispersion

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    import lfm as _lfm_t
    from lfm.experiment.detector import DetectorScreen

__all__ = [
    "double_slit",
    "DoubleSlit",
    # ── Public geometry / movie API ────────────────────────────────────────
    # Any script that runs its own physics backend (e.g. a GPU/cloud runner)
    # should use these instead of the private underscore functions.  The
    # invariant is: physics snapshots → fringe MEASUREMENT only; the 3-D
    # movie always comes from make_movie_snapshots().  This separation is
    # enforced by the API — the two functions are intentionally distinct.
    "ExperimentGeometry",
    "make_geometry",
    "make_movie_snapshots",
]

# ── Internal constants ─────────────────────────────────────────────────────
_BOUNDARY_FRAC: float = 0.15  # absorbing sponge layer (fraction of N/2)
_CHI0_SIM: float = 1.0  # effective chi0: fast waves, short transit time
_KAPPA_SIM: float = 1e-6  # near-static chi (pure wave-optics limit)
_DT: float = 0.02  # leapfrog timestep (CFL-safe for chi0=1, omega≤2)
_AMPLITUDE: float = 3.0  # CW source injection amplitude
_BOOST: float = 10.0  # CW boost factor — fast steady-state buildup
_BARRIER_HEIGHT: float = 51.0  # chi inside solid barrier (>> omega → evanescent)
_MOVIE_BARRIER_H: float = 6.0  # semi-transparent barrier for movie visibility
_MAX_STEPS: int = 64_000  # global step cap (~25 min per variant on RTX 4060)


# ── Geometry ───────────────────────────────────────────────────────────────


@dataclass
class _Geometry:
    """All pre-computed positions and step counts for one experiment."""

    N: int
    omega: float
    chi0: float
    source_z: int
    barrier_z: int
    detector_z: int
    half_sep: int
    slit_width: int
    sigma_src: float
    movie_sigma: float
    movie_source_z: int
    movie_slit_width: int
    movie_steps: int
    v_group: float
    k_z: float
    wavelength: float
    steps: int
    record_every: int
    snap_every: int

    @property
    def D(self) -> int:
        """Barrier-to-detector distance in cells."""
        return self.detector_z - self.barrier_z

    @property
    def d(self) -> int:
        """Slit centre-to-centre separation in cells."""
        return 2 * self.half_sep

    @property
    def fresnel_number(self) -> float:
        """Fresnel number N_F = d² / (λ D)."""
        if self.D <= 0 or self.wavelength <= 0:
            return float("inf")
        return self.d**2 / (self.wavelength * self.D)

    @property
    def fringe_spacing(self) -> float:
        """Theoretical first-order fringe spacing λD/d in cells."""
        if self.d <= 0:
            return float("inf")
        return self.wavelength * self.D / self.d


def _compute_geometry(N: int, *, far_field: bool = False) -> _Geometry:
    """Internal implementation — see :func:`make_geometry`."""
    return make_geometry(N, far_field=far_field)


def make_geometry(N: int, *, far_field: bool = False) -> _Geometry:
    """Compute all experiment geometry from *N* and the regime flag.

    This is the **only** function that should configure a double-slit run.
    All positions (source, barrier, detector), slit dimensions, step counts,
    and movie parameters are derived from N — the caller sets nothing else.

    Parameters
    ----------
    N : int
        Grid size (N × N × N).  Minimum 128 — smaller grids do not give
        enough room for the interference pattern to develop cleanly.
    far_field : bool
        ``True`` → far-field (Young's stripes, Fraunhofer regime).
        ``False`` → near-field (Fresnel regime).

    Returns
    -------
    ExperimentGeometry
        All pre-computed positions and step counts.  Pass this directly to
        :func:`make_movie_snapshots` — no manual geometry is ever needed.
    """
    if N < 128:
        raise ValueError(
            f"Grid size N={N} is too small for a meaningful double-slit demonstration.\n"
            "The minimum is N=128 — the interference pattern needs at least 128 cells\n"
            "in each direction for clean fringes and a good-looking 3-D movie.\n"
            "Use --grid 128 (quick) or --grid 256 (publication quality)."
        )

    # ── original body follows unchanged ────────────────────────────────────
    """* Source: ``active_r² − (src_z − N/2)² ≥ sigma_src²``
      (source Gaussian fits inside the safe sphere).
    * Detector: ``active_r² − (det_z − N/2)² ≥ (2 × half_sep)²``
      (full fringe pattern fits inside the safe sphere).
    """
    bf = _BOUNDARY_FRAC
    active_r = (1.0 - bf) * (N / 2.0)
    z_lo = int(math.ceil(N / 2.0 - active_r)) + 1
    z_hi = int(math.floor(N / 2.0 + active_r)) - 1
    chi0 = _CHI0_SIM
    dt = _DT

    if not far_field:
        # ── Near-field (Fresnel) geometry ───────────────────────────────
        omega = 1.5
        disp = dispersion(omega=omega, chi0=chi0, dt=dt)
        v_group = disp.v_group
        k_z = disp.k_z
        wavelength = 2.0 * math.pi / k_z

        half_sep = N // 8
        slit_width = max(2, min(N // 16, 6))

        # Source: Pythagorean constraint — source Gaussian must fit inside sphere
        sigma_src = max(6.0, float(half_sep) * 2.0)
        src_margin2 = active_r**2 - sigma_src**2
        z_src_min = (
            int(math.ceil(N / 2.0 - math.sqrt(max(1.0, src_margin2)))) + 1
            if src_margin2 > 0
            else N // 4
        )
        source_z = max(z_src_min, z_lo)
        barrier_z = N // 2

        # Detector: ensure transverse safe radius covers 2 × half_sep
        max_dz = math.sqrt(max(0.0, active_r**2 - (2.0 * half_sep) ** 2))
        detector_z = max(barrier_z + 4, min(z_hi, int(math.floor(N / 2.0 + max_dz)) - 1))

    else:
        # ── Far-field (Fraunhofer / Young's stripes) geometry ───────────
        omega = 2.0
        disp = dispersion(omega=omega, chi0=chi0, dt=dt)
        v_group = disp.v_group
        k_z = disp.k_z
        wavelength = 2.0 * math.pi / k_z

        half_sep = max(4, N // 40)
        slit_width = max(3, N // 64)

        sigma_src = float(max(int(N * 0.12), 3 * half_sep))
        src_margin2 = active_r**2 - sigma_src**2
        z_src_min = (
            int(math.ceil(N / 2.0 - math.sqrt(max(1.0, src_margin2)))) + 1
            if src_margin2 > 0
            else N // 4
        )
        source_z = max(z_src_min, z_lo)
        barrier_z = max(
            source_z + 4,
            min(N // 2 + max(8, N // 32), z_hi - 20),
        )

        # Detector: solve quadratic so 3 fringe-spacings fit inside safe sphere
        n_stripes = 3.0
        alpha = n_stripes * wavelength / (2.0 * half_sep)
        C = float(barrier_z - N // 2)
        qa = alpha**2 + 1.0
        qb = 2.0 * C
        qc = C**2 - active_r**2
        disc = max(0.0, qb**2 - 4.0 * qa * qc)
        D_opt = max(8, int((-qb + math.sqrt(disc)) / (2.0 * qa)))
        detector_z = max(barrier_z + 4, min(z_hi, barrier_z + D_opt))

    # ── Step counts ────────────────────────────────────────────────────
    vg_per_step = v_group * dt
    transit = int((detector_z - source_z) / max(vg_per_step, 1e-6))
    if not far_field:
        steps = min(max(N * 250, transit * 3), _MAX_STEPS)
    else:
        steps = min(max(transit * 4, N * 300), _MAX_STEPS)

    record_every = max(1, N // 32)
    _max_frames = 4000
    if steps // record_every > _max_frames:
        record_every = max(record_every, steps // _max_frames)
    snap_every = max(80, steps // 20)

    # ── Movie parameters ───────────────────────────────────────────────
    movie_steps = max(N * 25, int(transit * 1.6))
    movie_slit_width = max(4, slit_width * 2)

    if far_field:
        # FF movie: use a narrow compact blob at the same source_z
        movie_sigma = min(20.0, sigma_src * 0.25)
        movie_source_z = source_z  # source_z already safe for smaller sigma
    else:
        # NF movie: capped sigma so the blob looks like a localised particle
        movie_sigma = max(4.0, min(float(half_sep), 12.0))
        ms_margin2 = active_r**2 - movie_sigma**2
        ms_min = int(math.ceil(N / 2.0 - math.sqrt(ms_margin2))) + 1 if ms_margin2 > 0 else N // 4
        movie_source_z = max(ms_min, z_lo)

    return _Geometry(
        N=N,
        omega=omega,
        chi0=chi0,
        source_z=source_z,
        barrier_z=barrier_z,
        detector_z=detector_z,
        half_sep=half_sep,
        slit_width=slit_width,
        sigma_src=sigma_src,
        movie_sigma=float(movie_sigma),
        movie_source_z=movie_source_z,
        movie_slit_width=movie_slit_width,
        movie_steps=movie_steps,
        v_group=v_group,
        k_z=k_z,
        wavelength=wavelength,
        steps=steps,
        record_every=record_every,
        snap_every=snap_every,
    )


# ── Result class ───────────────────────────────────────────────────────────


class DoubleSlit:
    """Result of a :func:`double_slit` experiment run.

    Attributes
    ----------
    pattern : ndarray, shape (N, N)
        Time-integrated interference pattern on the detector screen.
    snapshots : list[dict]
        Field snapshots from the physics run (for offline replay).
    movie_snapshots : list[dict]
        Dense snapshots from the wave-packet movie pass (empty if
        ``animate=False`` was given to :func:`double_slit`).
    geometry : _Geometry
        All computed positions and step counts.
    slits : list[Slit]
        Slit configuration that was used.
    label : str
        Human-readable description of this variant.
    """

    def __init__(
        self,
        pattern: np.ndarray,
        snapshots: list[dict],
        movie_snapshots: list[dict],
        geometry: _Geometry,
        slits: list[Slit],
        label: str,
        _screen: DetectorScreen,
    ) -> None:
        self.pattern = pattern
        self.snapshots = snapshots
        self.movie_snapshots = movie_snapshots
        self.geometry = geometry
        self.slits = slits
        self.label = label
        self._screen = _screen

    # ── Derived physics ────────────────────────────────────────────────

    @property
    def fringe_spacing(self) -> float:
        """Theoretical fringe spacing in cells (λD/d)."""
        return self.geometry.fringe_spacing

    @property
    def fresnel_number(self) -> float:
        """Fresnel number N_F = d² / (λD)."""
        return self.geometry.fresnel_number

    @property
    def visibility(self) -> float:
        """Fringe visibility V = (Imax − Imin) / (Imax + Imin) along the central row."""
        row = self.pattern[self.geometry.N // 2, :].astype(float)
        I_max, I_min = float(row.max()), float(row.min())
        denom = I_max + I_min
        return (I_max - I_min) / denom if denom > 0 else 0.0

    # ── Outputs ────────────────────────────────────────────────────────

    def click_pattern(self, n_particles: int = 10_000, seed: int = 42) -> np.ndarray:
        """Monte-Carlo single-particle click events sampled from :attr:`pattern`.

        Returns an ``(N, N)`` integer array of hit counts.
        """
        return self._screen.click_pattern(n_particles=n_particles, seed=seed)

    def plot(
        self,
        *,
        colormap: str = "inferno",
        show_profile: bool = True,
        figsize: tuple[float, float] = (12, 5),
        title: str | None = None,
    ) -> Figure:
        """Return a matplotlib figure with the interference-pattern heatmap
        and 1-D transverse profile."""
        import lfm.viz as viz

        return viz.plot_interference_pattern(
            self.pattern,
            title=title or self.label,
            colormap=colormap,
            show_profile=show_profile,
            figsize=figsize,
        )

    def save(
        self,
        stem: str,
        *,
        directory: str | Path | None = None,
        dpi: int = 150,
        save_clicks: bool = True,
        save_movie: bool = True,
        save_snapshots_npz: bool = True,
    ) -> dict[str, Path]:
        """Write all outputs for this variant to *directory*.

        Parameters
        ----------
        stem : str
            Filename stem (e.g. ``"double_slit_01_no_detector"``).
        directory : str or Path or None
            Output directory.  Defaults to current working directory.
        dpi : int
            PNG resolution.
        save_clicks : bool
            Also save a Monte-Carlo click-pattern PNG.
        save_movie : bool
            Also save a 3-D perspective MP4 movie (requires
            ``animate=True`` in the original :func:`double_slit` call).
        save_snapshots_npz : bool
            Also save physics-run snapshots as a compressed NPZ.

        Returns
        -------
        dict[str, Path]
            Maps output type (``"pattern"``, ``"clicks"``, ``"movie"``,
            ``"snapshots"``) to the path written.
        """
        import matplotlib.pyplot as plt

        import lfm.viz as viz
        from lfm.io import save_snapshots as _save_snaps
        from lfm.viz.quantum import animate_double_slit_3d

        out = Path(directory) if directory else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        geo = self.geometry

        # ── Pattern PNG ────────────────────────────────────────────────
        fig = self.plot()
        p = out / f"{stem}.png"
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written["pattern"] = p

        # ── Clicks PNG ─────────────────────────────────────────────────
        if save_clicks:
            clicks = self.click_pattern()
            fig_c = viz.plot_interference_pattern(
                clicks.astype(np.float32),
                title=f"{self.label} — clicks (N=10 000)",
                colormap="hot",
                show_profile=True,
                figsize=(12, 5),
            )
            p_c = out / f"{stem}_clicks.png"
            fig_c.savefig(str(p_c), dpi=dpi, bbox_inches="tight")
            plt.close(fig_c)
            written["clicks"] = p_c

        # ── Snapshots NPZ ──────────────────────────────────────────────
        if save_snapshots_npz and self.snapshots:
            p_s = out / f"{stem}_snapshots.npz"
            _save_snaps(self.snapshots, p_s)
            written["snapshots"] = p_s

        # ── 3-D movie MP4 ──────────────────────────────────────────────
        if save_movie and self.movie_snapshots:
            p_mp4 = out / f"{stem}_3d_movie.mp4"
            slit_centers = [s.center for s in self.slits]
            try:
                animate_double_slit_3d(
                    self.movie_snapshots,
                    barrier_axis=2,
                    barrier_position=geo.barrier_z,
                    detector_position=geo.detector_z,
                    source_position=geo.movie_source_z,
                    slit_centers=slit_centers,
                    slit_width=geo.movie_slit_width,
                    field="psi_real",
                    colormap="RdBu_r",
                    fps=15,
                    max_frames=100,
                    title=self.label,
                    save_path=str(p_mp4),
                )
                written["movie"] = p_mp4
            except Exception as exc:
                print(f"    (3-D movie skipped: {exc})")

        return written

    # ── Class-level comparison ─────────────────────────────────────────

    @staticmethod
    def compare(
        results: list[DoubleSlit],
        *,
        save_path: str | Path | None = None,
        dpi: int = 150,
    ) -> Figure:
        """Side-by-side 1-D profile comparison of multiple results.

        Parameters
        ----------
        results : list[DoubleSlit]
            Results to compare (any subset of V1–V8).
        save_path : str or Path or None
            If given, save the figure to this path before returning.

        Returns
        -------
        Figure
            The matplotlib figure (caller is responsible for
            ``plt.close()``).
        """
        import matplotlib.pyplot as plt

        _COLORS = [
            "#3b82f6",
            "#22c55e",
            "#f97316",
            "#ef4444",
            "#ec4899",
            "#a855f7",
            "#06b6d4",
            "#f43f5e",
        ]
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]
        fig.suptitle("LFM Double-Slit: Variant Comparison", fontsize=14, fontweight="bold")
        for ax, res, color in zip(axes, results, _COLORS, strict=False):
            N = res.geometry.N
            profile = res.pattern.sum(axis=0).astype(float)
            mx = profile.max()
            if mx > 0:
                profile /= mx
            ax.plot(profile, np.arange(len(profile)), color=color, lw=1.8)
            ax.set_title(res.label, fontsize=9)
            ax.set_xlabel("Normalised Intensity")
            ax.set_xlim(-0.05, 1.15)
            ax.set_ylim(0, N - 1)
            ax.axvline(0, color="gray", lw=0.5)
        axes[0].set_ylabel("Transverse position (cells)")
        fig.tight_layout()
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig


# ── Internal simulation helpers ────────────────────────────────────────────


def _build_sim(geo: _Geometry) -> _lfm_t.Simulation:
    import lfm

    cfg = lfm.SimulationConfig(
        grid_size=geo.N,
        field_level=lfm.FieldLevel.COMPLEX,
        boundary_type=lfm.BoundaryType.ABSORBING,
        boundary_fraction=_BOUNDARY_FRAC,
        chi0=geo.chi0,
        kappa=_KAPPA_SIM,
    )
    return lfm.Simulation(cfg)


def _run_cw_physics(
    geo: _Geometry,
    slits: list[Slit],
    *,
    verbose: bool,
) -> tuple[DetectorScreen, list[dict]]:
    """CW physics run with ContinuousSource for maximum-contrast fringes.

    Uses the GPU-direct evolver loop (avoids the N³/step PCIe cost of
    ``run_with_snapshots`` with a callback).
    """
    sim = _build_sim(geo)

    # CW source — injects sinusoidal plane-wave into the source plane every step
    source = sim.add_source(
        axis=2,
        position=geo.source_z,  # absolute cell index (>= 1 → treated as cells)
        omega=geo.omega,
        amplitude=_AMPLITUDE,
        envelope_sigma=geo.sigma_src,  # absolute cells (>= 1)
        boost=_BOOST,
    )
    # transit_steps: how many leapfrog steps the wave spends inside the slit
    # region.  Used to calibrate the per-step absorption γ so that after one
    # full transit the remaining amplitude equals (1 − detector_strength).
    _transit_slit = max(1, int(geo.slit_width / max(geo.v_group * _DT, 1e-9)))
    barrier = sim.place_barrier(
        axis=2,
        position=geo.barrier_z,
        height=_BARRIER_HEIGHT,
        thickness=2,
        slits=slits,
        absorb=True,
        transit_steps=_transit_slit,
    )
    screen = sim.add_detector(axis=2, position=geo.detector_z)

    # GPU-direct evolver loop: zero N³ copies during stepping
    # (run_with_snapshots with a step_callback forces report_interval=1
    #  which triggers a full N³ GPU→CPU transfer every step — 4 TB at N=256)
    evolver = sim._evolver
    snaps: list[dict] = []
    t0 = time.perf_counter()

    for step in range(1, geo.steps + 1):
        evolver.evolve(1)
        source.step_callback(sim, step)
        barrier.step_callback(sim, step)
        if step % geo.record_every == 0:
            screen.record()  # GPU fast-path: copies only N²×4 bytes
        if step % geo.snap_every == 0:
            snaps.append({"step": evolver.step, "energy_density": sim.energy_density.copy()})
        if verbose and step % max(1, geo.steps // 10) == 0:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            print(
                f"    step {step}/{geo.steps}  "
                f"({rate:.0f} steps/s, ~{(geo.steps - step) / rate:.0f}s left)",
                flush=True,
            )

    return screen, snaps


def _run_packet_physics(
    geo: _Geometry,
    slits: list[Slit],
    *,
    verbose: bool,
) -> tuple[DetectorScreen, list[dict]]:
    """Wave-packet variant: single soliton boosted toward the barrier."""

    sim = _build_sim(geo)
    mid = geo.N // 2

    v_z = 0.9
    sim.place_soliton(
        position=(mid, mid, geo.source_z),
        amplitude=_AMPLITUDE * 5.0,
        sigma=max(float(geo.half_sep) * 1.5, 5.0),
        velocity=(0.0, 0.0, v_z),
    )
    barrier = sim.place_barrier(
        axis=2,
        position=geo.barrier_z,
        height=_BARRIER_HEIGHT,
        thickness=2,
        slits=slits,
        absorb=True,
    )
    screen = sim.add_detector(axis=2, position=geo.detector_z)

    pkt_k = _CHI0_SIM * v_z
    pkt_vg = pkt_k / math.sqrt(pkt_k**2 + _CHI0_SIM**2)
    travel = (geo.detector_z - geo.source_z) / max(pkt_vg, 0.1)
    total_steps = max(1500, int(travel / _DT * 1.5))
    max_frames = min(400, max(50, int(1e9 / (geo.N**3 * 4))))
    snap_every = max(2, total_steps // max_frames)

    if verbose:
        print(f"    packet run: {total_steps} steps", flush=True)

    def _cb(s: _lfm_t.Simulation, step: int) -> None:
        barrier.step_callback(s, step)
        screen.record()

    snaps = sim.run_with_snapshots(
        steps=total_steps,
        snapshot_every=snap_every,
        fields=["energy_density"],
        step_callback=_cb,
    )
    return screen, snaps


def _run_movie(
    geo: _Geometry,
    slits: list[Slit],
    *,
    mode: str,
) -> list[dict]:
    """Internal alias — see :func:`make_movie_snapshots`."""
    return make_movie_snapshots(geo, slits, mode=mode)


def make_movie_snapshots(
    geo: _Geometry,
    slits: list[Slit],
    *,
    mode: str = "cw",
) -> list[dict]:
    """Run a short dedicated wave-packet pass and return 3-D movie frames.

    This is **always** how the 3-D movie should be generated — regardless of
    which backend (CPU / local GPU / cloud) ran the physics measurement.

    The physics run drives a ``ContinuousSource`` (plane wave) to build
    steady-state fringes and measure visibility.  That is correct for
    *measurement* but wrong for *visualisation*: every frame would show the
    wave filling the entire box.  This function instead fires a compact
    Gaussian wave packet that travels through the slits and looks like a
    real particle — the same approach used by ``23_double_slit.py``.

    A localised packet only activates ~1 000–5 000 scatter points regardless
    of N, so the renderer never has density problems at any supported grid
    size.

    Parameters
    ----------
    geo : ExperimentGeometry
        Geometry returned by :func:`make_geometry` — must match the grid used
        for the physics run.
    slits : list[Slit]
        Slit configuration that was used in the physics run.
    mode : "cw" or "packet"
        Source mode — ``"cw"`` (default) gives the cleanest localised packet;
        ``"packet"`` uses a heavier soliton for the wave-packet variant.

    Returns
    -------
    list[dict]
        Dense snapshots suitable for ``animate_double_slit_3d``.
    """
    # ── original _run_movie body follows ─────────────────────────────────
    import lfm

    sim = _build_sim(geo)
    mid = geo.N // 2

    if mode == "packet":
        # Same soliton as the physics run but stronger for visibility
        v_z = 0.9
        sim.place_soliton(
            position=(mid, mid, geo.source_z),
            amplitude=_AMPLITUDE * 5.0,
            sigma=max(float(geo.half_sep) * 1.5, 5.0),
            velocity=(0.0, 0.0, v_z),
        )
    else:
        lfm.create_particle(
            "electron",
            sim=sim,
            position=(mid, mid, geo.movie_source_z),
            velocity=(0.0, 0.0, geo.v_group),
            use_eigenmode=False,
            chi0=geo.chi0,
            sigma=geo.movie_sigma,
            amplitude=_AMPLITUDE * 3.0,
        )

    movie_slits = [Slit(center=s.center, width=max(s.width, geo.movie_slit_width)) for s in slits]
    barrier = sim.place_barrier(
        axis=2,
        position=geo.barrier_z,
        height=_MOVIE_BARRIER_H,
        thickness=2,
        slits=movie_slits,
        absorb=True,
    )

    def _cb(s: _lfm_t.Simulation, step: int) -> None:
        barrier.step_callback(s, step)

    max_frames = min(200, max(40, int(1_000_000_000 // (geo.N**3 * 4))))
    snap_every = max(1, geo.movie_steps // max_frames)

    # Capture initial state (step 0) so the animation starts at the source
    initial: dict = {"step": 0, "psi_real": sim.psi_real.copy()}
    rest = sim.run_with_snapshots(
        steps=geo.movie_steps,
        snapshot_every=snap_every,
        fields=["psi_real"],
        step_callback=_cb,
        evolve_chi=False,  # κ=1e-6 → χ frozen (static medium); avoids PCIe overhead
    )
    return [initial, *rest]


# ── Public name aliases ────────────────────────────────────────────────────
#: Public alias for the geometry dataclass — use this name in type hints.
ExperimentGeometry = _Geometry


# ── Public API ─────────────────────────────────────────────────────────────


def double_slit(
    N: int = 64,
    *,
    n_slits: int = 2,
    which_path: float = 0.0,
    which_path_slit: int = 0,
    far_field: bool = False,
    mode: Literal["cw", "packet"] = "cw",
    animate: bool = True,
    label: str | None = None,
    verbose: bool = True,
) -> DoubleSlit:
    """Run a complete double-slit experiment and return a result object.

    Parameters
    ----------
    N : int
        Grid size (N × N × N lattice).
        32 = fast preview, 64 = default, 128 = HD, 256 = publication quality.
    n_slits : int
        1 = single slit (diffraction envelope only), 2 = double slit.
    which_path : float
        Which-path detector strength, in [0, 1].

        * 0.0 — no detector, clean fringes
        * 0.5 — partial information, fringes weakened
        * 1.0 — full which-path, fringes destroyed

        Applied to slit index *which_path_slit*.
    which_path_slit : int
        Index of the slit that receives the detector (default 0).
    far_field : bool
        ``True`` → far-field Fraunhofer regime: classic Young's parallel
        stripes (small Fresnel number).  ``False`` → Fresnel near-field.
    mode : "cw" or "packet"
        ``"cw"`` drives a :class:`ContinuousSource` for maximum-contrast
        steady-state fringes.  ``"packet"`` fires a single soliton
        wave-packet (what people picture as "one electron at a time").
    animate : bool
        If ``True``, also run a short dense movie-capture pass and store
        the result in ``result.movie_snapshots``.  Adds roughly 10 % of
        the total runtime.
    label : str or None
        Human-readable label stored in ``result.label``.  Auto-generated
        if ``None``.
    verbose : bool
        Print step-rate progress to stdout.

    Returns
    -------
    DoubleSlit
        Result with ``.pattern``, ``.fringe_spacing``, ``.fresnel_number``,
        ``.visibility``, ``.plot()``, ``.click_pattern()``, ``.save()``.

    Examples
    --------
    >>> from lfm.experiment import double_slit, DoubleSlit
    >>> v1 = double_slit(N=64)
    >>> v1.plot()
    >>> v7 = double_slit(N=256, far_field=True, which_path=0.5)
    >>> v7.save("ff_partial", directory="outputs/")
    >>> DoubleSlit.compare([v1, v7])
    """
    geo = _compute_geometry(N, far_field=far_field)
    mid = geo.N // 2

    # ── Build slit list ────────────────────────────────────────────────
    slits: list[Slit] = []
    if n_slits == 1:
        slits = [Slit(center=mid, width=geo.slit_width)]
    else:
        for i in range(n_slits):
            sign = -1 if i == 0 else 1
            is_detector = (which_path > 0.0) and (i == which_path_slit)
            slits.append(
                Slit(
                    center=mid + sign * geo.half_sep,
                    width=geo.slit_width,
                    detector=is_detector,
                    detector_strength=float(which_path) if is_detector else 1.0,
                )
            )

    # ── Auto-generate label ────────────────────────────────────────────
    if label is None:
        regime = "far-field" if far_field else "near-field"
        if n_slits == 1:
            label = f"Single slit ({regime})"
        elif which_path == 0.0:
            label = f"No detector ({regime})"
        elif which_path < 1.0:
            label = f"Partial detector α={which_path:.1f} ({regime})"
        else:
            label = f"Full which-path ({regime})"
        if mode == "packet":
            label += ", wave packet"

    if verbose:
        d_str = f"d={geo.d}" if n_slits > 1 else "single slit"
        print(
            f"\n{'─' * 60}\n"
            f"Running: {label}\n"
            f"  Grid {N}³  source={geo.source_z}  barrier={geo.barrier_z}  "
            f"detector={geo.detector_z}  {d_str}\n"
            f"  ω={geo.omega}  v_g={geo.v_group:.3f}c  λ={geo.wavelength:.2f}  "
            f"N_F={geo.fresnel_number:.2f}  steps={geo.steps}",
            flush=True,
        )

    # ── Physics run ────────────────────────────────────────────────────
    if mode == "packet":
        screen, snaps = _run_packet_physics(geo, slits, verbose=verbose)
    else:
        screen, snaps = _run_cw_physics(geo, slits, verbose=verbose)

    if verbose:
        print(
            f"  done.  peak={float(screen.pattern.max()):.4f}  "
            f"frames={screen.n_frames}  visibility={_compute_visibility(screen.pattern, N):.3f}",
            flush=True,
        )

    # ── Movie capture ──────────────────────────────────────────────────
    movie_snaps: list[dict] = []
    if animate:
        if verbose:
            print("  Movie capture …", end="", flush=True)
        movie_snaps = _run_movie(geo, slits, mode=mode)
        if verbose:
            print(f" {len(movie_snaps)} frames captured")

    return DoubleSlit(
        pattern=screen.pattern,
        snapshots=snaps,
        movie_snapshots=movie_snaps,
        geometry=geo,
        slits=slits,
        label=label,
        _screen=screen,
    )


def _compute_visibility(pattern: np.ndarray, N: int) -> float:
    row = pattern[N // 2, :].astype(float)
    I_max, I_min = float(row.max()), float(row.min())
    denom = I_max + I_min
    return (I_max - I_min) / denom if denom > 0 else 0.0
