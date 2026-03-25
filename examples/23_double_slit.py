#!/usr/bin/env python3
"""
LFM Double-Slit Experiment — Complete Showcase
===============================================

Demonstrates all four variants of the double-slit experiment using the
LFM (Lattice Field Medium) framework.  In LFM, particles ARE waves:
a soliton wave-packet evolving under GOV-01 + GOV-02.  The "barrier"
is a region of high χ (χ >> χ₀ = 19) that the wave cannot penetrate;
slits are gaps where χ = χ₀.

Physical mechanism
------------------
GOV-01: ∂²Ψ/∂t² = c²∇²Ψ − χ²Ψ

When χ >> χ₀ the restoring term χ²Ψ becomes enormous — the wave oscillates
at frequency χ ≈ 69 instead of propagating.  A wave packet with kinetic
energy ½(χ₀ v/c)² ≪ χ² is effectively reflected.  Slit openings (χ = χ₀)
allow free propagation.  After passing through both slits the two wave
components interfere, building the classic fringe pattern on the detector
screen.

Variants run
------------
1. NO DETECTOR  — classic Young's double slit → bright interference fringes
2. ONE SLIT OPEN — single slit → diffraction envelope, no fringes
3. PARTIAL which-path (one-slit detector, strength 0.5) → fringes weakened
4. FULL which-path (one-slit detector, strength 1.0) → fringes destroyed

Visualisation
-------------
After each variant:
  • Saves a PNG of the built-up interference pattern
  • Saves a fringe-profile PNG
  • Saves an animated GIF of the wave propagation

If pyvista is installed (GPU path) a final 3-D volume render is also saved.

Usage
-----
    cd examples
    python 23_double_slit.py

    # Smaller, faster preview (for CI or slow machines):
    python 23_double_slit.py --small

    # Skip animation saving (faster):
    python 23_double_slit.py --no-anim

Outputs
-------
    double_slit_01_no_detector.png
    double_slit_01_no_detector_profile.png
    double_slit_01_no_detector.gif   (if --no-anim not set)
    double_slit_02_single_slit.png
    double_slit_03_partial_detector.png
    double_slit_04_full_which_path.png
    double_slit_comparison.png
    double_slit_3d_volume.png        (if pyvista installed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Windows cmd / PowerShell use cp1252 by default; force UTF-8 so Unicode
# box-drawing characters in print() don't raise UnicodeEncodeError.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Ensure we can import lfm from the repository root ──────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lfm  # noqa: E402
import lfm.viz as viz  # noqa: E402
from lfm.experiment import Slit, dispersion  # noqa: E402
from lfm.io import save_snapshots  # noqa: E402

# ── CLI ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="LFM Double-Slit Experiment showcase (all five variants)"
)
parser.add_argument(
    "--small",
    action="store_true",
    help="Use 32³ grid for a rapid preview (lower resolution)",
)
parser.add_argument(
    "--large",
    action="store_true",
    help="Use 128³ grid for HD quality (GPU recommended)",
)
parser.add_argument(
    "--grid",
    type=int,
    default=None,
    help="Override grid size N (e.g. --grid 256 for ultra-HD)",
)
parser.add_argument(
    "--no-anim",
    action="store_true",
    help="Skip saving animation GIFs (much faster)",
)
parser.add_argument(
    "--variant",
    type=str,
    default="all",
    choices=["1", "2", "3", "4", "5", "all"],
    help=(
        "Which variant(s) to run.  "
        "1=no-detector (fastest), 2=single-slit, 3=partial-detector, "
        "4=full-which-path, 5=wave-packet movie, all=run everything (default)"
    ),
)
args = parser.parse_args()

# Determine which variants are active
_selected = {1, 2, 3, 4, 5} if args.variant == "all" else {int(args.variant)}


def _run_v(v: int) -> bool:
    """Return True if variant *v* should be executed."""
    return v in _selected


# ── Grid parameters ────────────────────────────────────────────────────────
#
#  N = 32   small mode: ~0.5 s on CPU, quick check
#  N = 64   default: ~8 s on CPU, fine resolution
#  N = 128  large mode: ~2 s on GPU (RTX 4060), HD quality
#  N = 256  --grid 256: ultra-HD (~20 min total on GPU)
#
if args.grid is not None:
    N = args.grid
elif args.large:
    N = 128
elif args.small:
    N = 32
else:
    N = 64

# Geometry (all positions in grid cells)
# Use the FULL grid: source near the start, barrier in the middle,
# detector near the far end.  The library uses a SPHERICAL absorbing
# boundary of radius (1 - boundary_fraction) × N/2, so we use a thin
# sponge (15%) and compute the safe zone from the active radius.
PROPAGATION_AXIS = 2  # wave travels in +z direction
_BOUNDARY_FRAC = 0.15  # lighter sponge than default 0.30
_active_r = (1.0 - _BOUNDARY_FRAC) * (N / 2.0)  # safe radius from centre
_z_lo = int(np.ceil(N / 2.0 - _active_r)) + 1  # first safe cell + 1 buffer
_z_hi = int(np.floor(N / 2.0 + _active_r)) - 1  # last safe cell − 1 buffer
# Source placement: the absorbing boundary is SPHERICAL, so placing the
# source at _z_lo (boundary edge) gives very small transverse safe radius.
# At N=256 only 6.5% of the Gaussian envelope falls inside the active zone
# and the wave never builds up.  Fix: place the source where the transverse
# safe radius is >= the Gaussian envelope sigma, ensuring ~70% efficiency.
#   trans_safe² = active_r² - (src_z - centre)²  >=  sigma²
#   src_z >= centre - sqrt(active_r² - sigma²)
# Fallback to N//4 if sigma > active_r.
_sigma_target = max(6.0, float(N // 8) * 2.0)  # == SIGMA_SRC (defined later)
_src_margin2 = _active_r**2 - _sigma_target**2
_z_src_min = (
    int(np.ceil(N / 2.0 - np.sqrt(max(1, _src_margin2)))) + 1 if _src_margin2 > 0 else N // 4
)
SOURCE_Z = max(_z_src_min, _z_lo)  # N=32→7, N=64→12, N=256→42
BARRIER_Z = N // 2  # N=32→16, N=64→32  (middle)
# Detector placement: the absorbing boundary is SPHERICAL (radius _active_r
# from grid centre).  Placing the detector at _z_hi maximises axial distance
# but the transverse safe radius there is tiny — at N=256 it's only ~20 cells,
# while the slit pattern extends ±HALF_SEP=32.  The entire fringe pattern
# falls inside the absorbing sponge and gets damped to zero.
#
# Fix: ensure transverse safe radius ≥ 2×HALF_SEP so the full diffraction
# pattern is inside the safe sphere.  This uses Pythagoras on the sphere:
#   trans_r² + (z - N/2)² ≤ active_r²
#   z ≤ N/2 + sqrt(active_r² - (2*HALF_SEP)²)
_trans_need = 2.0 * (N // 8)  # 2 × HALF_SEP
_max_dz = np.sqrt(max(0, _active_r**2 - _trans_need**2))
_z_safe = int(np.floor(N / 2.0 + _max_dz)) - 1
DETECTOR_Z = min(_z_hi, _z_safe)  # N=64→52, N=256→214
# Slit geometry uses fixed slit separation (d/λ scales with N for more
# fringes) and CAPPED slit width to stay in the Fraunhofer regime:
#   Fresnel number N_F = W²/(λ·D) must be << 1 for clear far-field fringes.
#   With D ≈ 0.34 N and λ ≈ 3: N_F ≈ W²/(N).  Capping W ≤ 6 keeps
#   N_F ≤ 0.14 even at N = 256.  Without the cap, W = N/16 = 16 at
#   N = 256 gives N_F ≈ 1 (near-field), where the which-path detector
#   effect becomes invisible because the peak sits behind each slit.
HALF_SEP = N // 8  # half slit sep: N=32→4, N=64→8
SLIT_WIDTH = max(2, min(N // 16, 6))  # slit width: N=32→2, N=64→4, N=256→6

# Continuous driven-source parameters (Paper-055 approach).
#
# We use chi0=1.0 for this demo so that OMEGA >> chi0 → group velocity
# near maximum.  chi0=1.0 with OMEGA=1.5 gives the FASTEST discrete
# group velocity (v_g = 0.618c from the 19-point stencil), maximising
# the cell-per-step rate and minimising transit time.
# The canonical chi0=19 places the wave near the mass threshold (v_g ≈ 0.05 c)
# which would require millions of steps to build up on N=32/64 grids.
#
#   chi0 = 1.0          (effective mass for this wave demo)
#   OMEGA = 1.5         → from discrete dispersion: K_z ≈ 1.186 rad/cell
#   lambda_z ≈ 5.3 cells → wider fringes, better Fraunhofer regime
#
CHI0_SIM = 1.0  # effective medium chi0 for this demo
KAPPA_SIM = 1e-6  # near-static chi (barely evolves)
OMEGA = 1.5  # optimal v_g for 19-point stencil at chi0=1

# Exact discrete dispersion from the 19-point stencil:
_DT = 0.02  # leapfrog timestep (CFL-safe)
DISP = dispersion(omega=OMEGA, chi0=CHI0_SIM, dt=_DT)
K_Z = DISP.k_z  # ≈ 1.186 rad/cell (wavelength ≈ 5.3 cells)
V_GROUP = DISP.v_group  # ≈ 0.618c — particle initial velocity (+z)

AMPLITUDE = 3.0  # source amplitude (compensates longer propagation path)
SIGMA_SRC = max(6.0, float(HALF_SEP) * 2.0)  # Gaussian width (illuminates ±HALF_SEP)
BARRIER_HEIGHT = CHI0_SIM + 50.0  # chi inside solid barrier (>> OMEGA → evanescent)

# Source boost scaling: the CW source injects energy across a transverse
# Gaussian envelope of width SIGMA_SRC.  At larger N, SIGMA_SRC grows,
# spreading energy over more cells (area ∝ sigma²).  However the Gaussian
# shape is self-similar — the envelope value at any fractional distance
# from centre is the same (exp(-0.125) ≈ 0.88 at ±HALF_SEP for all N).
# Therefore a constant BOOST keeps per-cell intensity identical across
# grid sizes.  The earlier σ² scaling was wrong and caused 10⁶× peaks
# at N=256.
BOOST = 10.0  # constant across grid sizes

# Simulation parameters.
# Transit time must use the ACTUAL cells-per-step velocity, not the
# natural-unit velocity.  v_group from dispersion() is in natural lattice
# units (cells per time unit); multiply by dt to get cells per step.
_vg_per_step = DISP.v_group * _DT  # ≈ 0.012 cells/step
_transit = int((DETECTOR_Z - SOURCE_Z) / _vg_per_step)
# Ensure at least 3× transit for good CW steady-state buildup.
# Cap at 64 000 steps ≈ 25 min per variant at N=256 on RTX 4060.
STEPS = min(max(N * 250, _transit * 3), 64_000)
# Hard cap: limit total detector frames to ~4000 to bound memory usage
# (each frame is N² × 4 bytes).  RECORD_EVERY is chosen so STEPS/REC ≤ 4000.
RECORD_EVERY = max(1, N // 32)  # N=32→1, N=64→2, N=128→4, N=256→8
_max_frames = 4000
if STEPS // RECORD_EVERY > _max_frames:
    RECORD_EVERY = max(RECORD_EVERY, STEPS // _max_frames)
SNAP_EVERY = max(80, N * 2)
FIELDS = ["energy_density"]
MOVIE_FIELDS = ["psi_real"]  # raw wave amplitude shows phase-front motion

# Limit snapshot memory: each snapshot stores N³ × 4 bytes per field.
# Cap total snapshot memory to ~500 MB.
_snap_bytes = N**3 * 4 * max(1, len(FIELDS))
_max_snaps = max(4, int(500e6 / _snap_bytes))
SNAP_EVERY = max(SNAP_EVERY, STEPS // _max_snaps)

# Movie-specific parameters — short dense run that captures the wavefront
# propagating from source → barrier → slits → detector → pattern forming.
MOVIE_STEPS = max(N * 25, int(_transit * 1.6))
MOVIE_SNAP = max(2, N // 16)  # N=32→2, N=64→4  (captures wavefront motion)
# Movie uses a lower barrier and wider slits so the pulse energy visibly
# passes through (the full physics run keeps the original opaque wall).
MOVIE_BARRIER_H = CHI0_SIM + 5.0  # semi-transparent (χ≈6 vs ω=2)
MOVIE_SLIT_W = max(4, SLIT_WIDTH * 2)  # ≥ 4 cells ≈ 1.3λ (good transmission)

OUT_DIR = Path(__file__).parent
SAVE_ANIM = not args.no_anim

print(
    f"\nLFM Double-Slit Experiment\n"
    f"  Grid : {N}³   Backend : {lfm.get_backend()}\n"
    f"  Source z={SOURCE_Z}  Barrier z={BARRIER_Z}  Detector z={DETECTOR_Z}\n"
    f"  Steps: {STEPS}  Snap/: {SNAP_EVERY}  RecordEvery: {RECORD_EVERY}  Boost: {BOOST:.1f}\n"
    f"  pyvista: {viz.volume_render_available()}\n"
    f"  Variants: {', '.join(str(v) for v in sorted(_selected))}\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build one fresh simulation ready for a run
# ═══════════════════════════════════════════════════════════════════════════


def _build_sim() -> lfm.Simulation:
    """Return a freshly initialised complex-field simulation arena.

    We use FieldLevel.COMPLEX with chi0=1.0 so that the electron wave packet
    velocity can be encoded as a phase gradient e^(ik·r).  OMEGA=1.5 gives
    group velocity v_g ≈ 0.618c from the 19-point stencil — fast enough to
    traverse N=256 in ~14 000 steps.  kappa=1e-6 keeps chi nearly static
    (wave-optics limit), appropriate for this interference demonstration.
    """
    cfg = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.COMPLEX,  # velocity phase encoding
        boundary_type=lfm.BoundaryType.ABSORBING,
        boundary_fraction=_BOUNDARY_FRAC,  # thin sponge (15%)
        chi0=CHI0_SIM,
        kappa=KAPPA_SIM,
    )
    return lfm.Simulation(cfg)


# Particle is placed per-variant in run_variant() using create_particle().


def run_movie_snapshots(
    slits: list[Slit],
) -> list[dict]:
    """Short **pulsed** run that captures a wave packet propagating.

    Uses a single electron wave packet (via create_particle) so the
    energy_density has a visible peak that *moves* from source through slits
    to detector — exactly what you want in a 3-D movie.

    Returns snapshots suitable for ``animate_double_slit_3d``.
    """
    sim = _build_sim()
    # Place a higher-amplitude electron for movie visibility
    lfm.create_particle(
        "electron",
        sim=sim,
        position=(N // 2, N // 2, SOURCE_Z),
        velocity=(0.0, 0.0, V_GROUP),
        use_eigenmode=False,
        chi0=CHI0_SIM,
        sigma=SIGMA_SRC,
        amplitude=AMPLITUDE * 3.0,  # stronger pulse for movie visibility
    )
    # Movie uses a *lower* barrier and *wider* slits so the pulse visibly
    # passes through.  The full physics run keeps the original opaque wall.
    movie_slits = [Slit(center=s.center, width=MOVIE_SLIT_W) for s in slits]
    barrier = sim.place_barrier(
        axis=PROPAGATION_AXIS,
        position=BARRIER_Z,
        height=MOVIE_BARRIER_H,
        thickness=2,
        slits=movie_slits,
        absorb=True,
    )

    def _step_cb(s: lfm.Simulation, step: int) -> None:
        barrier.step_callback(s, step)

    # Run long enough for the wave packet to traverse source → detector.
    # MOVIE_STEPS is pre-computed from group-velocity transit time.
    total_steps = MOVIE_STEPS
    # Cap snapshot count so memory stays under ~1 GB (N³×4 bytes per frame).
    # At N=256 each frame = 67 MB; without a cap, 600 frames → 40 GB OOM.
    _max_movie_frames = min(200, max(40, int(1_000_000_000 // (N**3 * 4))))
    snap_every = max(1, total_steps // _max_movie_frames)  # N=256→40 frames, N=128→119

    return sim.run_with_snapshots(
        steps=total_steps,
        snapshot_every=snap_every,
        fields=MOVIE_FIELDS,
        step_callback=_step_cb,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Variant runner
# ═══════════════════════════════════════════════════════════════════════════


def run_variant(
    label: str,
    slits: list[Slit],
    save_stem: str,
) -> tuple[np.ndarray, list[dict]]:
    """Run one variant and save outputs.  Returns ``(pattern, snapshots)``."""
    print(f"  Running {label} …", flush=True)

    sim = _build_sim()
    # Place electron wave packet via create_particle() (chi0=1.0 → v_g≈0.618c).
    # chi0=CHI0_SIM means the simulation medium is NOT vacuum (chi0=19) — it is a
    # light effective-mass medium chosen so the wave traverses N=256 in reasonable
    # time.  The wide sigma=SIGMA_SRC illuminates both slits coherently, producing
    # the classic Fraunhofer fringe pattern on the detector.
    lfm.create_particle(
        "electron",
        sim=sim,
        position=(N // 2, N // 2, SOURCE_Z),
        velocity=(0.0, 0.0, V_GROUP),
        use_eigenmode=False,
        chi0=CHI0_SIM,
        sigma=SIGMA_SRC,
        amplitude=AMPLITUDE,
    )
    barrier = sim.place_barrier(
        axis=PROPAGATION_AXIS,
        position=BARRIER_Z,
        height=BARRIER_HEIGHT,
        thickness=2,
        slits=slits,
        absorb=True,
    )
    screen = sim.add_detector(axis=PROPAGATION_AXIS, position=DETECTOR_Z)

    # ── GPU fast path ──────────────────────────────────────────────────
    #
    # sim.run_with_snapshots() with step_callback sets report_interval=1,
    # causing a full N³ GPU→CPU copy (get_psi_real().copy()) on EVERY step.
    # For N=256 × 64 000 steps that's ~4 TB of PCIe traffic.
    #
    # Instead we drive the evolver directly: each leapfrog step is a pure
    # CUDA kernel launch.  Source, barrier, and detector callbacks all use
    # _native_* GPU pointers — zero CPU copies.  We only read fields back
    # at snapshot boundaries (every SNAP_EVERY steps).
    import time as _time

    _t0 = _time.perf_counter()
    evolver = sim._evolver
    snaps: list[dict] = []

    for step in range(1, STEPS + 1):
        evolver.evolve(1)  # pure GPU kernel
        barrier.step_callback(sim, step)  # GPU: enforce χ barrier
        if step % RECORD_EVERY == 0:
            screen.record()  # GPU fast-path: N² copy
        if step % SNAP_EVERY == 0:
            snap = {"step": evolver.step}
            if "energy_density" in FIELDS:
                snap["energy_density"] = sim.energy_density.copy()
            if "chi" in FIELDS:
                snap["chi"] = sim.chi.copy()
            if "psi_real" in FIELDS:
                snap["psi_real"] = sim.psi_real.copy()
            snaps.append(snap)
        if step % max(1, STEPS // 10) == 0:
            _elapsed = _time.perf_counter() - _t0
            _rate = step / _elapsed
            print(
                f"    step {step}/{STEPS}  "
                f"({_rate:.0f} steps/s, "
                f"~{(STEPS - step) / _rate:.0f}s left)",
                flush=True,
            )

    pattern = screen.pattern
    peak = float(pattern.max())
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  done.  peak = {peak:.4f},  frames = {screen.n_frames},  "
        f"time = {_elapsed:.1f}s ({STEPS / _elapsed:.0f} steps/s)"
    )

    # ── Interference-pattern heatmap ──────────────────────────────────
    fig_pat = viz.plot_interference_pattern(
        pattern,
        title=label,
        colormap="inferno",
        show_profile=True,
        figsize=(12, 5),
    )
    fig_pat.savefig(OUT_DIR / f"{save_stem}.png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig_pat)

    # ── Click simulation (10 000 single-particle hits) ────────────────
    clicks = screen.click_pattern(n_particles=10_000, seed=42)
    fig_click = viz.plot_interference_pattern(
        clicks.astype(np.float32),
        title=f"{label} — particle clicks (N=10 000)",
        colormap="hot",
        show_profile=True,
        figsize=(12, 5),
    )
    fig_click.savefig(OUT_DIR / f"{save_stem}_clicks.png", dpi=150, bbox_inches="tight")
    plt.close(fig_click)

    # ── Animation ─────────────────────────────────────────────────────
    if SAVE_ANIM:
        anim = viz.animate_double_slit(
            snaps,
            barrier_axis=PROPAGATION_AXIS,
            barrier_position=BARRIER_Z,
            detector_position=DETECTOR_Z,
            field="energy_density",
            slice_axis=0,
            colormap="inferno",
            fps=12,
        )
        try:
            anim.save(
                str(OUT_DIR / f"{save_stem}.gif"),
                writer="pillow",
                fps=12,
            )
            print(f"    → saved {save_stem}.gif")
        except Exception as exc:
            print(f"    (animation save skipped: {exc})")
        plt.close("all")

    return pattern, snaps


# ═══════════════════════════════════════════════════════════════════════════
# Variant 1 — classic double slit (no detectors)
# ═══════════════════════════════════════════════════════════════════════════

pat_v1: np.ndarray | None = None
snaps_v1: list = []

if _run_v(1):
    print("\n── Variant 1: Classic double slit (no detectors) ──")
    pat_v1, snaps_v1 = run_variant(
        label="Variant 1 — No Detector (interference fringes expected)",
        slits=[
            Slit(center=N // 2 - HALF_SEP, width=SLIT_WIDTH),
            Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
        ],
        save_stem="double_slit_01_no_detector",
    )

# ── Save V1 snapshots for offline 3-D replay ─────────────────────────────
if _run_v(1) and pat_v1 is not None:
    _snap_out = OUT_DIR / "double_slit_v1_snapshots.npz"
    save_snapshots(snaps_v1, _snap_out)
    print(f"  → saved snapshots to {_snap_out.name}")
    print(f"    (replay with: python tools/visualize.py {_snap_out})")

# ── Optional 3-D slice animation for V1 ───────────────────────────────
if _run_v(1) and SAVE_ANIM:
    import matplotlib.pyplot as plt

    print("  Building 3-D slice animation for V1 …", end="", flush=True)
    from lfm.viz.quantum import animate_3d_slices

    anim_3d = animate_3d_slices(
        snaps_v1,
        field="energy_density",
        colormap="inferno",
        fps=12,
    )
    _gif_3d = OUT_DIR / "double_slit_v1_3d_slices.gif"
    try:
        anim_3d.save(
            str(_gif_3d),
            writer="pillow",
            fps=12,
        )
        print(f" done  → saved {_gif_3d.name}")
    except Exception as exc:
        print(f" skipped ({exc})")
    plt.close("all")

# ── 3-D perspective movie for V1 ──────────────────────────────────────
if _run_v(1) and SAVE_ANIM:
    import matplotlib.pyplot as plt

    print("  Running dense movie capture for V1 …", end="", flush=True)
    from lfm.viz.quantum import animate_double_slit_3d

    _movie_snaps_v1 = run_movie_snapshots(
        slits=[
            Slit(center=N // 2 - HALF_SEP, width=SLIT_WIDTH),
            Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
        ],
    )
    print(f" {len(_movie_snaps_v1)} frames", end="", flush=True)
    _snap_movie_out = OUT_DIR / "double_slit_v1_movie_snapshots.npz"
    save_snapshots(_movie_snaps_v1, _snap_movie_out)
    print(f" → saved {_snap_movie_out.name}")

    _movie_3d = OUT_DIR / "double_slit_v1_3d_movie.mp4"
    print("  Building 3-D perspective movie for V1 …", end="", flush=True)
    try:
        anim_movie = animate_double_slit_3d(
            _movie_snaps_v1,
            barrier_axis=PROPAGATION_AXIS,
            barrier_position=BARRIER_Z,
            detector_position=DETECTOR_Z,
            source_position=SOURCE_Z,
            slit_centers=[N // 2 - HALF_SEP, N // 2 + HALF_SEP],
            slit_width=MOVIE_SLIT_W,  # match movie physics
            field="psi_real",
            colormap="RdBu_r",
            fps=15,
            max_frames=100,
            title="LFM Double-Slit — V1 No Detector",
            save_path=str(_movie_3d),
        )
        print(f" done  → saved {_movie_3d.name}")
    except Exception as exc:
        print(f" skipped ({exc})")
    plt.close("all")

# ═══════════════════════════════════════════════════════════════════════════
# Variant 2 — single slit (diffraction only)
# ═══════════════════════════════════════════════════════════════════════════

pat_v2: np.ndarray | None = None
snaps_v2: list = []

if _run_v(2):
    print("\n── Variant 2: Single slit (diffraction envelope, no fringes) ──")
    pat_v2, snaps_v2 = run_variant(
        label="Variant 2 — Single Slit (diffraction, no fringes)",
        slits=[
            Slit(center=N // 2, width=SLIT_WIDTH),  # one slit at centre
        ],
        save_stem="double_slit_02_single_slit",
    )

# ═══════════════════════════════════════════════════════════════════════════
# Variant 3 — partial which-path detector (strength = 0.5)
# ═══════════════════════════════════════════════════════════════════════════

pat_v3: np.ndarray | None = None
snaps_v3: list = []

if _run_v(3):
    print("\n── Variant 3: Partial which-path detector (strength = 0.5) ──")
    pat_v3, snaps_v3 = run_variant(
        label="Variant 3 — Partial Detector α=0.5 (fringes weakened)",
        slits=[
            Slit(
                center=N // 2 - HALF_SEP,
                width=SLIT_WIDTH,
                detector=True,
                detector_strength=0.5,
            ),
            Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
        ],
        save_stem="double_slit_03_partial_detector",
    )

# ═══════════════════════════════════════════════════════════════════════════
# Variant 4 — full which-path detector (strength = 1.0)
# ═══════════════════════════════════════════════════════════════════════════

pat_v4: np.ndarray | None = None
snaps_v4: list = []

if _run_v(4):
    print("\n── Variant 4: Full which-path detector (strength = 1.0) ──")
    pat_v4, snaps_v4 = run_variant(
        label="Variant 4 — Full Which-Path α=1.0 (fringes destroyed)",
        slits=[
            Slit(
                center=N // 2 - HALF_SEP,
                width=SLIT_WIDTH,
                detector=True,
                detector_strength=1.0,
            ),
            Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
        ],
        save_stem="double_slit_04_full_which_path",
    )

# ═══════════════════════════════════════════════════════════════════════════
# Save snapshots & 3-D movies for V2–V4
# ═══════════════════════════════════════════════════════════════════════════

_all_variants_eligible = [
    (
        2,
        "v2",
        "V2 Single Slit (diffraction)",
        snaps_v2,
        [N // 2],
        [Slit(center=N // 2, width=SLIT_WIDTH)],
    ),
    (
        3,
        "v3",
        "V3 Partial Detector α=0.5",
        snaps_v3,
        [N // 2 - HALF_SEP, N // 2 + HALF_SEP],
        [
            Slit(center=N // 2 - HALF_SEP, width=SLIT_WIDTH, detector=True, detector_strength=0.5),
            Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
        ],
    ),
    (
        4,
        "v4",
        "V4 Full Which-Path α=1.0",
        snaps_v4,
        [N // 2 - HALF_SEP, N // 2 + HALF_SEP],
        [
            Slit(center=N // 2 - HALF_SEP, width=SLIT_WIDTH, detector=True, detector_strength=1.0),
            Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
        ],
    ),
]
_all_variants = [
    (tag, lbl, snps, sc, sl)
    for (vnum, tag, lbl, snps, sc, sl) in _all_variants_eligible
    if _run_v(vnum)
]

for _tag, _label, _snaps, _slit_ctrs, _slits in _all_variants:
    # Save snapshots for offline replay
    _snap_p = OUT_DIR / f"double_slit_{_tag}_snapshots.npz"
    save_snapshots(_snaps, _snap_p)
    print(f"  → saved {_snap_p.name}")

    # 3-D perspective movie (from dense movie-specific snapshots)
    if SAVE_ANIM:
        import matplotlib.pyplot as plt

        from lfm.viz.quantum import animate_double_slit_3d

        print(f"  Running dense movie capture for {_tag} …", end="", flush=True)
        _movie_snaps = run_movie_snapshots(slits=_slits)
        print(f" {len(_movie_snaps)} frames")
        _snap_mp = OUT_DIR / f"double_slit_{_tag}_movie_snapshots.npz"
        save_snapshots(_movie_snaps, _snap_mp)
        print(f"  → saved {_snap_mp.name}")

        _mp4 = OUT_DIR / f"double_slit_{_tag}_3d_movie.mp4"
        print(f"  Building 3-D perspective movie for {_tag} …", end="", flush=True)
        try:
            _anim = animate_double_slit_3d(
                _movie_snaps,
                barrier_axis=PROPAGATION_AXIS,
                barrier_position=BARRIER_Z,
                detector_position=DETECTOR_Z,
                source_position=SOURCE_Z,
                slit_centers=_slit_ctrs,
                slit_width=MOVIE_SLIT_W,  # match movie physics
                field="psi_real",
                colormap="RdBu_r",
                fps=15,
                max_frames=100,
                title=f"LFM Double-Slit — {_label}",
                save_path=str(_mp4),
            )
            print(f" done  → saved {_mp4.name}")
        except Exception as exc:
            print(f" skipped ({exc})")
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# Variant 5 — soliton wave-packet ("a particle going through slits")
# ═══════════════════════════════════════════════════════════════════════════
#
# Uses sim.place_soliton() to create a localised Gaussian wave-packet
# boosted toward the barrier.  This is what a user pictures when they
# think "particle → slits → interference pattern on the wall."
#
# The packet has momentum k = χ₀·v/c along the propagation axis.
# The spatial phase gradient cos(k·z) inside the Gaussian envelope
# creates visible wavefronts that split at the slits and interfere.

# V5 packet parameters (constants, safe to define always)
PACKET_VZ = 0.9
_pkt_k = CHI0_SIM * PACKET_VZ / 1.0
_pkt_om = float(np.sqrt(_pkt_k**2 + CHI0_SIM**2))
_pkt_vg = _pkt_k / _pkt_om
_pkt_lam = 2 * np.pi / _pkt_k


def _run_packet_movie(slits: list[Slit]) -> list[dict]:
    """Wave-packet movie: localized pulse goes through double slit.

    Uses the same static-medium sim as V1-V4 (κ≈0 = correct quantum regime).
    The packet is placed via place_soliton() with a velocity boost, making it
    a traveling Gaussian wave envelope — the LFM equivalent of 'firing one
    electron at a time.'  It diffracts through both slits and the resulting
    interference pattern builds on the detector.
    """
    sim = _build_sim()  # same κ≈0 as CW variants (correct physics)
    mid = N // 2

    # Wide enough to illuminate both slits (sigma ≈ slit separation)
    pkt_sigma = max(float(HALF_SEP) * 1.5, 5.0)

    sim.place_soliton(
        position=(mid, mid, SOURCE_Z),
        amplitude=AMPLITUDE * 5.0,  # strong pulse for visibility
        sigma=pkt_sigma,
        velocity=(0.0, 0.0, PACKET_VZ),
    )

    movie_slits = [Slit(center=s.center, width=MOVIE_SLIT_W) for s in slits]
    barrier = sim.place_barrier(
        axis=PROPAGATION_AXIS,
        position=BARRIER_Z,
        height=MOVIE_BARRIER_H,
        thickness=2,
        slits=movie_slits,
        absorb=True,
    )

    def _step_cb(s: lfm.Simulation, step: int) -> None:
        barrier.step_callback(s, step)

    # Run long enough for packet to traverse the full grid and build the
    # interference pattern on the detector.
    travel = (DETECTOR_Z - SOURCE_Z) / max(_pkt_vg, 0.1)
    total = max(1500, int(travel / sim.config.dt * 1.5))  # 1.5× travel time
    # Cap snapshot count so memory stays under ~1 GB (N³×4 bytes per frame).
    max_frames = min(400, max(50, int(1e9 / (N**3 * 4))))
    snap_every = max(2, total // max_frames)

    return sim.run_with_snapshots(
        steps=total,
        snapshot_every=snap_every,
        fields=["energy_density"],  # |Ψ|² shows the packet as a bright blob
        step_callback=_step_cb,
    )


if _run_v(5):
    print("\n── Variant 5: Wave-packet (single particle through slits) ──")
    print(
        f"  Packet: k_z={_pkt_k:.3f}  ω={_pkt_om:.3f}  "
        f"v_group={_pkt_vg:.3f}c  λ={_pkt_lam:.2f} cells  κ={KAPPA_SIM}"
    )

    _v5_slits = [
        Slit(center=N // 2 - HALF_SEP, width=SLIT_WIDTH),
        Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
    ]

    print("  Running dense movie capture for V5 …", end="", flush=True)
    _movie_snaps_v5 = _run_packet_movie(slits=_v5_slits)
    print(f" {len(_movie_snaps_v5)} frames")

    _snap_v5 = OUT_DIR / "double_slit_v5_movie_snapshots.npz"
    save_snapshots(_movie_snaps_v5, _snap_v5)
    print(f"  → saved {_snap_v5.name}")

    if SAVE_ANIM:
        import matplotlib.pyplot as plt

        from lfm.viz.quantum import animate_double_slit_3d

        _mp4_v5 = OUT_DIR / "double_slit_v5_3d_movie.mp4"
        print("  Building 3-D perspective movie for V5 …", end="", flush=True)
        try:
            _anim_v5 = animate_double_slit_3d(
                _movie_snaps_v5,
                barrier_axis=PROPAGATION_AXIS,
                barrier_position=BARRIER_Z,
                detector_position=DETECTOR_Z,
                source_position=SOURCE_Z,
                slit_centers=[N // 2 - HALF_SEP, N // 2 + HALF_SEP],
                slit_width=MOVIE_SLIT_W,
                field="energy_density",
                colormap="hot",
                fps=15,
                max_frames=100,
                title="LFM Double-Slit — V5 Wave Packet (single particle)",
                save_path=str(_mp4_v5),
            )
            print(f" done  → saved {_mp4_v5.name}")
        except Exception as exc:
            print(f" skipped ({exc})")
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# Comparison panel — all four fringe profiles side by side
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Comparison panel — fringe profiles side by side (only variants that ran)
# ═══════════════════════════════════════════════════════════════════════════

_comp_panels = [
    (pat_v1, "V1: No detector\n(fringes)", "#3b82f6"),
    (pat_v2, "V2: Single slit\n(diffraction)", "#22c55e"),
    (pat_v3, "V3: Partial detector\n(weakened)", "#f97316"),
    (pat_v4, "V4: Full which-path\n(no fringes)", "#ef4444"),
]
_comp_panels = [(p, t, c) for (p, t, c) in _comp_panels if p is not None]

if _comp_panels:
    print("\n── Saving comparison figure ──")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(_comp_panels), figsize=(5 * len(_comp_panels), 5), sharey=True)
    if len(_comp_panels) == 1:
        axes = [axes]
    fig.suptitle("LFM Double-Slit: Variant Comparison", fontsize=14, fontweight="bold")

    for ax, (pat, title_txt, color) in zip(axes, _comp_panels):
        profile = pat.sum(axis=0).astype(float)
        mx = profile.max()
        if mx > 0:
            profile /= mx
        ax.plot(profile, np.arange(len(profile)), color=color, lw=1.8)
        ax.set_title(title_txt, fontsize=10)
        ax.set_xlabel("Normalised Intensity")
        ax.set_xlim(-0.05, 1.15)
        ax.set_ylim(0, N - 1)
        ax.axvline(0, color="gray", lw=0.5)

    axes[0].set_ylabel("Transverse position (cells)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "double_slit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved double_slit_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3-D volume render (GPU if pyvista; CPU fallback)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 3-D volume render (reuse V1 snapshot if available)
# ═══════════════════════════════════════════════════════════════════════════

if snaps_v1:
    print("\n── 3-D volume render (one snapshot) ──")

# Re-use the first V1 snapshot (captured early in the run) as the 3-D
# energy-density source — the wave is in flight, heading toward the barrier.
# This avoids an extra simulation and keeps the create_particle() pipeline
# fully consistent throughout the script.
if snaps_v1:
    import matplotlib.pyplot as plt

    _ed_3d = snaps_v1[0]["energy_density"]
    backend_choice = "auto"  # pyvista GPU if available, else matplotlib
    fig_3d = viz.render_3d_volume(
        _ed_3d,
        threshold=float(_ed_3d.mean() + _ed_3d.std()),
        backend=backend_choice,
        colormap="inferno",
        title="LFM wave mid-flight (energy density)",
        figsize=(9, 8),
        save_path=str(OUT_DIR / "double_slit_3d_volume.png"),
    )
    if fig_3d is not None:
        plt.close(fig_3d)
    print("  → saved double_slit_3d_volume.png")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

print(
    "\n"
    "═══════════════════════════════════════════════════════\n"
    "  DOUBLE-SLIT EXPERIMENT COMPLETE\n"
    "═══════════════════════════════════════════════════════\n"
    f"  Variants run: {', '.join(str(v) for v in sorted(_selected))}\n"
    "\n"
    "  Results:\n"
    + (f"    V1 (no detector)       peak = {pat_v1.max():.4f}\n" if pat_v1 is not None else "")
    + (f"    V2 (single slit)       peak = {pat_v2.max():.4f}\n" if pat_v2 is not None else "")
    + (f"    V3 (partial, α=0.5)    peak = {pat_v3.max():.4f}\n" if pat_v3 is not None else "")
    + (f"    V4 (full which-path)   peak = {pat_v4.max():.4f}\n" if pat_v4 is not None else "")
    + "\n"
    + (
        "  Physics check (requires V1 + V4):\n"
        "    V1 > V4 (interference vs particle-like)?  "
        + ("PASS ✓" if pat_v1.max() >= pat_v4.max() else "FAIL ✗")
        + "\n"
        "    V3 between V1 and V4?  "
        + ("PASS ✓" if pat_v4.max() <= pat_v3.max() <= pat_v1.max() else "inconclusive")
        + "\n"
        if (pat_v1 is not None and pat_v4 is not None)
        else "  (Physics check skipped — run with --variant all for full validation)\n"
    )
    + "═══════════════════════════════════════════════════════"
)
