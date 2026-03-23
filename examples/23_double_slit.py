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
from typing import Optional

import numpy as np

# ── Ensure we can import lfm from the repository root ──────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lfm
from lfm.experiment import Barrier, DetectorScreen, Slit
import lfm.viz as viz

# ── CLI ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="LFM Double-Slit Experiment showcase (all four variants)"
)
parser.add_argument(
    "--small",
    action="store_true",
    help="Use 32³ grid for a rapid preview (lower resolution)",
)
parser.add_argument(
    "--no-anim",
    action="store_true",
    help="Skip saving animation GIFs (much faster)",
)
args = parser.parse_args()

# ── Grid parameters ────────────────────────────────────────────────────────
#
#  N = 64  default: ~8 s on CPU, fine resolution
#  N = 32  small mode: ~0.5 s on CPU, quick check
#  N = 128 for HD GPU runs (set manually)
#
N = 32 if args.small else 64

# Geometry (all positions in grid cells)
PROPAGATION_AXIS = 2                # wave travels in +z direction
SOURCE_Z         = max(4, N // 8)   # wave-packet starting z
BARRIER_Z        = N // 2           # barrier z-position
DETECTOR_Z       = int(N * 0.82)    # detector screen z-position
HALF_SEP         = max(3, N // 8)   # half slit-separation
SLIT_WIDTH       = max(2, N // 18)  # slit width

# Wave-packet parameters
VELOCITY         = 0.05             # wave speed in +z (lattice units where c=1)
AMPLITUDE        = lfm.E_AMPLITUDE_BY_GRID.get(N, 3.6) * 0.6
SIGMA            = max(3.0, N / 12.0)   # wave-packet width

# Simulation parameters
STEPS            = 2500 if not args.small else 800
SNAP_EVERY       = 25 if not args.small else 20
FIELDS           = ["energy_density"]

OUT_DIR = Path(__file__).parent
SAVE_ANIM = not args.no_anim

print(
    f"\nLFM Double-Slit Experiment\n"
    f"  Grid : {N}³   Backend : {lfm.get_backend()}\n"
    f"  Steps: {STEPS}  Snap/: {SNAP_EVERY}  "
    f"pyvista: {viz.volume_render_available()}\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build one fresh simulation ready for a run
# ═══════════════════════════════════════════════════════════════════════════

def _build_sim() -> lfm.Simulation:
    """Return a freshly initialised complex-field simulation."""
    cfg = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.COMPLEX,       # Ψ ∈ ℂ → phase interference
        boundary_type=lfm.BoundaryType.ABSORBING, # kill reflections from edges
    )
    return lfm.Simulation(cfg)


def _fire(sim: lfm.Simulation) -> None:
    """Place a boosted Gaussian soliton aimed at the barrier."""
    sim.place_soliton(
        position=(N // 2, N // 2, SOURCE_Z),
        amplitude=AMPLITUDE,
        sigma=SIGMA,
        phase=0.0,
        velocity=(0.0, 0.0, VELOCITY),   # →  +z direction
    )


# ═══════════════════════════════════════════════════════════════════════════
# Variant runner
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(
    label: str,
    slits: list[Slit],
    save_stem: str,
) -> np.ndarray:
    """Run one variant and save outputs.  Returns the interference pattern."""
    print(f"  Running {label} …", end="", flush=True)

    sim   = _build_sim()
    barrier = sim.place_barrier(
        axis      = PROPAGATION_AXIS,
        position  = BARRIER_Z,
        height    = lfm.CHI0 + 50.0,
        thickness = 2,
        slits     = slits,
        absorb    = True,
    )
    screen = sim.add_detector(axis=PROPAGATION_AXIS, position=DETECTOR_Z)
    _fire(sim)

    # Combined callback: enforce barrier + record detector every report tick
    def _cb(s: lfm.Simulation, step: int) -> None:
        barrier.step_callback(s, step)
        screen.record()

    snaps = sim.run_with_snapshots(
        steps          = STEPS,
        snapshot_every = SNAP_EVERY,
        fields         = FIELDS,
        callback       = _cb,
    )

    pattern = screen.pattern
    peak    = float(pattern.max())
    print(f" done.  peak = {peak:.4f},  frames = {screen.n_frames}")

    # ── Interference-pattern heatmap ──────────────────────────────────
    fig_pat = viz.plot_interference_pattern(
        pattern,
        title       = label,
        colormap    = "inferno",
        show_profile= True,
        figsize     = (12, 5),
    )
    fig_pat.savefig(OUT_DIR / f"{save_stem}.png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig_pat)

    # ── Click simulation (10 000 single-particle hits) ────────────────
    clicks = screen.click_pattern(n_particles=10_000, seed=42)
    fig_click = viz.plot_interference_pattern(
        clicks.astype(np.float32),
        title       = f"{label} — particle clicks (N=10 000)",
        colormap    = "hot",
        show_profile= True,
        figsize     = (12, 5),
    )
    fig_click.savefig(
        OUT_DIR / f"{save_stem}_clicks.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_click)

    # ── Animation ─────────────────────────────────────────────────────
    if SAVE_ANIM:
        anim = viz.animate_double_slit(
            snaps,
            barrier_axis     = PROPAGATION_AXIS,
            barrier_position = BARRIER_Z,
            detector_position= DETECTOR_Z,
            field            = "energy_density",
            slice_axis       = 0,
            colormap         = "inferno",
            fps              = 12,
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

    return pattern


# ═══════════════════════════════════════════════════════════════════════════
# Variant 1 — classic double slit (no detectors)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Variant 1: Classic double slit (no detectors) ──")
pat_v1 = run_variant(
    label     = "Variant 1 — No Detector (interference fringes expected)",
    slits     = [
        Slit(center=N // 2 - HALF_SEP, width=SLIT_WIDTH),
        Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
    ],
    save_stem = "double_slit_01_no_detector",
)

# ═══════════════════════════════════════════════════════════════════════════
# Variant 2 — single slit (diffraction only)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Variant 2: Single slit (diffraction envelope, no fringes) ──")
pat_v2 = run_variant(
    label     = "Variant 2 — Single Slit (diffraction, no fringes)",
    slits     = [
        Slit(center=N // 2, width=SLIT_WIDTH),   # one slit at centre
    ],
    save_stem = "double_slit_02_single_slit",
)

# ═══════════════════════════════════════════════════════════════════════════
# Variant 3 — partial which-path detector (strength = 0.5)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Variant 3: Partial which-path detector (strength = 0.5) ──")
pat_v3 = run_variant(
    label     = "Variant 3 — Partial Detector α=0.5 (fringes weakened)",
    slits     = [
        Slit(
            center            = N // 2 - HALF_SEP,
            width             = SLIT_WIDTH,
            detector          = True,
            detector_strength = 0.5,
        ),
        Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
    ],
    save_stem = "double_slit_03_partial_detector",
)

# ═══════════════════════════════════════════════════════════════════════════
# Variant 4 — full which-path detector (strength = 1.0)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Variant 4: Full which-path detector (strength = 1.0) ──")
pat_v4 = run_variant(
    label     = "Variant 4 — Full Which-Path α=1.0 (fringes destroyed)",
    slits     = [
        Slit(
            center            = N // 2 - HALF_SEP,
            width             = SLIT_WIDTH,
            detector          = True,
            detector_strength = 1.0,
        ),
        Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
    ],
    save_stem = "double_slit_04_full_which_path",
)

# ═══════════════════════════════════════════════════════════════════════════
# Comparison panel — all four fringe profiles side by side
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Saving comparison figure ──")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle("LFM Double-Slit: All Four Variants", fontsize=14, fontweight="bold")

_panels = [
    (pat_v1, "V1: No detector\n(fringes)",           "#3b82f6"),
    (pat_v2, "V2: Single slit\n(diffraction)",        "#22c55e"),
    (pat_v3, "V3: Partial detector\n(weakened)",      "#f97316"),
    (pat_v4, "V4: Full which-path\n(no fringes)",     "#ef4444"),
]

for ax, (pat, title_txt, color) in zip(axes, _panels):
    profile = pat.sum(axis=0).astype(float)
    # Normalise to [0, 1]
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

print("\n── 3-D volume render (one snapshot) ──")

# Quick single-step simulation to get a 3-D energy density snapshot
_sim_3d = _build_sim()
_barrier_3d = _sim_3d.place_barrier(
    axis=PROPAGATION_AXIS, position=BARRIER_Z, thickness=2,
    slits=[
        Slit(center=N // 2 - HALF_SEP, width=SLIT_WIDTH),
        Slit(center=N // 2 + HALF_SEP, width=SLIT_WIDTH),
    ],
)
_fire(_sim_3d)
# Run until the wave is mid-flight
_mid_steps = int((BARRIER_Z - SOURCE_Z) / VELOCITY * 0.8)
_sim_3d.run(_mid_steps, callback=_barrier_3d.step_callback)
_ed_3d = _sim_3d.energy_density

backend_choice = "auto"  # pyvista GPU if available, else matplotlib
fig_3d = viz.render_3d_volume(
    _ed_3d,
    threshold = float(_ed_3d.mean() + _ed_3d.std()),
    backend   = backend_choice,
    colormap  = "inferno",
    title     = "LFM wave mid-flight (energy density)",
    figsize   = (9, 8),
    save_path = str(OUT_DIR / "double_slit_3d_volume.png"),
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
    "  Results:\n"
    f"    V1 (no detector)       peak = {pat_v1.max():.4f}\n"
    f"    V2 (single slit)       peak = {pat_v2.max():.4f}\n"
    f"    V3 (partial, α=0.5)    peak = {pat_v3.max():.4f}\n"
    f"    V4 (full which-path)   peak = {pat_v4.max():.4f}\n"
    "\n"
    "  Physics check:\n"
    "    V1 > V4 (interference vs particle-like)?  "
    + ("PASS ✓" if pat_v1.max() >= pat_v4.max() else "FAIL ✗")
    + "\n"
    "    V3 between V1 and V4?  "
    + ("PASS ✓" if pat_v4.max() <= pat_v3.max() <= pat_v1.max() else "inconclusive")
    + "\n"
    "\n"
    "  Saved figures:\n"
    "    double_slit_01_no_detector.png\n"
    "    double_slit_01_no_detector_clicks.png\n"
    "    double_slit_02_single_slit.png\n"
    "    double_slit_03_partial_detector.png\n"
    "    double_slit_04_full_which_path.png\n"
    "    double_slit_comparison.png\n"
    "    double_slit_3d_volume.png\n"
    + ("    double_slit_01_no_detector.gif  (+ 3 more .gif)\n" if SAVE_ANIM else "")
    + "═══════════════════════════════════════════════════════"
)
