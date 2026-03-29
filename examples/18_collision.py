#!/usr/bin/env python3
"""
LFM Particle Collision Experiment — Showcase
=============================================

Smash two particles together on a 3-D lattice and generate a dramatic
MP4 movie of the collision — all physics from GOV-01 + GOV-02 alone.

The default collision is **proton + antiproton** annihilation: opposite-
phase solitons approach, interfere destructively at contact, and radiate
energy outward.

Usage
-----
    cd examples
    python 24_collision.py                           # default N=128 p+p̅
    python 24_collision.py --small                   # N=64 quick preview
    python 24_collision.py --large                   # N=256 publication
    python 24_collision.py --grid 192                # custom grid size
    python 24_collision.py --particles electron positron
    python 24_collision.py --speed 0.12              # faster approach
    python 24_collision.py --no-anim                 # skip 3-D movie

Outputs (in ``outputs/collision/``)
------------------------------------
    collision_proton_antiproton.png            summary figure
    collision_proton_antiproton_snapshots.npz  field snapshots
    collision_proton_antiproton_3d_movie.mp4   volumetric 3-D movie
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Windows: force UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure we can import lfm from the repository root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfm.experiment import collision, CollisionResult  # noqa: E402

# ── CLI ──────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(
    description="LFM Particle Collision — head-on collision experiment"
)
_parser.add_argument("--small", action="store_true", help="N=64 quick preview")
_parser.add_argument("--large", action="store_true", help="N=256 publication quality (GPU)")
_parser.add_argument("--grid", type=int, default=None, metavar="N",
                     help="Override grid size")
_parser.add_argument("--particles", nargs=2, default=["proton", "antiproton"],
                     metavar=("A", "B"),
                     help="Two particle names from catalog (default: proton antiproton)")
_parser.add_argument("--speed", type=float, default=0.10,
                     help="Approach speed per particle in units of c (default: 0.10)")
_parser.add_argument("--amplitude", type=float, default=3.0,
                     help="Soliton peak amplitude (default: 3.0, motion-safe)")
_parser.add_argument("--no-anim", action="store_true",
                     help="Skip 3-D movie capture (faster)")
args = _parser.parse_args()

if args.grid is not None:
    N = args.grid
elif args.large:
    N = 256
elif args.small:
    N = 64
else:
    N = 128

ANIMATE = not args.no_anim
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "collision"

# ── Run ──────────────────────────────────────────────────────────────────

particle_a, particle_b = args.particles
stem = f"collision_{particle_a}_{particle_b}"

print(f"\n{'=' * 60}")
print(f"LFM Particle Collision Experiment")
print(f"{'=' * 60}")
print(f"  Particles : {particle_a} + {particle_b}")
print(f"  Grid      : {N}³ = {N**3:,} cells")
print(f"  Speed     : {args.speed} c")
print(f"  Amplitude : {args.amplitude}")
print(f"  Animate   : {ANIMATE}")
print(f"  Output    : {OUT_DIR / stem}")
print(f"{'=' * 60}\n")

t0 = time.perf_counter()

result = collision(
    particle_a,
    particle_b,
    N=N,
    speed=args.speed,
    amplitude=args.amplitude,
    animate=ANIMATE,
    verbose=True,
)

elapsed = time.perf_counter() - t0
print(f"\n  Physics complete in {elapsed:.1f}s")
print(f"  Annihilation fraction: {result.annihilation_fraction:.1%}")

# ── Save outputs ─────────────────────────────────────────────────────────

print(f"  Saving to {OUT_DIR} …")
written = result.save(
    stem,
    directory=OUT_DIR,
    save_movie=ANIMATE,
    save_snapshots_npz=True,
)

print(f"\n  Outputs:")
for kind, path in written.items():
    sz = path.stat().st_size / 1024
    unit = "KB"
    if sz > 1024:
        sz /= 1024
        unit = "MB"
    print(f"    {kind:12s} → {path.name}  ({sz:.1f} {unit})")

print(f"\n{'=' * 60}")
print(f"Done.  Total wall time: {time.perf_counter() - t0:.1f}s")
print(f"{'=' * 60}\n")
