#!/usr/bin/env python3
"""
LFM Ringdown Projection Check — Showcase
========================================

Compares point-probe ringdown extraction against global low-k mode projection
and optionally writes a fixed-camera 3-D movie of the non-uniform χ response.

Usage
-----
    cd examples
    python 38_qnm_mode_projection.py
    python 38_qnm_mode_projection.py --small --movie
    python 38_qnm_mode_projection.py --grid 96 --profile uniform

Outputs (in ``outputs/ringdown_projection/``)
---------------------------------------------
    qnm_projection_check.png
    qnm_projection_check.json
    qnm_projection_check_3d_movie.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfm.experiment import qnm_mode_projection_check  # noqa: E402

parser = argparse.ArgumentParser(description="LFM ringdown projection diagnostic")
parser.add_argument("--small", action="store_true", help="Quick preview on N=48")
parser.add_argument("--large", action="store_true", help="Larger N=96 run")
parser.add_argument("--grid", type=int, default=None, metavar="N", help="Override grid size")
parser.add_argument(
    "--profile",
    choices=["uniform", "quadrupole"],
    default="quadrupole",
    help="Perturbation profile",
)
parser.add_argument("--movie", action="store_true", help="Save fixed-camera 3-D movie")
parser.add_argument("--movie-every", type=int, default=10, help="Capture one movie frame every N callbacks")
parser.add_argument("--movie-max-points", type=int, default=6000, help="Max voxels per movie frame")
args = parser.parse_args()

if args.grid is not None:
    N = args.grid
elif args.large:
    N = 96
elif args.small:
    N = 48
else:
    N = 64

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "ringdown_projection"

print("\n" + "=" * 60)
print("LFM Ringdown Projection Check")
print("=" * 60)
print(f"  Grid      : {N}^3")
print(f"  Profile   : {args.profile}")
print(f"  Movie     : {args.movie}")
print(f"  Output    : {OUT_DIR}")
print("=" * 60 + "\n")

result = qnm_mode_projection_check(
    N=N,
    perturb_profile=args.profile,
    capture_movie=args.movie,
    movie_every=args.movie_every,
    movie_max_points=args.movie_max_points,
)
written = result.save(
    "qnm_projection_check",
    directory=OUT_DIR,
    save_movie=args.movie,
)

comparison = result.summary["comparison"]
print("Comparison:")
print(f"  Probe spread     : {comparison['probe_omega_spread_frac']:.4f}")
print(f"  Projected spread : {comparison['projected_mode_omega_spread_frac']:.4f}")
print(f"  H0 status        : {comparison['h0_status']}")
print("\nOutputs:")
for kind, path in written.items():
    print(f"  {kind:12s} -> {path}")
