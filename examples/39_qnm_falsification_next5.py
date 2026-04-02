#!/usr/bin/env python3
"""
LFM Next-5 Ringdown Falsification Suite — Projection v2
=======================================================

Runs the corrected projection-based Next-5 suite inside `lfm-physics`.
This is the library-native version of the stabilized extraction campaign.

Usage
-----
    cd examples
    python 39_qnm_falsification_next5.py
    python 39_qnm_falsification_next5.py --quick

Outputs (in ``outputs/ringdown_next5/``)
----------------------------------------
    next5_projection_v2.json
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

from lfm.experiment import next5_falsification_projection_v2  # noqa: E402

parser = argparse.ArgumentParser(description="LFM Next-5 falsification suite (projection v2)")
parser.add_argument("--quick", action="store_true", help="Reduced-cost smoke configuration")
args = parser.parse_args()

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "ringdown_next5"

kwargs = {}
if args.quick:
    kwargs = {
        "f1_grid": 32,
        "f2_grids": (24, 32, 40),
        "f3_grid": 32,
        "f4_grid": 32,
        "f1_ring_steps": 300,
        "f2_ring_steps": 240,
        "f3_ring_steps": 300,
        "f4_ring_steps": 320,
        "merger_pre_steps": 120,
        "record_every": 4,
    }

result = next5_falsification_projection_v2(**kwargs)
written = result.save("next5_projection_v2", directory=OUT_DIR)

print("\n" + "=" * 72)
print("NEXT-5 FALSIFICATION VERDICTS (PROJECTION V2)")
print("=" * 72)
for key, value in result.summary["verdicts"].items():
    print(f"{key}: {value}")
print(
    f"PASS={result.summary['pass_count']}, "
    f"FAIL={result.summary['fail_count']}, "
    f"INVALID={result.summary['invalid_count']}"
)
print("\nOutputs:")
for kind, path in written.items():
    print(f"  {kind:12s} -> {path}")
