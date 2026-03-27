#!/usr/bin/env python3
"""
LFM Double-Slit Experiment — Complete Showcase
===============================================

Demonstrates all eight variants of the double-slit experiment using the
:func:`lfm.experiment.double_slit` high-level API.

Variants
--------
1. No detector         — classic Young's double slit -> bright fringes
2. Single slit         — diffraction envelope only (no fringes)
3. Partial which-path  — one-slit detector (a = 0.5) -> fringes weakened
4. Full which-path     — one-slit detector (a = 1.0) -> fringes destroyed
5. Wave packet         — single soliton "one particle at a time"
6. Far-field           — Fraunhofer regime -> parallel Young's stripes
7. Far-field partial   — FF with partial which-path detector
8. Far-field full      — FF with full which-path detector

Usage
-----
    cd examples
    python 23_double_slit.py              # default N=64, all variants
    python 23_double_slit.py --small      # N=32 quick preview
    python 23_double_slit.py --large      # N=128 HD
    python 23_double_slit.py --grid 256   # ultra-HD (publication quality)
    python 23_double_slit.py --variant 1  # single variant
    python 23_double_slit.py --no-anim    # skip 3-D movie capture

Outputs (in ``outputs/double_slit/``)
--------------------------------------
    double_slit_01_no_detector.png / _clicks.png / _3d_movie.mp4
    double_slit_02_single_slit.png / ...
    double_slit_03_partial_detector.png / ...
    double_slit_04_full_which_path.png / ...
    double_slit_05_wave_packet.png / ...
    double_slit_06_ff_no_detector.png / ...
    double_slit_07_ff_partial_detector.png / ...
    double_slit_08_ff_full_which_path.png / ...
    double_slit_comparison_nf.png     (near-field variants side-by-side)
    double_slit_comparison_ff.png     (far-field variants side-by-side)
    double_slit_comparison_regime.png (near-field vs far-field)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows: force UTF-8 so Unicode characters in progress output do not raise
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure we can import lfm from the repository root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfm.experiment import double_slit, DoubleSlit  # noqa: E402

# -- CLI -------------------------------------------------------------------

_parser = argparse.ArgumentParser(
    description="LFM Double-Slit Experiment showcase -- all eight variants"
)
_parser.add_argument("--small",   action="store_true", help="N=32 fast preview")
_parser.add_argument("--large",   action="store_true", help="N=128 HD quality (GPU recommended)")
_parser.add_argument("--grid",    type=int, default=None, metavar="N", help="Override grid size")
_parser.add_argument("--no-anim", action="store_true", help="Skip 3-D movie capture (faster)")
_parser.add_argument(
    "--variant",
    type=str,
    nargs="+",
    default=["all"],
    metavar="V",
    help="Variant(s) to run: 1-8 or 'all' (default: all)",
)
args = _parser.parse_args()

if args.grid is not None:
    N = args.grid
elif args.large:
    N = 128
elif args.small:
    N = 32
else:
    N = 64

ANIMATE = not args.no_anim
_selected: set[int] = (
    {1, 2, 3, 4, 5, 6, 7, 8}
    if "all" in args.variant
    else {int(v) for v in args.variant}
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "double_slit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _run(v: int) -> bool:
    """Return True if variant *v* should be executed."""
    return v in _selected


# -- Storage ----------------------------------------------------------------
# Keyed by variant number; None = not requested or not yet run.
V: dict[int, DoubleSlit | None] = {v: None for v in range(1, 9)}

# -- Variants 1-5: near-field -----------------------------------------------

if _run(1):
    V[1] = double_slit(N=N, animate=ANIMATE, label="No detector (near-field)")
    V[1].save("double_slit_01_no_detector", directory=OUT_DIR)

if _run(2):
    V[2] = double_slit(N=N, n_slits=1, animate=ANIMATE, label="Single slit")
    V[2].save("double_slit_02_single_slit", directory=OUT_DIR)

if _run(3):
    V[3] = double_slit(N=N, which_path=0.5, animate=ANIMATE,
                       label="Partial which-path a=0.5 (near-field)")
    V[3].save("double_slit_03_partial_detector", directory=OUT_DIR)

if _run(4):
    V[4] = double_slit(N=N, which_path=1.0, animate=ANIMATE,
                       label="Full which-path a=1.0 (near-field)")
    V[4].save("double_slit_04_full_which_path", directory=OUT_DIR)

if _run(5):
    V[5] = double_slit(N=N, mode="packet", animate=ANIMATE,
                       label="Wave packet (near-field)")
    V[5].save("double_slit_05_wave_packet", directory=OUT_DIR)

# -- Variants 6-8: far-field ------------------------------------------------

if _run(6):
    V[6] = double_slit(N=N, far_field=True, animate=ANIMATE,
                       label="No detector (far-field)")
    V[6].save("double_slit_06_ff_no_detector", directory=OUT_DIR)

if _run(7):
    V[7] = double_slit(N=N, far_field=True, which_path=0.5, animate=ANIMATE,
                       label="Partial which-path a=0.5 (far-field)")
    V[7].save("double_slit_07_ff_partial_detector", directory=OUT_DIR)

if _run(8):
    V[8] = double_slit(N=N, far_field=True, which_path=1.0, animate=ANIMATE,
                       label="Full which-path a=1.0 (far-field)")
    V[8].save("double_slit_08_ff_full_which_path", directory=OUT_DIR)

# -- Comparison figures -----------------------------------------------------

ran = {v: res for v, res in V.items() if res is not None}

if len(ran) > 1:
    import matplotlib.pyplot as plt

    # Near-field variants (1-5) side-by-side
    nf = [ran[v] for v in [1, 2, 3, 4, 5] if v in ran]
    if len(nf) > 1:
        fig = DoubleSlit.compare(nf, save_path=OUT_DIR / "double_slit_comparison_nf.png")
        plt.close(fig)
        print(f"\nSaved: {OUT_DIR / 'double_slit_comparison_nf.png'}")

    # Far-field variants (6-8) side-by-side
    ff = [ran[v] for v in [6, 7, 8] if v in ran]
    if len(ff) > 1:
        fig = DoubleSlit.compare(ff, save_path=OUT_DIR / "double_slit_comparison_ff.png")
        plt.close(fig)
        print(f"Saved: {OUT_DIR / 'double_slit_comparison_ff.png'}")

    # Near-field vs far-field (V1 and V6)
    regime = [ran[v] for v in [1, 6] if v in ran]
    if len(regime) == 2:
        fig = DoubleSlit.compare(regime,
                                  save_path=OUT_DIR / "double_slit_comparison_regime.png")
        plt.close(fig)
        print(f"Saved: {OUT_DIR / 'double_slit_comparison_regime.png'}")

    # Full 8-variant summary if everything ran
    if len(ran) == 8:
        fig = DoubleSlit.compare(
            [ran[v] for v in range(1, 9)],
            save_path=OUT_DIR / "double_slit_comparison_all.png",
        )
        plt.close(fig)
        print(f"Saved: {OUT_DIR / 'double_slit_comparison_all.png'}")

# -- 3-D energy-density volume render from V1 physics snapshots -------------
if 1 in ran and ran[1].snapshots:
    try:
        import lfm.viz as viz
        fig3d = viz.volume_render(
            ran[1].snapshots[-1]["energy_density"],
            title="V1 Energy density (final frame)",
        )
        p_3d = OUT_DIR / "double_slit_3d_volume.png"
        fig3d.savefig(str(p_3d), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig3d)
        print(f"Saved: {p_3d}")
    except Exception as exc:
        print(f"(3-D volume render skipped: {exc})")

# -- Summary ----------------------------------------------------------------
print("\n" + "=" * 68)
print(f"{'Variant':<6}  {'Label':<40}  {'Visibility':>10}  {'N_F':>6}")
print("-" * 68)
for v in sorted(ran):
    res = ran[v]
    print(f"  V{v}     {res.label:<40}  {res.visibility:>10.3f}  {res.fresnel_number:>6.2f}")
print("=" * 68)
print(f"\nAll outputs written to: {OUT_DIR}")
