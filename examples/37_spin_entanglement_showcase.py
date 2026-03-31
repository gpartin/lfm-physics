#!/usr/bin/env python3
"""LFM Spin Entanglement Experiment — Showcase
===============================================

Two spin-1/2 spinor solitons initialized in correlated quantum spin
states on a 3-D lattice.  Physics from GOV-01 (Lattice Wave Equation)
+ GOV-02 (χ Field Equation) alone — no quantum postulates injected.

What you will see
-----------------
*  Two bright spots (chi deficit wells) that stay stable in the 3-D movie
   regardless of spin configuration — confirming **spin-blind gravity**.
*  Spin expectation values ⟨σ_z^A⟩, ⟨σ_z^B⟩ preserved over the run.
*  CHSH Bell parameter S ≤ 2 for product states (local-deterministic
   limit from GOV-01 alone).

Usage
-----
From the ``examples/`` directory::

    python 37_spin_entanglement_showcase.py             # default N=64
    python 37_spin_entanglement_showcase.py --small     # N=48, quick preview
    python 37_spin_entanglement_showcase.py --large     # N=96, high-fidelity
    python 37_spin_entanglement_showcase.py --no-anim   # skip 3-D movie
    python 37_spin_entanglement_showcase.py --all       # all 4 spin configs
    python 37_spin_entanglement_showcase.py --config triplet
    python 37_spin_entanglement_showcase.py --config singlet --grid 80
    python 37_spin_entanglement_showcase.py --steps 8000

Outputs saved to ``outputs/spin_entanglement/<config>/``
---------------------------------------------------------
*  ``spin_entanglement_<config>.png``  — summary figure
*  ``spin_entanglement_<config>_snapshots.npz``  — full field snapshots
*  ``spin_entanglement_<config>_3d_movie.mp4``   — animated chi-deficit movie

Reference
---------
LFM-PAPER-048 (Spinor Representation in the Lattice Field Medium).
See also: examples/36_spin_entanglement.py (static analysis / CHSH table).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Add library root to path when running without installation ─────────────
_HERE = Path(__file__).resolve().parent
_LIB_ROOT = _HERE.parent
if str(_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIB_ROOT))

# ── Output directory ───────────────────────────────────────────────────────
OUT_ROOT = _HERE / "outputs" / "spin_entanglement"

# ── CLI ───────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LFM Spin Entanglement showcase — GOV-01 + GOV-02 only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="antiparallel",
        choices=["triplet", "antiparallel", "product_x", "singlet", "all"],
        help="Spin configuration (or 'all' for the comparison suite).",
    )
    size_g = p.add_mutually_exclusive_group()
    size_g.add_argument("--small", action="store_true",
                        help="Use N=48 (fast preview).")
    size_g.add_argument("--large", action="store_true",
                        help="Use N=96 (high-fidelity).")
    p.add_argument("--grid", type=int, default=64,
                   help="Grid size N (overridden by --small/--large).")
    p.add_argument("--all", dest="run_all", action="store_true",
                   help="Run all 4 spin configurations (suite mode).")
    p.add_argument("--no-anim", dest="no_anim", action="store_true",
                   help="Skip 3-D movie rendering (faster).")
    p.add_argument("--steps", type=int, default=4000,
                   help="Number of leapfrog evolution steps.")
    p.add_argument("--amplitude", type=float, default=5.0,
                   help="Spinor soliton peak amplitude.")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Spinor Gaussian width (lattice cells).")
    p.add_argument("--verbose", action="store_true",
                   help="Print step-rate progress during evolution.")
    return p.parse_args()


def _print_banner() -> None:
    print()
    print("=" * 62)
    print("  LFM Spin Entanglement Showcase")
    print("  Two GOV-01 spinor solitons in correlated spin states")
    print("=" * 62)


def _grid_size(args: argparse.Namespace) -> int:
    if args.small:
        return 48
    if args.large:
        return 96
    return args.grid


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()
    _print_banner()

    from lfm.experiment import entanglement, SPIN_CONFIGS

    N = _grid_size(args)
    animate = not args.no_anim
    config = "all" if args.run_all else args.config

    print(f"\n  Config     : {config}")
    print(f"  Grid       : {N}³ = {N**3:,} cells")
    print(f"  Steps      : {args.steps}")
    print(f"  Amplitude  : {args.amplitude}")
    print(f"  Sigma      : {args.sigma}")
    print(f"  Animation  : {'yes (3-D MP4)' if animate else 'no'}")
    print()

    # ── Run ───────────────────────────────────────────────────────
    result = entanglement(
        config=config,
        N=N,
        amplitude=args.amplitude,
        sigma=args.sigma,
        total_steps=args.steps,
        animate=animate,
        verbose=args.verbose or True,
    )

    # ── Save outputs ──────────────────────────────────────────────
    is_suite = config == "all"
    stem = "spin_entanglement"

    if is_suite:
        out_dir = OUT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        written = result.save(
            stem,
            directory=out_dir,
            save_movie=animate,
        )
        print("\n  Files written:")
        for cfg_key, files in written.items():
            for kind, path in files.items():
                print(f"    [{cfg_key}/{kind}]  {path.relative_to(_HERE)}")

        print()
        print("  CHSH Bell Parameter Summary")
        print("  " + "-" * 44)
        print(result.chsh_table())

    else:
        out_dir = OUT_ROOT / config
        out_dir.mkdir(parents=True, exist_ok=True)
        written = result.save(
            f"{stem}_{config}",
            directory=out_dir,
            save_movie=animate,
        )

        print("\n  Files written:")
        for kind, path in written.items():
            print(f"    [{kind}]  {path.relative_to(_HERE)}")

        print()
        print(f"  Config          : {result.config_name}")
        print(f"  χ_min (initial) : {result.chi_min_initial:.4f}")
        chi_fin = result.chi_min_history[-1] if result.chi_min_history else float("nan")
        print(f"  χ_min (final)   : {chi_fin:.4f}")
        print(f"  CHSH S (initial): {result.chsh_initial:.4f}")
        print(f"  CHSH S (final)  : {result.chsh_final:.4f}  "
              f"({'≤ 2 ✓' if result.chsh_final <= 2.0 + 1e-6 else '> 2 !'} classical bound)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
