#!/usr/bin/env python
"""Run all 8 quantitative experiments and print a summary table.

Usage::

    python experiments/quantitative/run_all.py

Each experiment tests a QUANTITATIVE prediction of LFM against
known physics, using ONLY GOV-01 + GOV-02 (no external physics).
"""

from __future__ import annotations

import importlib
import sys
import time

EXPERIMENTS = [
    "exp_01_mass_energy",
    "exp_02_newton_gravity",
    "exp_03_coulomb_law",
    "exp_04_kg_dispersion",
    "exp_05_gravitational_redshift",
    "exp_06_mass_hierarchy",
    "exp_07_string_tension",
    "exp_08_dark_energy_modes",
]


def main() -> None:
    results: list[dict] = []
    timings: list[float] = []

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   LFM QUANTITATIVE EXPERIMENT SUITE — 8 EXPERIMENTS              ║")
    print("║   Goal: Close every cold-reviewer gap with NUMBERS               ║")
    print("╚" + "═" * 68 + "╝")
    print()

    for name in EXPERIMENTS:
        mod = importlib.import_module(f"experiments.quantitative.{name}")
        print(f"\n{'▸' * 3} Running {name} {'◂' * 3}\n")
        t0 = time.perf_counter()
        result = mod.run()
        elapsed = time.perf_counter() - t0
        results.append(result)
        timings.append(elapsed)
        print(f"\n  [{elapsed:.1f}s]")
        print()

    # ── Summary table ──────────────────────────────────
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " SUMMARY ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    header = f"  {'#':>2s}  {'Experiment':<20s}  {'Measured':<22s}  {'Expected':<16s}  {'Pass':>4s}"
    print(f"║{header:<78s}║")
    print("║" + "─" * 78 + "║")
    for i, (r, _t) in enumerate(zip(results, timings, strict=False)):
        status = "✅" if r["passed"] else "❌"
        line = (
            f"  {i + 1:2d}  {r['name']:<20s}  {r['measured']:<22s}  "
            f"{r['expected']:<16s}  {status:>4s}"
        )
        print(f"║{line:<78s}║")
    print("╠" + "═" * 78 + "╣")
    total_time = sum(timings)
    foot = f"  {n_pass}/{n_total} passed  ({total_time:.1f}s total)"
    print(f"║{foot:<78s}║")
    print("╚" + "═" * 78 + "╝")
    print()

    if n_pass == n_total:
        print("🏆 ALL EXPERIMENTS PASSED — LFM makes quantitative predictions.")
    else:
        print(f"⚠️  {n_total - n_pass} experiment(s) failed. Review output above.")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
