"""Benchmark: leapfrog evolver at several grid sizes.

Run standalone::

    python bench_evolver.py

Or with pytest-benchmark::

    pytest bench_evolver.py -v --benchmark-sort=mean
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

import lfm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(n: int, backend: str = "cpu") -> lfm.Simulation:
    config = lfm.SimulationConfig(grid_size=n)
    sim = lfm.Simulation(config, backend=backend)
    sim.place_soliton((n // 2, n // 2, n // 2), amplitude=5.0, sigma=3.0)
    sim.equilibrate()
    return sim


def _steps_per_second(sim: lfm.Simulation, steps: int = 200) -> float:
    t0 = time.perf_counter()
    sim.run(steps=steps, record_metrics=False)
    elapsed = time.perf_counter() - t0
    return steps / elapsed


# ---------------------------------------------------------------------------
# pytest-benchmark fixtures (used when running under pytest)
# ---------------------------------------------------------------------------

def test_evolver_n32_cpu(benchmark: object) -> None:  # type: ignore[type-arg]
    sim = _make_sim(32, "cpu")
    benchmark(sim.run, steps=100, record_metrics=False)  # type: ignore[call-arg]


def test_evolver_n64_cpu(benchmark: object) -> None:  # type: ignore[type-arg]
    sim = _make_sim(64, "cpu")
    benchmark(sim.run, steps=50, record_metrics=False)  # type: ignore[call-arg]


def test_evolver_n128_cpu(benchmark: object) -> None:  # type: ignore[type-arg]
    sim = _make_sim(128, "cpu")
    benchmark(sim.run, steps=10, record_metrics=False)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Standalone timing (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("LFM Evolver Benchmark — CPU")
    print("=" * 45)
    for n in (32, 48, 64, 96, 128):
        sim = _make_sim(n, "cpu")
        sps = _steps_per_second(sim)
        cells = n ** 3
        print(f"  N={n:3d}  ({cells:>10,} cells)  {sps:>10,.0f} steps/sec")

    if lfm.gpu_available():
        print()
        print("LFM Evolver Benchmark — GPU")
        print("=" * 45)
        for n in (64, 128, 256):
            sim = _make_sim(n, "gpu")
            sps = _steps_per_second(sim)
            cells = n ** 3
            print(f"  N={n:3d}  ({cells:>10,} cells)  {sps:>10,.0f} steps/sec")
    else:
        print("\n(No GPU detected — skipping GPU benchmarks)")
