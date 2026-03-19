"""Benchmark: field allocation, equilibration, and analysis operations.

Run standalone::

    python bench_fields.py

Or with pytest-benchmark::

    pytest bench_fields.py -v --benchmark-sort=mean
"""

from __future__ import annotations

import time

import numpy as np

import lfm
from lfm.analysis import radial_profile, energy_components


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_sim(n: int) -> lfm.Simulation:
    config = lfm.SimulationConfig(grid_size=n)
    sim = lfm.Simulation(config, backend="cpu")
    sim.place_soliton((n // 2,) * 3, amplitude=8.0, sigma=3.0)
    sim.equilibrate()
    sim.run(steps=100, record_metrics=False)
    return sim


# ---------------------------------------------------------------------------
# Equilibration benchmarks
# ---------------------------------------------------------------------------

def test_equilibrate_n32(benchmark: object) -> None:
    config = lfm.SimulationConfig(grid_size=32)
    sim = lfm.Simulation(config, backend="cpu")
    sim.place_soliton((16, 16, 16), amplitude=5.0, sigma=3.0)
    benchmark(sim.equilibrate)  # type: ignore[call-arg]


def test_equilibrate_n64(benchmark: object) -> None:
    config = lfm.SimulationConfig(grid_size=64)
    sim = lfm.Simulation(config, backend="cpu")
    sim.place_soliton((32, 32, 32), amplitude=5.0, sigma=3.0)
    benchmark(sim.equilibrate)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Analysis benchmarks
# ---------------------------------------------------------------------------

def test_radial_profile_n32(benchmark: object) -> None:
    sim = _base_sim(32)
    benchmark(radial_profile, sim.fields.chi, center=(16, 16, 16))  # type: ignore[call-arg]


def test_radial_profile_n64(benchmark: object) -> None:
    sim = _base_sim(64)
    benchmark(radial_profile, sim.fields.chi, center=(32, 32, 32))  # type: ignore[call-arg]


def test_energy_components_n32(benchmark: object) -> None:
    sim = _base_sim(32)
    benchmark(energy_components, sim.fields)  # type: ignore[call-arg]


def test_energy_components_n64(benchmark: object) -> None:
    sim = _base_sim(64)
    benchmark(energy_components, sim.fields)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Standalone timing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("LFM Field-Ops Benchmark")
    print("=" * 48)

    for n in (32, 64, 128):
        sim = _base_sim(n)

        # radial_profile
        N_REPS = 20
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            radial_profile(sim.fields.chi, center=(n // 2,) * 3)
        t_rp = (time.perf_counter() - t0) / N_REPS * 1_000

        # energy_components
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            energy_components(sim.fields)
        t_ec = (time.perf_counter() - t0) / N_REPS * 1_000

        print(f"  N={n:3d}  radial_profile={t_rp:6.2f} ms   energy_components={t_ec:6.2f} ms")
