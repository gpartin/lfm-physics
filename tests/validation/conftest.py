"""Shared fixtures and helpers for particle-validation tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from lfm.analysis.energy import total_energy
from lfm.simulation import Simulation

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """Auto-add ``validation`` marker to every test in this package."""
    for item in items:
        if "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default grid / step budget — small enough for fast CI.
GRID = 32
GRID_TWO = 48  # Larger grid for two-particle interaction tests.
STEPS = 500  # Enough to confirm soliton survival (profiled: retention >9× at 500).
WARMUP = 2  # Steps needed so that psi_real_prev is populated.
BACKEND = "auto"  # Use GPU when available (CuPy + CUDA libs installed).


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def snapshot_energy(sim: Simulation) -> float:
    """Return total field energy for the current simulation state.

    Requires at least one ``sim.run()`` call (so *psi_real_prev* exists).
    """
    pr_prev = sim.psi_real_prev
    assert pr_prev is not None, "psi_real_prev is None — run at least 1 step first"
    return float(
        total_energy(
            sim.psi_real,
            pr_prev,
            sim.chi,
            dt=sim.config.dt,
            psi_i=sim.psi_imag,
            psi_i_prev=sim.psi_imag_prev,
        )
    )


def peak_psi_sq(sim: Simulation) -> float:
    """Max |Ψ|² (summed over color components if present)."""
    psi_sq: np.ndarray = sim.psi_real**2
    pi = sim.psi_imag
    if pi is not None:
        psi_sq = psi_sq + pi**2
    if psi_sq.ndim == 4:
        psi_sq = psi_sq.sum(axis=0)
    return float(psi_sq.max())


# ---------------------------------------------------------------------------
# ParticleMetrics — one-shot create → equilibrate → evolve → measure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleMetrics:
    """Immutable snapshot of a particle's stability metrics."""

    chi_min: float  # χ_min after equilibration (before evolution).
    chi0: float  # Background χ₀ from config.
    retention: float  # E_final / E_initial after *steps* of evolution.
    peak: float  # max |Ψ|² after evolution.


def measure_particle(
    name: str,
    preset_fn: Callable[..., Any],
    *,
    steps: int = STEPS,
) -> ParticleMetrics:
    """Create a particle, equilibrate, evolve, and return metrics.

    This is the single expensive operation — all assertions should be
    cheap checks on the returned :class:`ParticleMetrics`.
    """
    cfg = preset_fn(grid_size=GRID)
    sim = Simulation(cfg, backend=BACKEND)
    sim.place_particle(name)
    sim.equilibrate()

    chi_min = float(sim.chi.min())
    chi0 = float(cfg.chi0)

    sim.run(steps=WARMUP)
    e0 = snapshot_energy(sim)
    sim.run(steps=steps)
    e1 = snapshot_energy(sim)
    retention = e1 / e0 if abs(e0) > 1e-30 else 0.0
    peak = peak_psi_sq(sim)

    return ParticleMetrics(chi_min=chi_min, chi0=chi0, retention=retention, peak=peak)
