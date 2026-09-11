"""Regression tests for v38 chi self-interaction and chi-mode wiring."""

from __future__ import annotations

import numpy as np
import pytest

import lfm


def _expected_flat_octic_acceleration(chi: float, chi0: float, lambda_self: float) -> float:
    delta = chi * chi - chi0 * chi0
    return -(8.0 * lambda_self / chi0**4) * chi * delta**3


def _expected_quartic_acceleration(chi: float, chi0: float, lambda_self: float) -> float:
    return -4.0 * lambda_self * chi * (chi * chi - chi0 * chi0)


def test_production_run_uses_v38_flat_octic_self_interaction() -> None:
    n = 8
    dt = 0.02
    delta = 0.125
    chi_value = float(lfm.CHI0) + delta
    cfg = lfm.SimulationConfig(
        grid_size=n,
        field_level=lfm.FieldLevel.REAL,
        chi_mode=lfm.ChiMode.WAVE,
        boundary_type=lfm.BoundaryType.PERIODIC,
        dt=dt,
        lambda_self=lfm.LAMBDA_H,
        kappa=0.0,
        e0_sq=0.0,
        enable_chi_floor=False,
        precision=lfm.Precision.FLOAT64,
        report_interval=100_000,
    )
    sim = lfm.Simulation(cfg, backend="cpu")
    chi_start = np.full((n, n, n), chi_value, dtype=np.float64)
    zero = np.zeros((n, n, n), dtype=np.float64)
    sim.chi = chi_start
    sim.chi_previous = chi_start.copy()
    sim.set_psi_real(zero)
    sim.set_psi_real_prev(zero)

    sim.run(steps=1, record_metrics=False)

    measured = float(np.mean((sim.chi - chi_start) / (dt * dt)))
    flat_octic = _expected_flat_octic_acceleration(chi_value, float(lfm.CHI0), float(lfm.LAMBDA_H))
    quartic = _expected_quartic_acceleration(chi_value, float(lfm.CHI0), float(lfm.LAMBDA_H))
    assert measured == pytest.approx(flat_octic, rel=1.0e-10, abs=1.0e-12)
    assert measured != pytest.approx(quartic, rel=1.0e-3)


def test_chi_mode_memory_is_not_silently_ignored() -> None:
    cfg = lfm.SimulationConfig(
        grid_size=8,
        field_level=lfm.FieldLevel.REAL,
        chi_mode=lfm.ChiMode.MEMORY,
        boundary_type=lfm.BoundaryType.PERIODIC,
    )
    with pytest.raises(NotImplementedError, match="ChiMode.MEMORY"):
        lfm.Simulation(cfg, backend="cpu")


def test_galactic_scale_uses_live_v38_wave_path() -> None:
    cfg = lfm.SimulationConfig.for_scale(lfm.PhysicsScale.GALACTIC, grid_size=8)
    assert cfg.field_level == lfm.FieldLevel.REAL
    assert cfg.chi_mode == lfm.ChiMode.WAVE
    assert cfg.lambda_self == lfm.LAMBDA_H
