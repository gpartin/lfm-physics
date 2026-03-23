"""Integration tests for sweep, io, and driven evolution."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import lfm
from lfm import FieldLevel, Simulation, SimulationConfig
from lfm.io import load_checkpoint, save_checkpoint
from lfm.sweep import sweep


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# sweep()
# --------------------------------------------------------------------------


class TestSweep:
    """Test parameter sweep functionality."""

    def test_sweep_returns_list(self):
        cfg = SimulationConfig(grid_size=8)
        results = sweep(
            cfg,
            param="kappa",
            values=[0.01, 0.02],
            steps=5,
            equilibrate=False,
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sweep_includes_param_value(self):
        cfg = SimulationConfig(grid_size=8)
        results = sweep(
            cfg,
            param="kappa",
            values=[0.01, 0.02],
            steps=5,
            equilibrate=False,
        )
        assert results[0]["kappa"] == 0.01
        assert results[1]["kappa"] == 0.02

    def test_sweep_has_metrics(self):
        cfg = SimulationConfig(grid_size=8)
        results = sweep(
            cfg,
            param="kappa",
            values=[0.016],
            steps=5,
            equilibrate=False,
        )
        assert "chi_min" in results[0]
        assert "energy_total" in results[0]

    def test_sweep_metric_filter(self):
        cfg = SimulationConfig(grid_size=8)
        results = sweep(
            cfg,
            param="kappa",
            values=[0.016],
            steps=5,
            metric_names=["chi_min"],
            equilibrate=False,
        )
        assert "chi_min" in results[0]
        assert "total_energy" not in results[0]

    def test_sweep_with_soliton(self):
        cfg = SimulationConfig(grid_size=16)
        results = sweep(
            cfg,
            param="kappa",
            values=[1 / 63],
            steps=5,
            soliton={"amplitude": 3.0},
            equilibrate=False,
        )
        assert len(results) == 1
        # With a soliton, chi_min should be below chi0
        assert results[0]["chi_min"] < lfm.CHI0

    def test_sweep_with_equilibrate(self):
        cfg = SimulationConfig(grid_size=16)
        results = sweep(
            cfg,
            param="kappa",
            values=[1 / 63],
            steps=5,
            soliton={"amplitude": 3.0},
            equilibrate=True,
        )
        assert len(results) == 1


# --------------------------------------------------------------------------
# io: save_checkpoint / load_checkpoint
# --------------------------------------------------------------------------


class TestIO:
    """Test checkpoint save/load via the io module."""

    def test_save_creates_file(self):
        sim = Simulation(SimulationConfig(grid_size=8))
        sim.run(steps=2)
        with tempfile.TemporaryDirectory() as td:
            p = save_checkpoint(sim, Path(td) / "test.npz")
            assert p.exists()

    def test_round_trip_preserves_step(self):
        sim = Simulation(SimulationConfig(grid_size=8))
        sim.run(steps=5)
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "ckpt.npz"
            save_checkpoint(sim, fpath)
            sim2 = load_checkpoint(fpath)
            assert sim2.step == 5

    def test_round_trip_preserves_chi(self):
        cfg = SimulationConfig(grid_size=8)
        sim = Simulation(cfg)
        sim.place_soliton((4, 4, 4), amplitude=3.0)
        sim.run(steps=5)
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "ckpt.npz"
            save_checkpoint(sim, fpath)
            sim2 = load_checkpoint(fpath)
            np.testing.assert_allclose(sim2.chi, sim.chi, atol=1e-6)

    def test_round_trip_preserves_psi(self):
        cfg = SimulationConfig(grid_size=8)
        sim = Simulation(cfg)
        sim.place_soliton((4, 4, 4), amplitude=3.0)
        sim.run(steps=5)
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "ckpt.npz"
            save_checkpoint(sim, fpath)
            sim2 = load_checkpoint(fpath)
            np.testing.assert_allclose(sim2.psi_real, sim.psi_real, atol=1e-6)

    def test_continued_run_after_load(self):
        cfg = SimulationConfig(grid_size=8)
        sim = Simulation(cfg)
        sim.place_soliton((4, 4, 4), amplitude=3.0)
        sim.run(steps=3)
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "ckpt.npz"
            save_checkpoint(sim, fpath)
            sim2 = load_checkpoint(fpath)
            sim2.run(steps=3)
            assert sim2.step == 6


# --------------------------------------------------------------------------
# run_driven (parametric resonance interface)
# --------------------------------------------------------------------------


class TestRunDriven:
    """Test driven evolution with external chi forcing."""

    def test_driven_basic(self):
        cfg = SimulationConfig(grid_size=8)
        sim = Simulation(cfg)
        # Uniform chi forcing — should keep chi at CHI0
        sim.run_driven(
            steps=5,
            chi_forcing=lambda t: np.float32(lfm.CHI0),
        )
        assert sim.step == 5

    def test_driven_oscillating_chi(self):
        cfg = SimulationConfig(grid_size=8)
        sim = Simulation(cfg)
        omega = 2 * lfm.CHI0  # parametric resonance frequency
        sim.run_driven(
            steps=10,
            chi_forcing=lambda t: np.float32(lfm.CHI0 + 0.3 * np.sin(omega * t)),
        )
        assert sim.step == 10

    def test_driven_record_metrics(self):
        cfg = SimulationConfig(grid_size=8, report_interval=5)
        sim = Simulation(cfg)
        sim.run_driven(
            steps=10,
            chi_forcing=lambda t: np.float32(lfm.CHI0),
            record_metrics=True,
        )
        assert len(sim.history) >= 1


# --------------------------------------------------------------------------
# GPU COLOR (v14 kappa_c) regression
# --------------------------------------------------------------------------


class TestGPUColor:
    """Regression test for GPU COLOR with kappa_c > 0."""

    @pytest.mark.skipif(
        not _cupy_available(),
        reason="CuPy not available",
    )
    def test_gpu_color_kappa_c_no_nan(self):
        """GPU COLOR with kappa_c>0 must not produce NaN (regression)."""
        cfg = SimulationConfig(
            grid_size=16,
            field_level=FieldLevel.COLOR,
            kappa_c=1 / 189,
        )
        sim = Simulation(cfg, backend="gpu")
        sim.place_soliton((8, 8, 8), amplitude=6.0)
        sim.run(steps=20)
        assert not np.any(np.isnan(sim.chi))
        assert not np.any(np.isnan(sim.psi_real))
