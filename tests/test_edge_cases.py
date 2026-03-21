"""Edge-case and error-handling tests for lfm."""

import numpy as np
import pytest

from lfm.analysis.color import color_variance
from lfm.analysis.spectrum import power_spectrum
from lfm.analysis.tracker import flatten_trajectories
from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.simulation import Simulation

# ── Config edge cases ────────────────────────────────────────────────

class TestConfigEdgeCases:
    def test_kappa_c_without_color_level(self):
        """kappa_c > 0 at REAL level should still construct (config doesn't block)."""
        cfg = SimulationConfig(grid_size=16, kappa_c=1 / 189, field_level=FieldLevel.REAL)
        assert cfg.kappa_c > 0

    def test_epsilon_cc_without_color_level(self):
        cfg = SimulationConfig(grid_size=16, epsilon_cc=2 / 17, field_level=FieldLevel.REAL)
        assert cfg.epsilon_cc > 0

    def test_chi0_zero(self):
        """chi0=0 is pathological but should not crash config creation."""
        with pytest.raises(ValueError):
            SimulationConfig(grid_size=16, chi0=0.0)

    def test_periodic_boundary(self):
        cfg = SimulationConfig(grid_size=16, boundary_type=BoundaryType.PERIODIC)
        assert cfg.boundary_type == BoundaryType.PERIODIC


# ── Simulation edge cases ────────────────────────────────────────────

class TestSimulationEdgeCases:
    def test_zero_step_run(self):
        """Running 0 steps should be a no-op."""
        cfg = SimulationConfig(grid_size=16)
        sim = Simulation(cfg)
        sim.run(steps=0)
        assert sim.step == 0

    def test_single_step(self):
        cfg = SimulationConfig(grid_size=16)
        sim = Simulation(cfg)
        sim.run(steps=1)
        assert sim.step == 1
        assert not np.any(np.isnan(sim.chi))

    def test_no_soliton_no_crash(self):
        """Sim with no soliton placed should run without NaN."""
        cfg = SimulationConfig(grid_size=16, e_amplitude=0.0)
        sim = Simulation(cfg)
        sim.run(steps=10)
        assert not np.any(np.isnan(sim.chi))

    def test_complex_field_no_nan(self):
        cfg = SimulationConfig(grid_size=16, field_level=FieldLevel.COMPLEX)
        sim = Simulation(cfg)
        sim.place_soliton((8, 8, 8), amplitude=2.0)
        sim.run(steps=10)
        assert not np.any(np.isnan(sim.chi))

    def test_color_field_no_nan(self):
        cfg = SimulationConfig(grid_size=16, field_level=FieldLevel.COLOR)
        sim = Simulation(cfg)
        sim.place_soliton((8, 8, 8), amplitude=2.0)
        sim.run(steps=10)
        assert not np.any(np.isnan(sim.chi))

    def test_metrics_dict_has_expected_keys(self):
        cfg = SimulationConfig(grid_size=16)
        sim = Simulation(cfg)
        sim.place_soliton((8, 8, 8), amplitude=2.0)
        sim.run(steps=5)
        m = sim.metrics()
        assert "chi_min" in m
        assert "chi_max" in m
        assert "energy_total" in m


# ── Power spectrum edge cases ────────────────────────────────────────

class TestPowerSpectrumEdgeCases:
    def test_uniform_field(self):
        """Uniform field → all power at k=0 (DC excluded from bins)."""
        field = np.ones((16, 16, 16)) * 5.0
        result = power_spectrum(field, bins=8)
        assert "k" in result
        assert "power" in result
        # Non-DC bins should be near zero
        assert np.all(result["power"] < 1e-10)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            power_spectrum(np.ones((16, 16)))

    def test_single_bin(self):
        field = np.random.default_rng(42).standard_normal((8, 8, 8))
        result = power_spectrum(field, bins=1)
        assert result["k"].shape == (1,)


# ── Color variance edge cases ───────────────────────────────────────

class TestColorVarianceEdgeCases:
    def test_zero_field(self):
        """When all Ψ=0, f_c should be 0 (safe division)."""
        z = np.zeros((3, 8, 8, 8))
        result = color_variance(z, z, n_colors=3)
        assert result["f_c_mean"] == pytest.approx(0.0)

    def test_equal_colors(self):
        """Equal amplitudes across colors → f_c = 0 (singlet)."""
        rng = np.random.default_rng(42)
        amp = rng.standard_normal((8, 8, 8))
        psi_r = np.stack([amp, amp, amp], axis=0)
        psi_i = np.zeros_like(psi_r)
        result = color_variance(psi_r, psi_i, n_colors=3)
        assert result["f_c_mean"] == pytest.approx(0.0, abs=1e-12)

    def test_single_color_dominant(self):
        """One color only → f_c = 2/3."""
        psi_r = np.zeros((3, 8, 8, 8))
        psi_r[0] = 1.0  # Only color 0 active
        psi_i = np.zeros_like(psi_r)
        result = color_variance(psi_r, psi_i, n_colors=3)
        assert result["f_c_mean"] == pytest.approx(2 / 3, rel=1e-10)

    def test_flat_input(self):
        """1-D flat layout should work too (matches GPU flatten convention)."""
        N = 8
        total = N ** 3
        psi_r = np.zeros(3 * total)
        psi_r[:total] = 1.0  # color 0 only
        psi_i = np.zeros_like(psi_r)
        result = color_variance(psi_r, psi_i, n_colors=3)
        assert result["f_c_mean"] == pytest.approx(2 / 3, rel=1e-10)


# ── Tracker edge cases ──────────────────────────────────────────────

class TestFlattenTrajectories:
    def test_empty(self):
        result = flatten_trajectories([])
        assert result["step"].shape == (0,)

    def test_single_entry(self):
        traj = [[{"step": 10.0, "x": 1.0, "y": 2.0, "z": 3.0, "amplitude": 0.5}]]
        result = flatten_trajectories(traj)
        assert result["step"][0] == 10.0
        assert len(result["amplitude"]) == 1
