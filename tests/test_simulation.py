"""Tests for lfm.simulation facade."""

import numpy as np
import pytest

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import CHI0
from lfm.simulation import Simulation


N = 16  # small grid for fast tests


def _small_config(**kwargs):
    """Helper to create a small config suitable for quick tests."""
    defaults = dict(
        grid_size=N,
        field_level=FieldLevel.REAL,
        boundary_type=BoundaryType.FROZEN,
        report_interval=100,
        dt=0.02,
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


# ──── Construction ────


class TestConstruction:
    def test_default_config(self):
        sim = Simulation()
        assert sim.config.grid_size == 128  # default
        assert sim.step == 0

    def test_custom_config(self):
        cfg = _small_config()
        sim = Simulation(cfg)
        assert sim.config.grid_size == N

    def test_cpu_backend(self):
        sim = Simulation(_small_config(), backend="cpu")
        assert sim.step == 0

    def test_step_starts_at_zero(self):
        sim = Simulation(_small_config())
        assert sim.step == 0

    def test_empty_history(self):
        sim = Simulation(_small_config())
        assert sim.history == []


# ──── Field access ────


class TestFieldAccess:
    def test_get_chi_shape(self):
        sim = Simulation(_small_config())
        chi = sim.get_chi()
        assert chi.shape == (N, N, N)

    def test_get_chi_initial_value(self):
        sim = Simulation(_small_config())
        chi = sim.get_chi()
        np.testing.assert_allclose(chi, CHI0, atol=1e-5)

    def test_get_psi_real_shape(self):
        sim = Simulation(_small_config())
        pr = sim.get_psi_real()
        assert pr.shape == (N, N, N)

    def test_get_psi_imag_none_for_real_level(self):
        sim = Simulation(_small_config(field_level=FieldLevel.REAL))
        pi = sim.get_psi_imag()
        assert pi is None

    def test_get_psi_imag_shape_for_complex(self):
        sim = Simulation(_small_config(field_level=FieldLevel.COMPLEX))
        pi = sim.get_psi_imag()
        assert pi is not None
        assert pi.shape == (N, N, N)

    def test_get_energy_density_shape(self):
        sim = Simulation(_small_config())
        ed = sim.get_energy_density()
        assert ed.shape == (N, N, N)

    def test_energy_density_nonnegative(self):
        sim = Simulation(_small_config())
        ed = sim.get_energy_density()
        assert np.all(ed >= 0)


# ──── Soliton placement ────


class TestSolitonPlacement:
    def test_place_single_soliton(self):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        pr = sim.get_psi_real()
        assert pr.max() > 0  # soliton placed

    def test_place_soliton_adds_to_field(self):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=2.0, sigma=2.0)
        val1 = sim.get_psi_real().max()
        sim.place_soliton((N // 4, N // 4, N // 4), amplitude=2.0, sigma=2.0)
        val2 = sim.get_psi_real().max()
        # Max should be at least as large (could be same if centers far apart)
        assert val2 >= val1 - 1e-6

    def test_place_soliton_with_phase_complex(self):
        sim = Simulation(_small_config(field_level=FieldLevel.COMPLEX))
        # Phase = pi → psi_real negative, psi_imag ≈ 0
        sim.place_soliton(
            (N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0, phase=np.pi
        )
        pr = sim.get_psi_real()
        assert pr.min() < -1.0

    def test_place_multiple_solitons(self):
        sim = Simulation(_small_config())
        positions = [(4, 4, 4), (12, 12, 12)]
        sim.place_solitons(positions, amplitude=3.0, sigma=2.0)
        pr = sim.get_psi_real()
        assert pr.max() > 0


# ──── Equilibration ────


class TestEquilibration:
    def test_equilibrate_modifies_chi(self):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=5.0, sigma=2.0)
        chi_before = sim.get_chi().copy()
        sim.equilibrate()
        chi_after = sim.get_chi()
        assert not np.allclose(chi_before, chi_after)

    def test_equilibrated_chi_has_wells(self):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=8.0, sigma=2.0)
        sim.equilibrate()
        chi = sim.get_chi()
        assert chi.min() < CHI0 - 0.1  # well formed


# ──── Evolution ────


class TestEvolution:
    def test_evolve_advances_step(self):
        sim = Simulation(_small_config(report_interval=50))
        sim.run(100)
        assert sim.step == 100

    def test_evolve_with_soliton(self):
        sim = Simulation(_small_config(report_interval=50))
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        sim.equilibrate()
        sim.run(50)
        assert sim.step == 50

    def test_evolve_records_history(self):
        sim = Simulation(_small_config(report_interval=50))
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        sim.run(100, record_metrics=True)
        # report_interval=50 → callbacks at step 50, 100
        assert len(sim.history) == 2

    def test_evolve_no_metrics(self):
        sim = Simulation(_small_config(report_interval=50))
        sim.run(100, record_metrics=False)
        assert len(sim.history) == 0

    def test_callback_called(self):
        calls = []

        def cb(sim, step):
            calls.append(step)

        sim = Simulation(_small_config(report_interval=50))
        sim.run(100, callback=cb)
        assert len(calls) == 2
        assert calls[0] == 50
        assert calls[1] == 100

    def test_evolve_twice_accumulates(self):
        sim = Simulation(_small_config(report_interval=50))
        sim.run(50)
        sim.run(50)
        assert sim.step == 100


# ──── Metrics ────


class TestMetrics:
    def test_metrics_returns_dict(self):
        sim = Simulation(_small_config())
        m = sim.metrics()
        assert isinstance(m, dict)

    def test_metrics_has_energy_keys(self):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        m = sim.metrics()
        for key in [
            "energy_kinetic",
            "energy_gradient",
            "energy_potential",
            "energy_total",
        ]:
            assert key in m

    def test_metrics_has_chi_keys(self):
        sim = Simulation(_small_config())
        m = sim.metrics()
        for key in ["chi_min", "chi_max", "chi_mean", "chi_std"]:
            assert key in m

    def test_metrics_has_structure_keys(self):
        sim = Simulation(_small_config())
        m = sim.metrics()
        for key in ["well_fraction", "void_fraction", "n_clusters"]:
            assert key in m

    def test_total_energy_scalar(self):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        e = sim.total_energy()
        assert isinstance(e, float)
        assert e > 0

    def test_history_step_key(self):
        sim = Simulation(_small_config(report_interval=50))
        sim.run(50)
        assert len(sim.history) == 1
        assert "step" in sim.history[0]
        assert sim.history[0]["step"] == 50.0


# ──── Interior mask caching ────


class TestInteriorMask:
    def test_mask_cached(self):
        sim = Simulation(_small_config())
        m1 = sim.get_interior_mask()
        m2 = sim.get_interior_mask()
        assert m1 is m2  # same object

    def test_mask_shape(self):
        sim = Simulation(_small_config())
        m = sim.get_interior_mask()
        assert m.shape == (N, N, N)
        assert m.dtype == bool


# ──── Complex / Color levels ────


class TestFieldLevels:
    def test_complex_simulation(self):
        sim = Simulation(_small_config(field_level=FieldLevel.COMPLEX))
        sim.place_soliton(
            (N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0, phase=0.5
        )
        sim.equilibrate()
        sim.run(10, record_metrics=False)
        assert sim.step == 10

    def test_color_simulation(self):
        cfg = _small_config(field_level=FieldLevel.COLOR, n_colors=3)
        sim = Simulation(cfg)
        positions = [(N // 3, N // 3, N // 3), (2 * N // 3, 2 * N // 3, 2 * N // 3)]
        sim.place_solitons(positions, amplitude=3.0, sigma=2.0)
        sim.run(10, record_metrics=False)
        assert sim.step == 10


# ──── Top-level import ────


class TestTopLevelImport:
    def test_simulation_importable_from_lfm(self):
        import lfm

        assert hasattr(lfm, "Simulation")

    def test_construct_via_lfm(self):
        import lfm

        sim = lfm.Simulation(lfm.SimulationConfig(grid_size=N))
        assert sim.step == 0


# ──── Checkpoint / Resume ────


class TestCheckpoint:
    def test_save_creates_file(self, tmp_path):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        sim.equilibrate()
        sim.run(10, record_metrics=False)

        path = tmp_path / "ckpt.npz"
        sim.save_checkpoint(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_round_trip_preserves_step(self, tmp_path):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        sim.run(20, record_metrics=False)

        path = tmp_path / "ckpt.npz"
        sim.save_checkpoint(path)
        loaded = Simulation.load_checkpoint(path)
        assert loaded.step == sim.step

    def test_round_trip_preserves_chi(self, tmp_path):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        sim.equilibrate()
        sim.run(10, record_metrics=False)

        path = tmp_path / "ckpt.npz"
        sim.save_checkpoint(path)
        loaded = Simulation.load_checkpoint(path)
        np.testing.assert_allclose(loaded.get_chi(), sim.get_chi(), atol=1e-6)

    def test_round_trip_preserves_psi_real(self, tmp_path):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        sim.equilibrate()
        sim.run(10, record_metrics=False)

        path = tmp_path / "ckpt.npz"
        sim.save_checkpoint(path)
        loaded = Simulation.load_checkpoint(path)
        np.testing.assert_allclose(
            loaded.get_psi_real(), sim.get_psi_real(), atol=1e-6,
        )

    def test_round_trip_preserves_config(self, tmp_path):
        cfg = _small_config(chi0=18.5, kappa=0.02)
        sim = Simulation(cfg)
        sim.run(10, record_metrics=False)

        path = tmp_path / "ckpt.npz"
        sim.save_checkpoint(path)
        loaded = Simulation.load_checkpoint(path)
        assert loaded.config.grid_size == cfg.grid_size
        assert loaded.config.chi0 == pytest.approx(cfg.chi0)
        assert loaded.config.kappa == pytest.approx(cfg.kappa)

    def test_round_trip_preserves_history(self, tmp_path):
        sim = Simulation(_small_config(report_interval=5))
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0)
        sim.run(10)  # record_metrics=True by default

        path = tmp_path / "ckpt.npz"
        sim.save_checkpoint(path)
        loaded = Simulation.load_checkpoint(path)
        assert len(loaded.history) == len(sim.history)

    def test_complex_field_round_trip(self, tmp_path):
        cfg = _small_config(field_level=FieldLevel.COMPLEX)
        sim = Simulation(cfg)
        sim.place_soliton(
            (N // 2, N // 2, N // 2), amplitude=3.0, sigma=2.0, phase=0.5,
        )
        sim.equilibrate()
        sim.run(10, record_metrics=False)

        path = tmp_path / "ckpt.npz"
        sim.save_checkpoint(path)
        loaded = Simulation.load_checkpoint(path)
        assert loaded.config.field_level == FieldLevel.COMPLEX
        np.testing.assert_allclose(
            loaded.get_psi_imag(), sim.get_psi_imag(), atol=1e-6,
        )
