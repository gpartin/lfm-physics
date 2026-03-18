"""Tests for lfm.config — SimulationConfig validation and defaults."""

import pytest

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import CHI0, KAPPA


class TestDefaults:
    def test_default_config(self):
        cfg = SimulationConfig()
        assert cfg.grid_size == 128
        assert cfg.chi0 == CHI0
        assert cfg.kappa == KAPPA
        assert cfg.dt == 0.02
        assert cfg.field_level == FieldLevel.REAL
        assert cfg.boundary_type == BoundaryType.FROZEN

    def test_amplitude_lookup(self):
        """e_amplitude auto-fills from E_AMPLITUDE_BY_GRID."""
        cfg64 = SimulationConfig(grid_size=64)
        assert cfg64.e_amplitude == 6.0
        cfg128 = SimulationConfig(grid_size=128)
        assert cfg128.e_amplitude == 3.6
        cfg256 = SimulationConfig(grid_size=256)
        assert cfg256.e_amplitude == 1.8

    def test_sigma_computed(self):
        cfg = SimulationConfig(grid_size=128, blob_sigma_factor=12.0)
        assert cfg.sigma == 128 / 12.0

    def test_custom_amplitude_not_overwritten(self):
        cfg = SimulationConfig(grid_size=64, e_amplitude=10.0)
        assert cfg.e_amplitude == 10.0


class TestValidation:
    def test_grid_too_small(self):
        with pytest.raises(ValueError, match="grid_size"):
            SimulationConfig(grid_size=4)

    def test_dt_negative(self):
        with pytest.raises(ValueError, match="dt"):
            SimulationConfig(dt=-0.01)

    def test_dt_exceeds_cfl(self):
        with pytest.raises(ValueError, match="CFL"):
            SimulationConfig(dt=0.1)

    def test_kappa_negative(self):
        with pytest.raises(ValueError, match="kappa"):
            SimulationConfig(kappa=-1.0)

    def test_lambda_self_negative(self):
        with pytest.raises(ValueError, match="lambda_self"):
            SimulationConfig(lambda_self=-0.5)

    def test_boundary_fraction_out_of_range(self):
        with pytest.raises(ValueError, match="boundary_fraction"):
            SimulationConfig(boundary_fraction=1.5)


class TestFieldLevels:
    def test_real(self):
        assert FieldLevel.REAL == 0

    def test_complex(self):
        assert FieldLevel.COMPLEX == 1

    def test_color(self):
        assert FieldLevel.COLOR == 2
