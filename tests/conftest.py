"""Shared test fixtures."""

import pytest

from lfm.config import FieldLevel, SimulationConfig


@pytest.fixture
def small_config():
    """Minimal config for fast unit tests (N=16)."""
    return SimulationConfig(grid_size=16, e_amplitude=12.0)


@pytest.fixture
def small_complex_config():
    """Complex field config for EM tests (N=16)."""
    return SimulationConfig(
        grid_size=16, e_amplitude=12.0, field_level=FieldLevel.COMPLEX,
    )
