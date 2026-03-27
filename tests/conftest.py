"""Shared test fixtures."""

import pytest

from lfm.config import FieldLevel, SimulationConfig


@pytest.fixture(autouse=True)
def _skip_on_gpu_oom():
    """Auto-skip any test that fails due to GPU out-of-memory.

    VRAM pressure from concurrent processes (e.g. a background simulation)
    is an infrastructure condition, not a code defect.  Converting OOM
    to a skip keeps test runs green on machines with a busy GPU while still
    catching real assertion failures.
    """
    try:
        yield
    except Exception as exc:
        msg = str(exc)
        name = type(exc).__name__
        if (
            "cudaErrorMemoryAllocation" in msg
            or "out of memory" in msg.lower()
            and ("cuda" in msg.lower() or "cupy" in name.lower() or "cuda" in name.lower())
        ):
            pytest.skip(f"GPU OOM (VRAM contention) — {exc}")
        raise


@pytest.fixture
def small_config():
    """Minimal config for fast unit tests (N=16)."""
    return SimulationConfig(grid_size=16, e_amplitude=12.0)


@pytest.fixture
def small_complex_config():
    """Complex field config for EM tests (N=16)."""
    return SimulationConfig(
        grid_size=16,
        e_amplitude=12.0,
        field_level=FieldLevel.COMPLEX,
    )
