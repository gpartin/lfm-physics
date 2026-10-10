"""Base installs may import the complete visualization API without matplotlib."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

IMPORT_WITHOUT_MATPLOTLIB = r"""
import importlib
import importlib.abc
import pkgutil
import sys

class NoMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib deliberately unavailable", name=fullname)

sys.meta_path.insert(0, NoMatplotlib())
import lfm
import lfm.viz
for module in pkgutil.walk_packages(lfm.viz.__path__, lfm.viz.__name__ + "."):
    importlib.import_module(module.name)
assert not any(name.startswith("matplotlib") for name in sys.modules)
sim = lfm.Simulation(lfm.SimulationConfig(grid_size=8), backend="cpu")
sim.run(steps=1)
from lfm.viz.gravity_recovery import (
    plot_energy_history, plot_profile_comparison, plot_stability_sweep,
)
for plot, args in [
    (plot_energy_history, ([0, 1], [1, 1])),
    (plot_profile_comparison, ([1], [0], [1], [1])),
    (plot_stability_sweep, ([0.01], [1], [True])),
]:
    try:
        plot(*args, title="test", path="unused.png")
    except ImportError as exc:
        assert "lfm-physics[viz]" in str(exc), str(exc)
    else:
        raise AssertionError("plot should require the visualization extra")
"""


def test_base_import_without_matplotlib():
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITHOUT_MATPLOTLIB],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["energy_history", "profile_comparison", "stability_sweep"])
def test_gravity_recovery_plots_with_extra(name, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from lfm.viz import gravity_recovery

    values = np.array([1.0, 2.0])
    args = {
        "energy_history": (values, values),
        "profile_comparison": (values, values, values, values),
        "stability_sweep": (values, values, np.array([True, False])),
    }
    output = tmp_path / f"{name}.png"
    result = getattr(gravity_recovery, f"plot_{name}")(
        *args[name],
        title="Regression",
        path=output,
    )
    assert result == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
