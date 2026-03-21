"""Run a simulation across a range of parameter values."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from lfm.config import SimulationConfig
from lfm.simulation import Simulation


def sweep(
    config: SimulationConfig,
    param: str,
    values: list[Any],
    steps: int,
    metric_names: list[str] | None = None,
    *,
    equilibrate: bool = True,
    soliton: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run *steps* for each value in *values*, recording final metrics.

    Parameters
    ----------
    config : SimulationConfig
        Base configuration (will be copied for each run).
    param : str
        Name of a ``SimulationConfig`` attribute to vary
        (e.g. ``"kappa"``, ``"grid_size"``).
    values : list
        Parameter values to sweep over.
    steps : int
        Steps per run.
    metric_names : list[str] or None
        Which metrics to record.  ``None`` → all.
    equilibrate : bool
        Call ``sim.equilibrate()`` before running.
    soliton : dict or None
        If provided, place a soliton with these kwargs before each run.
        Example: ``{"position": (32,32,32), "amplitude": 6.0}``.

    Returns
    -------
    list[dict]
        One dict per value with ``{param: value, **metrics}``.
    """
    results: list[dict[str, Any]] = []

    for val in values:
        cfg = deepcopy(config)

        # Handle grid_size specially — position may need adjusting
        setattr(cfg, param, val)

        sim = Simulation(cfg)

        if soliton is not None:
            kw = dict(soliton)
            pos = kw.pop("position", None)
            if pos is None:
                n = cfg.grid_size
                pos = (n // 2, n // 2, n // 2)
            sim.place_soliton(pos, **kw)

        if equilibrate:
            sim.equilibrate()

        sim.run(steps=steps, record_metrics=False)
        m = sim.metrics()

        row: dict[str, Any] = {param: val}
        if metric_names is not None:
            for k in metric_names:
                row[k] = m.get(k)
        else:
            row.update(m)
        results.append(row)

    return results
