"""Run simulations across parameter ranges."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from lfm.simulation import Simulation

if TYPE_CHECKING:
    from lfm.config import SimulationConfig


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


def sweep_2d(
    config: SimulationConfig,
    param1: str,
    values1: list[Any],
    param2: str,
    values2: list[Any],
    steps: int,
    metric_names: list[str] | None = None,
    *,
    equilibrate: bool = True,
    soliton: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run *steps* for every combination of two parameters.

    Parameters
    ----------
    config : SimulationConfig
        Base configuration (copied for each run).
    param1 : str
        First parameter to vary.
    values1 : list
        Values for *param1*.
    param2 : str
        Second parameter to vary.
    values2 : list
        Values for *param2*.
    steps : int
        Steps per run.
    metric_names : list[str] or None
        Which metrics to record.  ``None`` → all.
    equilibrate : bool
        Call ``sim.equilibrate()`` before running.
    soliton : dict or None
        Soliton kwargs (see :func:`sweep`).

    Returns
    -------
    list[dict]
        One dict per (v1, v2) combination, including both param keys.
    """
    results: list[dict[str, Any]] = []

    for v1 in values1:
        for v2 in values2:
            cfg = deepcopy(config)
            setattr(cfg, param1, v1)
            setattr(cfg, param2, v2)

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

            row: dict[str, Any] = {param1: v1, param2: v2}
            if metric_names is not None:
                for k in metric_names:
                    row[k] = m.get(k)
            else:
                row.update(m)
            results.append(row)

    return results
