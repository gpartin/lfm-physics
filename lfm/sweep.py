"""Run simulations across parameter ranges."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from lfm.simulation import Simulation

if TYPE_CHECKING:
    from collections.abc import Callable

    from lfm.config import SimulationConfig


def sweep_cases(
    config: SimulationConfig,
    cases: list[dict[str, Any]],
    steps: int,
    *,
    initializer: Callable[[Simulation, dict[str, Any]], None],
    observer: Callable[[Simulation, dict[str, Any], int], dict[str, Any]],
    sample_every: int,
    backend: str = "auto",
) -> list[dict[str, Any]]:
    """Run a reproducible sweep with caller-defined substrate initial data.

    Unlike :func:`sweep`, this runner does not place a catalog soliton and does
    not equilibrate chi. It is intended for mechanism-discovery experiments
    that must begin from the native causal equations without a prepared well.

    The observer is called at step zero and after every sampling block. Each
    returned row includes all scalar case metadata plus ``sample_step``.
    Per-case configuration overrides may be supplied in a nested
    ``config_overrides`` mapping.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")

    results: list[dict[str, Any]] = []
    for case in cases:
        cfg = deepcopy(config)
        overrides = case.get("config_overrides", {})
        if not isinstance(overrides, dict):
            raise TypeError("config_overrides must be a mapping")
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"SimulationConfig has no attribute {key!r}")
            setattr(cfg, key, value)

        sim = Simulation(cfg, backend=backend)
        initializer(sim, case)
        scalar_case = {
            key: value
            for key, value in case.items()
            if key != "config_overrides" and isinstance(value, (str, int, float, bool, type(None)))
        }

        first = dict(scalar_case)
        first["sample_step"] = 0
        first.update(observer(sim, case, 0))
        results.append(first)

        completed = 0
        while completed < steps:
            block = min(sample_every, steps - completed)
            sim.run(
                steps=block,
                record_metrics=False,
                evolve_chi=bool(case.get("evolve_chi", True)),
            )
            completed += block
            row = dict(scalar_case)
            row["sample_step"] = completed
            row.update(observer(sim, case, completed))
            results.append(row)
    return results


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
