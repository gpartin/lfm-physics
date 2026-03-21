"""Track energy-density peaks across simulation steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from lfm.simulation import Simulation


def track_peaks(
    sim: Simulation,
    steps: int,
    interval: int,
    n_peaks: int = 5,
    min_separation: int = 3,
) -> list[list[dict[str, float]]]:
    """Run a simulation and record peak positions at regular intervals.

    Parameters
    ----------
    sim : Simulation
        An initialised (and optionally equilibrated) simulation.
    steps : int
        Total number of steps to run.
    interval : int
        Record peaks every *interval* steps.
    n_peaks : int
        Maximum number of peaks to track per snapshot.
    min_separation : int
        Minimum distance between detected peaks.

    Returns
    -------
    trajectories : list[list[dict]]
        One inner list per snapshot. Each entry is a dict with keys
        ``step``, ``x``, ``y``, ``z``, ``amplitude``.
    """
    from lfm.analysis.observables import find_peaks

    trajectories: list[list[dict[str, float]]] = []
    remaining = steps

    while remaining > 0:
        chunk = min(interval, remaining)
        sim.run(steps=chunk, record_metrics=False)
        remaining -= chunk

        ed = sim.energy_density
        peaks = find_peaks(ed, n=n_peaks, min_separation=min_separation)
        snap: list[dict[str, float]] = []
        for pk in peaks:
            snap.append({
                "step": float(sim.step),
                "x": float(pk[0]),
                "y": float(pk[1]),
                "z": float(pk[2]),
                "amplitude": float(ed[pk[0], pk[1], pk[2]]),
            })
        trajectories.append(snap)

    return trajectories


def flatten_trajectories(
    trajectories: list[list[dict[str, float]]],
) -> dict[str, NDArray]:
    """Flatten snapshot list into arrays keyed by step/x/y/z/amplitude."""
    all_entries = [e for snap in trajectories for e in snap]
    if not all_entries:
        empty = np.empty(0)
        return {"step": empty, "x": empty, "y": empty, "z": empty,
                "amplitude": empty}
    return {
        k: np.array([e[k] for e in all_entries])
        for k in ("step", "x", "y", "z", "amplitude")
    }
