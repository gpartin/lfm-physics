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
            snap.append(
                {
                    "step": float(sim.step),
                    "x": float(pk[0]),
                    "y": float(pk[1]),
                    "z": float(pk[2]),
                    "amplitude": float(ed[pk[0], pk[1], pk[2]]),
                }
            )
        trajectories.append(snap)

    return trajectories


def flatten_trajectories(
    trajectories: list[list[dict[str, float]]],
) -> dict[str, NDArray]:
    """Flatten snapshot list into arrays keyed by step/x/y/z/amplitude."""
    all_entries = [e for snap in trajectories for e in snap]
    if not all_entries:
        empty = np.empty(0)
        return {"step": empty, "x": empty, "y": empty, "z": empty, "amplitude": empty}
    return {k: np.array([e[k] for e in all_entries]) for k in ("step", "x", "y", "z", "amplitude")}


# ---------------------------------------------------------------------------
# Collision event detection (P012)
# ---------------------------------------------------------------------------


def detect_collision_events(
    trajectories: list[list[dict[str, float]]],
    min_sep: float = 3.0,
) -> list[dict]:
    """Detect approach and merge collision events in peak trajectories.

    Scans every snapshot and consecutive pair of snapshots produced by
    :func:`track_peaks` for two kinds of events:

    * ``"approach"`` — a pair of peaks whose centre-to-centre distance falls
      below *min_sep* for the first time.
    * ``"merge"`` — a peak present in snapshot *k* has no counterpart within
      *min_sep* in snapshot *k+1* (the peak disappeared, implying a merge).

    Parameters
    ----------
    trajectories : list[list[dict]]
        Output of :func:`track_peaks`.  Each inner list is one snapshot;
        dicts contain ``step``, ``x``, ``y``, ``z``, ``amplitude``.
    min_sep : float
        Distance threshold (grid cells) for classifying an approach or merge.

    Returns
    -------
    list[dict]
        Each dict has keys:

        * ``"type"``  – ``"approach"`` or ``"merge"``
        * ``"step"``  – simulation step at which the event occurred
        * ``"i"``     – index of first peak in the snapshot
        * ``"j"``     – index of second peak (``-1`` for merge events)
        * ``"sep"``   – centre-to-centre separation at the event step
    """
    events: list[dict] = []
    n_snaps = len(trajectories)

    # Track which (i, j) pairs have already been flagged as "approach" so we
    # only record the first occurrence for each pair.
    approached: set[tuple[int, int]] = set()

    def _pos(p: dict) -> NDArray:
        return np.array([p["x"], p["y"], p["z"]])

    for snap_idx, snap in enumerate(trajectories):
        n = len(snap)
        # Approach detection: any close pair in this snapshot
        for i in range(n):
            for j in range(i + 1, n):
                sep = float(np.linalg.norm(_pos(snap[i]) - _pos(snap[j])))
                if sep < min_sep:
                    key = (i, j)
                    if key not in approached:
                        approached.add(key)
                        events.append(
                            {
                                "type": "approach",
                                "step": snap[i]["step"],
                                "i": i,
                                "j": j,
                                "sep": sep,
                            }
                        )

    # Merge detection: peak in snap k has no match in snap k+1
    for snap_idx in range(n_snaps - 1):
        snap_now = trajectories[snap_idx]
        snap_next = trajectories[snap_idx + 1]
        for i, pk in enumerate(snap_now):
            pos = _pos(pk)
            if not snap_next:
                # All peaks vanished
                events.append(
                    {
                        "type": "merge",
                        "step": pk["step"],
                        "i": i,
                        "j": -1,
                        "sep": float("nan"),
                    }
                )
                continue
            min_dist = min(float(np.linalg.norm(pos - _pos(q))) for q in snap_next)
            if min_dist > min_sep:
                events.append(
                    {
                        "type": "merge",
                        "step": pk["step"],
                        "i": i,
                        "j": -1,
                        "sep": min_dist,
                    }
                )

    return events


def compute_impact_parameter(
    traj_i: dict[str, NDArray],
    traj_j: dict[str, NDArray],
) -> float:
    """Minimum perpendicular distance between two straight-line trajectory fits.

    Fits a straight line through the positions of each trajectory (least
    squares) and returns the closest-approach distance between the two
    resulting infinite lines in 3D.

    Parameters
    ----------
    traj_i : dict
        Flattened trajectory for particle *i* as returned by
        :func:`flatten_trajectories` (subset for one particle, or the full
        dict if only two peaks are being tracked).
    traj_j : dict
        Flattened trajectory for particle *j*.

    Returns
    -------
    float
        Impact parameter (grid cells).  Returns ``nan`` if either trajectory
        has fewer than 2 points.
    """

    def _fit_line(traj: dict[str, NDArray]):
        pts = np.column_stack([traj["x"], traj["y"], traj["z"]])
        if len(pts) < 2:
            return None, None
        t = traj["step"].astype(float)
        t -= t.mean()
        # Fit each coordinate independently vs time
        direction = np.empty(3)
        origin = np.empty(3)
        for dim, key in enumerate(("x", "y", "z")):
            coords = traj[key].astype(float)
            m, b = np.polyfit(t, coords, 1)
            direction[dim] = m
            origin[dim] = b
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return origin, None  # stationary
        return origin, direction / norm

    o1, d1 = _fit_line(traj_i)
    o2, d2 = _fit_line(traj_j)

    if o1 is None or o2 is None:
        return float("nan")

    if d1 is None and d2 is None:
        return float(np.linalg.norm(o1 - o2))

    if d1 is None:
        # o1 is a point; distance from point to line(o2, d2)
        w = o1 - o2
        return float(np.linalg.norm(w - np.dot(w, d2) * d2))

    if d2 is None:
        w = o2 - o1
        return float(np.linalg.norm(w - np.dot(w, d1) * d1))

    # Distance between two infinite lines in 3D
    w = o1 - o2
    cross = np.cross(d1, d2)
    denom = np.linalg.norm(cross)
    if denom < 1e-12:
        # Lines are parallel — perpendicular distance
        return float(np.linalg.norm(w - np.dot(w, d1) * d1))
    return float(abs(np.dot(w, cross)) / denom)
