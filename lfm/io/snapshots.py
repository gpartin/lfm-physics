"""
Snapshot serialization for offline visualization replay.
=========================================================

Save and load snapshot lists produced by
:meth:`~lfm.Simulation.run_with_snapshots` as portable ``.npz`` archives.

Typical workflow::

    import lfm
    from lfm.io import save_snapshots, load_snapshots
    from lfm.viz import animate_3d_slices

    # --- run ---
    sim = lfm.Simulation(lfm.SimulationConfig(grid_size=64))
    snaps = sim.run_with_snapshots(10_000, snapshot_every=200,
                                   fields=["energy_density"])
    save_snapshots(snaps, "wave_run.npz")

    # --- replay later ---
    snaps = load_snapshots("wave_run.npz")
    anim = animate_3d_slices(snaps, field="energy_density")
    anim.save("wave_slices.gif", writer="pillow", fps=12)

Format
------
The ``.npz`` archive contains:

``steps`` (int32 array)
    Step index for each snapshot.
``n_snapshots`` (int32 scalar)
    Total number of snapshots.
``<field_name>`` (float32 array, shape ``(n_snapshots, N, N, N)``)
    One entry per field requested at run time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_snapshots(
    snaps: list[dict],
    path: str | Path,
    *,
    compress: bool = True,
) -> Path:
    """Save a list of snapshots to a compressed NumPy archive.

    Parameters
    ----------
    snaps : list of dict
        Snapshots returned by :meth:`~lfm.Simulation.run_with_snapshots`.
        Each dict must have at least one array-valued key; the ``"step"``
        key (int) is stored separately if present.
    path : str or Path
        Output file path.  A ``.npz`` suffix is appended if absent.
    compress : bool
        Use ``np.savez_compressed`` (default) instead of plain ``np.savez``.
        Reduces file size by ~4\u00d7 for typical float32 fields.

    Returns
    -------
    Path
        Resolved path of the written file.

    Examples
    --------
    >>> from lfm.io import save_snapshots
    >>> save_snapshots(snaps, "double_slit_v1.npz")
    PosixPath('/abs/path/double_slit_v1.npz')
    """
    if not snaps:
        raise ValueError("snaps list is empty — nothing to save")

    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")

    # Field names = all keys that are numpy arrays (exclude "step")
    fields = [k for k, v in snaps[0].items() if isinstance(v, np.ndarray) and k != "step"]

    data: dict[str, np.ndarray] = {
        "n_snapshots": np.int32(len(snaps)),
        "steps": np.array(
            [int(s.get("step", i)) for i, s in enumerate(snaps)],
            dtype=np.int32,
        ),
    }
    for f in fields:
        data[f] = np.stack([s[f] for s in snaps], axis=0)  # (T, *shape)

    saver = np.savez_compressed if compress else np.savez
    saver(str(path), **data)
    return path.resolve()


def load_snapshots(path: str | Path) -> list[dict]:
    """Load snapshots from a ``.npz`` archive written by :func:`save_snapshots`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file.

    Returns
    -------
    list of dict
        Same structure as the original list passed to :func:`save_snapshots`.
        Each dict contains a ``"step"`` int and one entry per saved field.

    Examples
    --------
    >>> from lfm.io import load_snapshots
    >>> snaps = load_snapshots("double_slit_v1.npz")
    >>> print(snaps[0].keys())
    dict_keys(['step', 'energy_density'])
    """
    archive = np.load(str(Path(path)), allow_pickle=False)
    n = int(archive["n_snapshots"])
    steps = archive["steps"]
    fields = [k for k in archive if k not in ("steps", "n_snapshots")]
    return [{"step": int(steps[i]), **{f: archive[f][i] for f in fields}} for i in range(n)]


__all__ = ["save_snapshots", "load_snapshots"]
