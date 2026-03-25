"""I/O: checkpoint save/load for resumable simulations.

Use these to persist simulation state to disk and continue later::

    from lfm.io import save_checkpoint, load_checkpoint
    import lfm

    sim = lfm.Simulation(lfm.SimulationConfig(grid_size=64))
    sim.run(steps=10_000)
    save_checkpoint(sim, "run_10k.npz")

    # --- later, in another session ---
    sim2 = load_checkpoint("run_10k.npz")
    sim2.run(steps=5_000)

These are thin convenience wrappers around
:py:meth:`~lfm.Simulation.save_checkpoint` /
:py:meth:`~lfm.Simulation.load_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfm.simulation import Simulation


def save_checkpoint(sim: "Simulation", path: str | Path) -> Path:
    """Save simulation state to a compressed NumPy archive.

    Persists the full simulation state (fields, step counter, config,
    metric history) to a ``.npz`` file for later resumption with
    :func:`load_checkpoint`.

    Parameters
    ----------
    sim :
        The running :class:`~lfm.Simulation` instance to checkpoint.
    path :
        Output file path.  A ``.npz`` suffix is appended if absent.

    Returns
    -------
    Path
        Resolved path of the written file.

    Examples
    --------
    >>> from lfm.io import save_checkpoint
    >>> save_checkpoint(sim, "checkpoints/step_10k.npz")
    """
    p = Path(path)
    sim.save_checkpoint(p)
    return p.resolve()


def load_checkpoint(path: str | Path, backend: str = "auto") -> "Simulation":
    """Restore a simulation from a checkpoint file.

    Parameters
    ----------
    path :
        Path to a ``.npz`` file written by :func:`save_checkpoint`.
    backend :
        Backend preference: ``'auto'``, ``'cpu'``, or ``'gpu'``.

    Returns
    -------
    :class:`~lfm.Simulation`
        Fully restored simulation, ready to continue with
        :py:meth:`~lfm.Simulation.run`.

    Examples
    --------
    >>> from lfm.io import load_checkpoint
    >>> sim = load_checkpoint("checkpoints/step_10k.npz")
    >>> sim.run(steps=5_000)
    """
    from lfm.simulation import Simulation

    return Simulation.load_checkpoint(Path(path), backend=backend)


from lfm.io.snapshots import load_snapshots, save_snapshots  # noqa: E402

__all__ = ["save_checkpoint", "load_checkpoint", "save_snapshots", "load_snapshots"]
