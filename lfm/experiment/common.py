"""
Shared experiment utilities for LFM experiments.
=================================================

Provides the common infrastructure used by :mod:`~lfm.experiment.double_slit`,
:mod:`~lfm.experiment.collision`, and future experiment modules.

This module contains GPU-optimised snapshot capture, simulation setup
helpers, and the save/plot result infrastructure that every experiment
shares.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np


# ── Slice utilities ────────────────────────────────────────────────────────


def midplane_slice(
    field_3d: np.ndarray,
    collision_axis: int,
) -> np.ndarray:
    """Return a 2D slice through the midplane *perpendicular* to a
    transverse axis so that the collision axis is visible.

    Parameters
    ----------
    field_3d : ndarray, shape (N, N, N)
        A 3D field (energy_density, chi, etc.).
    collision_axis : int
        The axis along which the particles travel (0, 1, or 2).

    Returns
    -------
    ndarray, shape (N, N) — 2D plane showing both particles along the
    collision direction.
    """
    try:
        import cupy

        if isinstance(field_3d, cupy.ndarray):
            field_3d = cupy.asnumpy(field_3d)
    except ImportError:
        pass

    N = field_3d.shape[0]
    t1 = (collision_axis + 1) % 3  # first transverse axis
    slc: list[slice | int] = [slice(None)] * 3
    slc[t1] = N // 2
    return field_3d[tuple(slc)]


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    import lfm as _lfm_t

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "build_sim",
    "gpu_snapshot_loop",
]


# ── Experiment configuration ───────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Common configuration shared across all experiment types.

    Subclass or compose this for experiment-specific geometry.
    """

    N: int = 64
    chi0: float = 19.0
    kappa: float | None = None
    dt: float = 0.02
    total_steps: int = 20_000
    snap_every: int = 200
    record_every: int = 10
    boundary_type: str = "absorbing"
    boundary_fraction: float = 0.15
    field_level: str = "complex"
    evolve_chi: bool = True


# ── Simulation builder ─────────────────────────────────────────────────────


def build_sim(cfg: ExperimentConfig) -> "_lfm_t.Simulation":
    """Create a :class:`~lfm.Simulation` from an :class:`ExperimentConfig`.

    This centralises the pattern used by every experiment so that backend
    selection (CPU / CuPy / …) and boundary setup are identical everywhere.
    """
    import lfm

    fl_map = {
        "real": lfm.FieldLevel.REAL,
        "complex": lfm.FieldLevel.COMPLEX,
        "color": lfm.FieldLevel.COLOR,
    }
    fl = fl_map.get(cfg.field_level, lfm.FieldLevel.COMPLEX)

    bt_map = {
        "absorbing": lfm.BoundaryType.ABSORBING,
        "frozen": lfm.BoundaryType.FROZEN,
        "periodic": lfm.BoundaryType.PERIODIC,
    }
    bt = bt_map.get(cfg.boundary_type, lfm.BoundaryType.ABSORBING)

    kw: dict = dict(
        grid_size=cfg.N,
        field_level=fl,
        boundary_type=bt,
        chi0=cfg.chi0,
    )
    if cfg.kappa is not None:
        kw["kappa"] = cfg.kappa
    if bt == lfm.BoundaryType.ABSORBING:
        kw["boundary_fraction"] = cfg.boundary_fraction

    return lfm.Simulation(lfm.SimulationConfig(**kw))


# ── GPU-direct snapshot loop ───────────────────────────────────────────────


def gpu_snapshot_loop(
    sim: "_lfm_t.Simulation",
    *,
    total_steps: int,
    snap_every: int = 200,
    fields: list[str] | None = None,
    movie_every: int | None = None,
    movie_fields: list[str] | None = None,
    step_callback: Callable[["_lfm_t.Simulation", int], None] | None = None,
    verbose: bool = True,
    label: str = "experiment",
    evolve_chi: bool = True,
    metrics_every: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run the simulation using the GPU-direct evolver loop.

    This avoids the N³/step PCIe overhead of ``run_with_snapshots`` when a
    ``step_callback`` is present.  Field snapshots are captured every
    *snap_every* steps by copying only the requested fields.

    Parameters
    ----------
    sim
        An initialised :class:`~lfm.Simulation`.
    total_steps
        Number of leapfrog steps to run.
    snap_every
        Capture a full field snapshot every this many steps.
    fields
        Which fields to capture (default ``["energy_density", "chi"]``).
    movie_every : int or None
        If set, capture dense movie frames at this interval.
    movie_fields : list[str] or None
        Fields for movie frames (default ``["psi_real"]``).
    step_callback
        Called every step (e.g. barrier enforcement).  ``None`` is fine.
    verbose
        Print progress every 10 %.
    label
        Label for progress messages.
    evolve_chi
        If ``False``, freeze χ (GOV-01 only, ~40 % speedup).
    metrics_every
        If set, record lightweight metrics dicts at this interval.
        Metrics capture: step, chi_min, chi_max, energy_total, psi_max.

    Returns
    -------
    snapshots : list[dict]
        Field snapshots (heavy — full 3-D arrays).
    metrics : list[dict]
        Lightweight scalar metrics (empty if *metrics_every* is ``None``).
    movie_snapshots : list[dict]
        Dense movie frames (empty if *movie_every* is ``None``).
    """
    if fields is None:
        fields = ["energy_density", "chi"]

    _mfields = movie_fields if movie_fields is not None else ["psi_real"]

    # Auto-equilibrate if solitons were placed without calling equilibrate()
    sim._auto_equilibrate()

    evolver = sim._evolver
    snaps: list[dict] = []
    metrics: list[dict] = []
    movie_snaps: list[dict] = []
    t0 = time.perf_counter()

    # Capture initial frame for movie
    if movie_every:
        msnap: dict = {"step": 0}
        for mf in _mfields:
            if mf == "psi_real":
                msnap["psi_real"] = sim.psi_real.copy()
            elif mf == "chi":
                msnap["chi"] = sim.chi.copy()
            elif mf == "energy_density":
                msnap["energy_density"] = sim.energy_density.copy()
        movie_snaps.append(msnap)

    for step in range(1, total_steps + 1):
        evolver.evolve(1, freeze_chi=not evolve_chi)
        if step_callback is not None:
            step_callback(sim, step)

        if step % snap_every == 0:
            snap: dict = {"step": step}
            if "chi" in fields:
                snap["chi"] = sim.chi.copy()
            if "psi_real" in fields:
                snap["psi_real"] = sim.psi_real.copy()
            if "psi_imag" in fields and sim.psi_imag is not None:
                snap["psi_imag"] = sim.psi_imag.copy()
            if "energy_density" in fields:
                snap["energy_density"] = sim.energy_density.copy()
            snaps.append(snap)

        if metrics_every and step % metrics_every == 0:
            chi = sim.chi
            xp = np  # works for both numpy and cupy arrays
            try:
                import cupy

                if isinstance(chi, cupy.ndarray):
                    xp = cupy
            except ImportError:
                pass
            metrics.append(
                {
                    "step": step,
                    "chi_min": float(xp.min(chi)),
                    "chi_max": float(xp.max(chi)),
                    "energy_total": float(xp.sum(sim.energy_density)),
                    "psi_max": float(xp.max(xp.abs(sim.psi_real))),
                }
            )

        if movie_every and step % movie_every == 0:
            msnap = {"step": step}
            for mf in _mfields:
                if mf == "psi_real":
                    msnap["psi_real"] = sim.psi_real.copy()
                elif mf == "chi":
                    msnap["chi"] = sim.chi.copy()
                elif mf == "energy_density":
                    msnap["energy_density"] = sim.energy_density.copy()
            movie_snaps.append(msnap)

        if verbose and step % max(1, total_steps // 10) == 0:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            remaining = (total_steps - step) / rate
            print(
                f"  [{label}] step {step}/{total_steps}  "
                f"({rate:.0f} steps/s, ~{remaining:.0f}s left)",
                flush=True,
            )

    return snaps, metrics, movie_snaps


# ── Result base class ──────────────────────────────────────────────────────


@dataclass
class ExperimentResult:
    """Base result returned by high-level experiment functions.

    Provides save/plot infrastructure.  Subclass for experiment-specific
    derived physics (fringe spacing, annihilation fraction, etc.).
    """

    snapshots: list[dict]
    movie_snapshots: list[dict]
    metrics: list[dict]
    label: str
    N: int

    def save_snapshots_npz(self, path: str | Path) -> Path:
        """Save physics-run snapshots as compressed NPZ."""
        from lfm.io import save_snapshots

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_snapshots(self.snapshots, p)
        return p

    def save_movie(
        self,
        path: str | Path,
        *,
        animate_fn: Callable | None = None,
        **anim_kwargs,
    ) -> Path | None:
        """Save the 3-D movie MP4 using the provided animation function.

        Parameters
        ----------
        path
            Output file path (MP4 recommended).
        animate_fn
            The animation function to call.  Must accept ``(snapshots, **kwargs)``
            and return a ``FuncAnimation``.  If ``None`` and movie_snapshots
            is empty, returns ``None``.
        **anim_kwargs
            Forwarded to *animate_fn*.
        """
        if not self.movie_snapshots or animate_fn is None:
            return None

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            animate_fn(self.movie_snapshots, save_path=str(p), **anim_kwargs)
            return p
        except Exception as exc:
            print(f"  (movie save failed: {exc})")
            return None
