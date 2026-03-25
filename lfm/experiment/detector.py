"""
Detector-screen component for LFM double-slit experiments.
===========================================================

A :class:`DetectorScreen` records the field intensity at a fixed planar
slice of the grid, accumulating an *interference pattern* over time.

Usage
-----
Register :meth:`~DetectorScreen.record` as a callback (or call it
manually) at each step you wish to sample, then read
:attr:`~DetectorScreen.pattern` when the run is complete::

    screen = DetectorScreen(sim, axis=2, position=55)

    snaps = sim.run_with_snapshots(
        steps=3000,
        snapshot_every=10,
        fields=["energy_density"],
        callback=lambda s, t: screen.record(),
    )

    pattern = screen.pattern          # shape (N, N)  — total intensity
    frames  = screen.snapshots        # shape (K, N, N) — per-frame
    clicks  = screen.click_pattern()  # discrete particle hits (optional)

Physical interpretation
-----------------------
In the wave picture :attr:`pattern` shows the continuous intensity
distribution — the classic fringe pattern visible in Young's experiment.

To simulate *particle detection* (as in a quantum experiment with a
single-particle source), call :meth:`click_pattern` which samples
discrete "clicks" from the accumulated intensity distribution.  As more
particles are passed through, the histogram of clicks converges to the
smooth interference pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from lfm.simulation import Simulation

__all__ = ["DetectorScreen"]


class DetectorScreen:
    """Records energy-density intensities at a detector plane.

    Parameters
    ----------
    sim : Simulation
        The LFM simulation to monitor.
    axis : int
        Axis perpendicular to the detector plane.  E.g. ``axis=2`` places
        the screen in the *xy*-plane at some *z*-position.  This should be
        the same propagation axis as the barrier.
    position : int or None
        Lattice index along ``axis``.  Defaults to 80 % of the grid size
        (downstream of a midpoint barrier).
    field : str
        Which field to record: ``"energy_density"`` (default) or
        ``"psi_real"`` / ``"psi_imag"`` / ``"chi"``.

    Attributes
    ----------
    pattern : ndarray, shape (N, N)
        Time-integrated field intensity at the detector plane (the
        interference pattern).
    snapshots : ndarray, shape (K, N, N)
        Every individual recorded frame.
    """

    def __init__(
        self,
        sim: "Simulation",
        axis: int = 2,
        position: int | None = None,
        field: str = "energy_density",
    ) -> None:
        N = sim.config.grid_size
        self._sim = sim
        self._axis = axis
        self._pos = position if position is not None else int(N * 0.80)
        self._field_name = field
        self._N = N
        self._frames: list[NDArray[np.float32]] = []

    # ── Recording ─────────────────────────────────────────────────────

    def record(self) -> None:
        """Sample the field at the detector plane and store the frame.

        Call this at every simulation step (or every few steps) to build
        up the interference pattern with sufficient temporal resolution.

        GPU fast-path: when the simulation is running on a CuPy backend and
        the requested field is ``energy_density`` (the common case), we access
        the live GPU buffer directly, compute the 2-D slice on the GPU, and
        copy only the N×N result to CPU.  This avoids the full N³ GPU→CPU
        transfer that ``sim.energy_density`` would trigger.
        """
        if self._field_name == "energy_density" and self._try_record_gpu_fast():
            return
        field = self._get_field()
        idx = [slice(None), slice(None), slice(None)]
        idx[self._axis] = self._pos
        frame = field[tuple(idx)].astype(np.float32)
        self._frames.append(frame.copy())

    def _try_record_gpu_fast(self) -> bool:
        """GPU fast-path for energy_density recording.

        Accesses the live GPU buffer directly, squares it, extracts the
        2-D slice on-device, then copies only that slice to CPU.  This
        reduces per-step PCIe traffic from N³×4 bytes to N²×4 bytes
        (64× reduction for N=64).

        Returns True if the fast path succeeded, False to fall through to
        the normal CPU path.
        """
        try:
            import cupy as cp
        except ImportError:
            return False
        try:
            pr_gpu = self._sim._native_psi_real()  # live (N, N, N) CuPy view
        except AttributeError:
            return False
        if not isinstance(pr_gpu, cp.ndarray):
            return False
        idx: list = [slice(None), slice(None), slice(None)]
        idx[self._axis] = self._pos
        energy_slice = (pr_gpu[tuple(idx)] ** 2).astype(cp.float32)
        self._frames.append(cp.asnumpy(energy_slice))
        return True

    def step_callback(self, sim: "Simulation", step: int) -> None:
        """Simulation step callback.  Equivalent to calling :meth:`record`.

        Register this as the *callback* argument of
        :meth:`~lfm.Simulation.run_with_snapshots`::

            sim.run_with_snapshots(
                steps=3000, snapshot_every=10,
                callback=screen.step_callback,
            )
        """
        self.record()

    def reset(self) -> None:
        """Clear all recorded frames."""
        self._frames.clear()

    # ── Derived quantities ─────────────────────────────────────────────

    @property
    def pattern(self) -> NDArray[np.float32]:
        """Time-integrated intensity: the interference pattern.

        Returns
        -------
        ndarray, shape (N, N)
            Sum of all recorded frames.  Returns an all-zero array if no
            frames have been recorded yet.
        """
        if not self._frames:
            return np.zeros((self._N, self._N), dtype=np.float32)
        return np.sum(self._frames, axis=0).astype(np.float32)

    @property
    def snapshots(self) -> NDArray[np.float32]:
        """All recorded frames, shape (K, N, N).

        Returns
        -------
        ndarray, shape (K, N, N)
            Each frame is one recorded slice.
        """
        if not self._frames:
            return np.zeros((0, self._N, self._N), dtype=np.float32)
        return np.stack(self._frames, axis=0)

    def click_pattern(self, n_particles: int = 1000, seed: int | None = None) -> NDArray[np.int32]:
        """Simulate discrete *particle clicks* from the intensity pattern.

        Samples ``n_particles`` detection events from the probability
        distribution defined by :attr:`pattern`, mimicking a single-
        photon or single-electron experiment run repeatedly.

        Parameters
        ----------
        n_particles : int
            Number of simulated particle detections.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        ndarray, shape (N, N)
            2-D histogram of click positions.
        """
        pat = self.pattern
        total = pat.sum()
        if total == 0:
            return np.zeros(pat.shape, dtype=np.int32)

        prob = pat.ravel() / total
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(prob), size=n_particles, p=prob)
        hist, _ = np.histogram(indices, bins=np.arange(len(prob) + 1))
        return hist.reshape(pat.shape).astype(np.int32)

    def mean_pattern(self) -> NDArray[np.float32]:
        """Time-averaged intensity (normalised by number of frames).

        Returns
        -------
        ndarray, shape (N, N)
        """
        if not self._frames:
            return np.zeros((self._N, self._N), dtype=np.float32)
        return (np.sum(self._frames, axis=0) / len(self._frames)).astype(np.float32)

    def line_profile(self, perp_axis: int = 0) -> NDArray[np.float32]:
        """1-D projection of the interference pattern.

        Sums the integrated pattern along the axis perpendicular to the
        main fringes (typically along x), yielding the classic
        multi-peak fringe profile along y.

        Parameters
        ----------
        perp_axis : int
            The axis along which to sum (collapses that dimension).

        Returns
        -------
        ndarray, shape (N,)
        """
        return self.pattern.sum(axis=perp_axis).astype(np.float32)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def position(self) -> int:
        """Detector-screen position along the propagation axis."""
        return self._pos

    @property
    def axis(self) -> int:
        """Axis perpendicular to this detector plane."""
        return self._axis

    @property
    def n_frames(self) -> int:
        """Number of frames recorded so far."""
        return len(self._frames)

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_field(self) -> NDArray[np.float32]:
        name = self._field_name
        if name == "energy_density":
            return self._sim.energy_density
        if name == "psi_real":
            return self._sim.psi_real
        if name == "psi_imag":
            f = self._sim.psi_imag
            return f if f is not None else np.zeros((self._N, self._N, self._N), dtype=np.float32)
        if name == "chi":
            return self._sim.chi
        raise ValueError(
            f"Unknown field '{name}'. Choose from 'energy_density', 'psi_real', 'psi_imag', 'chi'."
        )

    def __repr__(self) -> str:
        return (
            f"DetectorScreen(axis={self._axis}, position={self._pos}, "
            f"field='{self._field_name}', n_frames={self.n_frames})"
        )
