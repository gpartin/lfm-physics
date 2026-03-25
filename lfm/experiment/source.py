"""
Continuous Wave Source for LFM Experiments
==========================================

A :class:`ContinuousSource` injects a monochromatic plane-wave source at a
chosen grid plane on every leapfrog step, producing a steady-state standing
or travelling wave (Paper-055 technique).  This is the recommended way to
"move a particle" across the grid for interference experiments.

Unlike :meth:`Simulation.place_soliton` (one-shot Gaussian, disperses) or
:meth:`Simulation.place_plane_wave` (fills domain at *t*\\ =0, crosses
barrier once), a continuous source builds up indefinitely, giving bright,
high-contrast fringes on the detector.

Fractional positions
--------------------
All position and size parameters accept either **absolute cells** (int or
float ≥ 1) or **fractions of N** (float in [0, 1)).  This means the same
source specification produces identical physics at *any* grid size::

    # These are equivalent at N = 64:
    ContinuousSource(sim, position=16)       # absolute cell 16
    ContinuousSource(sim, position=0.25)     # 25% of N = cell 16

Example
-------
>>> source = ContinuousSource(sim, position=0.25, omega=2.0, amplitude=2.0)
>>> barrier = sim.place_barrier(position=0.5, ...)
>>> screen = sim.add_detector(position=0.75)
>>> sim.run_with_snapshots(
...     steps=source.recommended_steps(target_z=0.75),
...     step_callback=lambda s, t: (
...         source.step_callback(s, t),
...         barrier.step_callback(s, t),
...         screen.record(),
...     ),
... )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from lfm.experiment.dispersion import Dispersion, dispersion

if TYPE_CHECKING:
    from lfm.simulation import Simulation

__all__ = ["ContinuousSource"]


def _resolve(value: float | int, N: int) -> int:
    """Convert a fractional-or-absolute coordinate to an integer cell index.

    * ``value < 1.0`` (and float) → fraction of *N*, rounded to nearest int.
    * ``value >= 1``  (int **or** float) → used as-is (clamped to [0, N-1]).

    This is the universal scaling rule for all experiment geometry.
    """
    if isinstance(value, float) and value < 1.0:
        return max(0, min(N - 1, round(value * N)))
    return max(0, min(N - 1, int(value)))


def _resolve_sigma(value: float | int, N: int) -> float:
    """Convert a fractional-or-absolute width to a float cell count.

    * ``value < 1.0`` → fraction of *N*.
    * ``value >= 1.0`` → absolute cells.
    """
    if isinstance(value, float) and value < 1.0:
        return value * N
    return float(value)


class ContinuousSource:
    """Monochromatic CW plane-wave source (Paper-055 technique).

    Injects ``Δt² × amplitude × sin(ω t) × envelope`` into ``psi_real``
    (and ``psi_imag`` for complex fields with nonzero phase) at a source
    plane every leapfrog step.

    Parameters
    ----------
    sim : Simulation
        The simulation this source is attached to.
    axis : int
        Propagation axis (0=x, 1=y, 2=z).  Default 2.
    position : float
        Source plane coordinate.  Fractional (< 1.0) → fraction of N;
        integer or float ≥ 1 → absolute cell.
    omega : float
        Drive angular frequency (rad / time-unit).  Must be > chi0 for
        the wave to propagate.
    amplitude : float
        Peak injection amplitude.
    envelope_sigma : float
        Transverse Gaussian 1/e half-width.  Fractional (< 1.0) →
        fraction of N; ≥ 1 → absolute cells.  Default 0.25.
    phase : float
        Complex phase of the source (0 = real/electron, π = positron).
    boost : float
        Extra multiplicative factor applied each step (default 10.0 for
        fast buildup on small grids).  Reduce for sensitive measurements.
    """

    def __init__(
        self,
        sim: "Simulation",
        *,
        axis: int = 2,
        position: float = 0.25,
        omega: float = 2.0,
        amplitude: float = 2.0,
        envelope_sigma: float = 0.25,
        phase: float = 0.0,
        boost: float = 10.0,
    ) -> None:
        N = sim.config.grid_size
        self._axis = axis
        self._position_z = _resolve(position, N)
        self._omega = omega
        self._amplitude = amplitude
        self._phase = phase
        self._boost = boost
        self._dt = sim.config.dt
        self._N = N

        # Compute dispersion
        self._disp = dispersion(omega=omega, chi0=sim.config.chi0, dt=sim.config.dt)

        # Build transverse Gaussian envelope (N × N)
        sigma_cells = _resolve_sigma(envelope_sigma, N)
        centre = N / 2.0
        trans_axes = sorted(i for i in range(3) if i != axis)
        coords = [np.arange(N, dtype=np.float32) - centre for _ in range(2)]

        # 2D Gaussian on the transverse plane
        c0 = coords[0].reshape(-1, 1)  # first transverse axis
        c1 = coords[1].reshape(1, -1)  # second transverse axis
        self._envelope = np.exp(-(c0**2 + c1**2) / (2.0 * sigma_cells**2)).astype(np.float32)

        # Cache envelope on the backend device (avoids per-step CPU→GPU copy)
        self._envelope_device = sim._to_device(self._envelope)
        N = self._N
        # Reshape to 2D on-device view for slice injection
        self._envelope_device = self._envelope_device.reshape(N, N)

        self._trans_axes = trans_axes

    # ── Public interface ────────────────────────────────────────────────

    @property
    def dispersion(self) -> Dispersion:
        """Dispersion relation for this source's frequency."""
        return self._disp

    @property
    def position(self) -> int:
        """Source plane cell index."""
        return self._position_z

    def transit_steps(self, distance_cells: float | int) -> int:
        """Estimated steps for wavefront to travel *distance_cells*.

        Uses the exact discrete group velocity.
        """
        if isinstance(distance_cells, float) and distance_cells < 1.0:
            distance_cells = distance_cells * self._N
        vg = self._disp.v_group
        if vg <= 0:
            return 999_999
        return int(math.ceil(float(distance_cells) / vg))

    def recommended_steps(
        self, target_z: float | int | None = None, buildup_factor: float = 2.0
    ) -> int:
        """Steps for transit + steady-state buildup.

        Parameters
        ----------
        target_z : float or int or None
            Detector position (fractional or absolute).  Default: 75% of N.
        buildup_factor : float
            Multiply transit time by this to allow pattern accumulation.
        """
        N = self._N
        if target_z is None:
            target_z_cell = _resolve(0.75, N)
        else:
            target_z_cell = _resolve(target_z, N)
        distance = abs(target_z_cell - self._position_z)
        transit = self.transit_steps(distance)
        return int(transit * (1.0 + buildup_factor))

    def step_callback(self, sim: "Simulation", step: int) -> None:
        """Inject CW source — register with ``run_with_snapshots``.

        Uses zero-copy buffer access to avoid GPU↔CPU roundtrips.
        """
        t = step * self._dt
        injection = self._dt**2 * self._amplitude * math.sin(self._omega * t) * self._boost
        if abs(injection) < 1e-30:
            return

        z = self._position_z
        env = self._envelope_device

        # Modify native buffer in-place (cupy view on GPU, numpy on CPU)
        psi = sim._native_psi_real()
        if self._axis == 0:
            psi[z, :, :] += injection * env
        elif self._axis == 1:
            psi[:, z, :] += injection * env
        else:
            psi[:, :, z] += injection * env

        # Complex field: inject imaginary part with phase offset
        if abs(self._phase) > 1e-12:
            pi = sim._native_psi_imag()
            if pi is not None:
                phase_factor = injection * math.sin(self._phase)
                if self._axis == 0:
                    pi[z, :, :] += phase_factor * env
                elif self._axis == 1:
                    pi[:, z, :] += phase_factor * env
                else:
                    pi[:, :, z] += phase_factor * env
