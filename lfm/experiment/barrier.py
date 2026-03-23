"""
Potential-barrier component for LFM double-slit experiments.
=============================================================

A :class:`Barrier` sets χ >> χ₀ in a planar region of the grid,
carving configurable slit openings.  Each slit can optionally host a
*which-path detector* that partially or fully damps the wave field as it
passes, reproducing the quantum-mechanical decoherence effect.

Physical mechanism
------------------
In GOV-01, the effective mass term is −χ²Ψ.  When χ >> χ₀ the wave
equation becomes very stiff — the wave oscillates at frequency χ rather
than propagating freely — so a wave packet with kinetic energy ≪ χ² is
effectively reflected.

Setting χ = χ₀ + Δ with Δ large (default Δ = 50) creates an almost
impenetrable barrier for any wave packet with |v| ≪ 1.

Slit openings restore χ = χ₀ so the wave can pass freely.  The
:meth:`~Barrier.step_callback` must be registered with
:meth:`~lfm.Simulation.run_with_snapshots` (or called manually after
each step) to re-enforce the desired χ values, since GOV-02 would
otherwise gradually diffuse the barrier away.

Which-path detection
--------------------
With ``Slit(detector=True, detector_strength=α)``, every time the
callback fires the wave amplitude at that slit is multiplied by
``(1 − α)``.  At α = 1 the slit is a perfect absorber (one-path
experiment).  At α = 0 the field is untouched (passive observation).
Intermediate values reproduce partial measurement / decoherence.

Example
-------
>>> barrier = Barrier(
...     sim, axis=2,
...     slits=[
...         Slit(center=24, width=4),                        # open slit
...         Slit(center=40, width=4, detector=True, detector_strength=0.95),  # detector
...     ],
... )
>>> snaps = sim.run_with_snapshots(
...     steps=3000, snapshot_every=50,
...     fields=["energy_density"],
...     callback=barrier.step_callback,
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from lfm.constants import CHI0

if TYPE_CHECKING:
    from lfm.simulation import Simulation

__all__ = ["Slit", "Barrier"]

# Default barrier χ-height: high enough to reflect wave packets with |v| < 0.9 c
_DEFAULT_BARRIER_HEIGHT: float = CHI0 + 50.0


@dataclass
class Slit:
    """Specification for a single slit opening in a :class:`Barrier`.

    Parameters
    ----------
    center : int
        Cell index along the slit axis (the primary transverse axis).
    width : int
        Number of cells spanning the slit opening.
    detector : bool
        If True, a which-path detector is installed at this slit.  The
        detector damps the wave field on every callback call.
    detector_strength : float
        Damping intensity in [0, 1].  0 = passive (field unchanged),
        1 = fully absorbing slit.
    """

    center: int
    width: int = 4
    detector: bool = False
    detector_strength: float = 1.0


class Barrier:
    """A χ-potential barrier with configurable slit openings.

    Parameters
    ----------
    sim : Simulation
        The LFM simulation to attach this barrier to.
    axis : int
        Index of the *propagation* axis perpendicular to the barrier
        plane.  0 = x-barrier, 1 = y-barrier, 2 = z-barrier (default).
        If the particle travels in the +z direction, use ``axis=2``.
    position : int or None
        Lattice index along ``axis`` at the barrier's centre.
        Defaults to the grid midpoint.
    height : float or None
        χ value inside the solid barrier regions.
        Defaults to ``CHI0 + 50`` (≈ 69).
    thickness : int
        Barrier depth in cells (default 2).
    slits : list of :class:`Slit` or None
        Slit specifications.  If *None*, two equal slits are placed
        symmetrically about the grid centre separated by ⌈N/8⌉ cells.
    slit_axis : int or None
        Which *transverse* axis positions the slits (i.e. the axis along
        which the slit *centres* vary).  Defaults to the first axis that
        is not the propagation axis, i.e. ``(axis + 1) % 3``.
    absorb : bool
        If True, Ψ is also zeroed inside the solid barrier cells on each
        :meth:`step_callback` call, making the barrier a perfect
        absorber as well as a reflector.  Default True.

    Attributes
    ----------
    mask : ndarray of bool, shape (N, N, N)
        Solid barrier cells (True) after slit carving.
    slit_masks : list of ndarray of bool
        Per-slit Boolean masks (same shape as *mask*).
    chi_barrier : ndarray
        χ array after the initial barrier placement.

    Notes
    -----
    The barrier must be *re-enforced* after every Leapfrog step, because
    GOV-02 gradually diffuses the high-χ region away.  Register
    :meth:`step_callback` with :meth:`~lfm.Simulation.run` or
    :meth:`~lfm.Simulation.run_with_snapshots`.
    """

    def __init__(
        self,
        sim: "Simulation",
        axis: int = 2,
        position: int | None = None,
        height: float | None = None,
        thickness: int = 2,
        slits: list[Slit] | None = None,
        slit_axis: int | None = None,
        absorb: bool = True,
    ) -> None:
        N = sim.config.grid_size
        self._sim = sim
        self._N = N
        self._axis = axis
        self._height = height if height is not None else _DEFAULT_BARRIER_HEIGHT
        self._pos = position if position is not None else N // 2
        self._thick = thickness
        self._absorb = absorb

        # Default two-slit geometry symmetric about the grid centre
        if slits is None:
            half_sep = max(4, N // 8)
            slit_w = max(2, N // 32)
            slits = [
                Slit(center=N // 2 - half_sep, width=slit_w),
                Slit(center=N // 2 + half_sep, width=slit_w),
            ]
        self._slits: list[Slit] = list(slits)

        # Slit axis: which transverse axis the slit centres vary along
        self._slit_axis = slit_axis if slit_axis is not None else (axis + 1) % 3

        # Pre-compute boolean masks
        self._barrier_mask, self._slit_masks = self._build_masks()
        self._full_slit_mask: NDArray[np.bool_] = np.zeros(
            (N, N, N), dtype=bool
        )
        for sm in self._slit_masks:
            self._full_slit_mask |= sm

        # Apply the initial barrier
        self.apply()

    # ── Mask construction ──────────────────────────────────────────────

    def _build_masks(
        self,
    ) -> tuple[NDArray[np.bool_], list[NDArray[np.bool_]]]:
        """Build the solid barrier mask and per-slit masks."""
        N, axis = self._N, self._axis
        start = max(0, self._pos - self._thick // 2)
        end = min(N, start + self._thick)

        # Full barrier band (all cells in the plane)
        barrier_mask = np.zeros((N, N, N), dtype=bool)
        sl = [slice(None), slice(None), slice(None)]
        sl[axis] = slice(start, end)
        barrier_mask[tuple(sl)] = True

        slit_masks: list[NDArray[np.bool_]] = []
        for slit in self._slits:
            s0 = max(0, slit.center - slit.width // 2)
            s1 = min(N, s0 + slit.width)
            sm = np.zeros((N, N, N), dtype=bool)
            ssl = [slice(None), slice(None), slice(None)]
            ssl[axis] = slice(start, end)
            ssl[self._slit_axis] = slice(s0, s1)
            sm[tuple(ssl)] = True
            slit_masks.append(sm)
            # Carve slit out of the solid barrier
            barrier_mask[tuple(ssl)] = False

        return barrier_mask, slit_masks

    # ── Core enforcement ───────────────────────────────────────────────

    def apply(self) -> None:
        """Re-impose barrier χ values and (optionally) zero Ψ inside.

        Call this every step (or every few steps) via
        :meth:`step_callback` to keep the barrier stable against GOV-02
        diffusion.
        """
        chi = self._sim.chi.copy()
        chi[self._barrier_mask] = self._height       # solid wall
        chi[self._full_slit_mask] = CHI0             # slit openings
        self._sim.set_chi(chi)

        if self._absorb:
            # Zero Ψ inside solid barrier cells (perfect absorber)
            pr = self._sim.psi_real.copy()
            pr[self._barrier_mask] = 0.0
            self._sim.set_psi_real(pr)

            if self._sim.psi_imag is not None:
                pi = self._sim.psi_imag.copy()
                pi[self._barrier_mask] = 0.0
                self._sim.set_psi_imag(pi)

    def attenuate_slits(self) -> None:
        """Apply which-path detector damping at all detector slits.

        For each slit with ``detector=True``, the wave amplitude inside
        the slit opening is multiplied by ``(1 − detector_strength)``.
        A strength of 1.0 completely absorbs the wave (full measurement);
        a strength of 0.0 leaves the field untouched.
        """
        for slit, sm in zip(self._slits, self._slit_masks):
            if not slit.detector or slit.detector_strength <= 0.0:
                continue
            factor = float(1.0 - min(1.0, slit.detector_strength))
            pr = self._sim.psi_real.copy()
            pr[sm] *= factor
            self._sim.set_psi_real(pr)

            if self._sim.psi_imag is not None:
                pi = self._sim.psi_imag.copy()
                pi[sm] *= factor
                self._sim.set_psi_imag(pi)

    def measure_slits(self) -> dict[str, float]:
        """Return mean energy density at each slit opening.

        Useful for passive bookkeeping (does not modify the field).

        Returns
        -------
        dict
            ``{"slit_0": float, "slit_1": float, ...}`` for each slit.
        """
        ed = self._sim.energy_density
        return {
            f"slit_{i}": float(ed[sm].mean())
            for i, sm in enumerate(self._slit_masks)
        }

    # ── Callback ───────────────────────────────────────────────────────

    def step_callback(self, sim: "Simulation", step: int) -> None:
        """Simulation step callback.

        Pass this to :meth:`~lfm.Simulation.run` or
        :meth:`~lfm.Simulation.run_with_snapshots` as the *callback*
        argument::

            sim.run_with_snapshots(steps=3000, callback=barrier.step_callback)

        On every call the barrier χ values are re-enforced and, if any
        slit has a detector installed, the which-path damping is applied.
        """
        self.apply()
        if any(s.detector for s in self._slits):
            self.attenuate_slits()

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Solid-barrier Boolean mask, shape (N, N, N)."""
        return self._barrier_mask

    @property
    def slit_masks(self) -> list[NDArray[np.bool_]]:
        """Per-slit Boolean masks (slit openings)."""
        return self._slit_masks

    @property
    def slits(self) -> list[Slit]:
        """Slit specifications as supplied at construction."""
        return self._slits

    @property
    def position(self) -> int:
        """Barrier centre position along the propagation axis."""
        return self._pos

    @property
    def axis(self) -> int:
        """Propagation axis index (0/1/2)."""
        return self._axis

    @property
    def height(self) -> float:
        """χ value inside the solid barrier cells."""
        return self._height

    def __repr__(self) -> str:
        return (
            f"Barrier(axis={self._axis}, position={self._pos}, "
            f"height={self._height:.1f}, thickness={self._thick}, "
            f"n_slits={len(self._slits)})"
        )
