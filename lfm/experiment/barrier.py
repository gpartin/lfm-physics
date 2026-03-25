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

from dataclasses import dataclass
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
        self._full_slit_mask: NDArray[np.bool_] = np.zeros((N, N, N), dtype=bool)
        for sm in self._slit_masks:
            self._full_slit_mask |= sm

        # Cache float32 masks on the compute device
        # (avoids per-step CPU→GPU copy in apply()).
        # Values are 1.0 (inside mask) / 0.0 (outside) for arithmetic masking.
        _td = sim._to_device
        self._barrier_mask_dev = _td(self._barrier_mask.astype(np.float32)).reshape(N, N, N)
        self._full_slit_mask_dev = _td(self._full_slit_mask.astype(np.float32)).reshape(N, N, N)
        self._slit_masks_dev = [
            _td(sm.astype(np.float32)).reshape(N, N, N) for sm in self._slit_masks
        ]

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

        Which-path detectors are implemented via χ modification: a
        detector at strength α sets χ at that slit to
        ``χ₀ + (height − χ₀) × α``.  At α=1 the slit is opaque
        (χ = height), at α=0 it is fully transparent (χ = χ₀).
        The wave is attenuated by GOV-01 evanescence — no Ψ modification
        is needed, avoiding global-velocity artefacts.
        """
        # All 4 chi buffers must be written so the barrier persists
        # across leapfrog double-buffer swaps.
        #
        # IMPORTANT: we replicate the active buffer to all 4 after
        # modifying barrier/slit cells.  This zeros the chi velocity
        # (chi_current − chi_prev) everywhere, which prevents the
        # barrier's high-χ value from radiating outward as a chi-wave
        # through the GOV-02 wave equation.  Without this, the barrier
        # χ floods the entire grid within a few hundred steps.
        active = self._sim._native_chi()  # 3D view of active buf
        bm = self._barrier_mask_dev  # float32 1/0
        sm_all = self._full_slit_mask_dev  # float32 1/0
        h = float(self._height)
        chi0 = float(self._sim.config.chi0)

        # Set barrier/slit/detector values in the active buffer
        active[:] = active * (1.0 - bm) + h * bm
        active[:] = active * (1.0 - sm_all) + chi0 * sm_all

        for slit, sm_dev in zip(self._slits, self._slit_masks_dev):
            if slit.detector and slit.detector_strength > 0.0:
                alpha = float(min(1.0, slit.detector_strength))
                det_chi = chi0 + (h - chi0) * alpha
                active[:] = active * (1.0 - sm_dev) + det_chi * sm_dev

        # Replicate active to all 4 chi buffers (zero velocity).
        # One copy is self→self (harmless); avoids complex pointer comparison.
        chi_bufs = self._sim._native_chi_pair()  # (A, B, prevA, prevB)
        for buf in chi_bufs:
            buf[:] = active

        if self._absorb:
            # Zero Ψ inside solid barrier cells (perfect absorber).
            pr = self._sim._native_psi_real()
            pr[:] = pr * (1.0 - bm)

            pi = self._sim._native_psi_imag()
            if pi is not None:
                pi[:] = pi * (1.0 - bm)

    def attenuate_slits(self) -> None:
        """No-op: which-path detection is now handled via χ in :meth:`apply`.

        The χ-based mechanism (raising χ at detector slits proportional to
        ``detector_strength``) naturally attenuates the wave through GOV-01
        evanescence without requiring any Ψ modification, which avoids the
        global-velocity artefact that psi-based attenuation causes.
        """

    def measure_slits(self) -> dict[str, float]:
        """Return mean energy density at each slit opening.

        Useful for passive bookkeeping (does not modify the field).

        Returns
        -------
        dict
            ``{"slit_0": float, "slit_1": float, ...}`` for each slit.
        """
        ed = self._sim.energy_density
        return {f"slit_{i}": float(ed[sm].mean()) for i, sm in enumerate(self._slit_masks)}

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
