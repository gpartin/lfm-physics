"""
LFM Simulation Job Schema
==========================

Stdlib-only dataclasses that define the request/response contract for the
``POST /v1/simulate_job`` WaveGuard endpoint.  These are used by
``remote_backend.py`` to build the request dict and parse the response.

No external dependencies required — plain Python dataclasses.
"""

from __future__ import annotations

import base64
import gzip
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class HookSpec:
    """One hook applied during a simulation phase (source, barrier, or detector)."""

    type: str  # "continuous_source" | "absorbing_barrier" | "detector_screen"
    axis: int = 2  # 0=x, 1=y, 2=z
    position: int = 0  # grid index along axis
    # continuous_source fields
    omega: float = 1.0
    amplitude: float = 1.0
    envelope_sigma: Optional[float] = None
    boost: float = 1.0
    # detector_screen fields
    tag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SnapshotSpec:
    every_n_steps: int = 0
    include_chi: bool = True
    include_psi: bool = True
    detector_z_slice: Optional[int] = (
        None  # legacy: server returns 2D slice; use downsample_stride instead
    )
    downsample_stride: Optional[int] = None  # 3D spatial stride: N=256+stride=4 → 64³ per snapshot

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        # always include the booleans even if False
        d.setdefault("include_chi", self.include_chi)
        d.setdefault("include_psi", self.include_psi)
        return d


@dataclass
class RunPlanStep:
    steps: int
    hooks: List[HookSpec] = field(default_factory=list)
    snapshots: SnapshotSpec = field(default_factory=SnapshotSpec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "hooks": [h.to_dict() for h in self.hooks],
            "snapshots": self.snapshots.to_dict(),
        }


@dataclass
class SimulationJob:
    """Full simulation job specification sent to POST /v1/simulate_job."""

    grid_size: int
    run_plan: List[RunPlanStep]
    chi0: float = 19.0
    kappa: float = 1.0 / 63.0
    dt: float = 0.02
    initial_psi: Optional[np.ndarray] = None  # shape (N,N,N) float32
    initial_chi: Optional[np.ndarray] = None  # shape (N,N,N) float32
    fusion_depth: Optional[int] = None
    pruner_enabled: bool = False
    freeze_chi: bool = False

    def to_request_dict(self) -> Dict[str, Any]:
        """Serialise to the JSON body expected by the WaveGuard API."""
        d: Dict[str, Any] = {
            "grid_size": self.grid_size,
            "chi0": self.chi0,
            "kappa": self.kappa,
            "dt": self.dt,
            "run_plan": [s.to_dict() for s in self.run_plan],
            "pruner_enabled": self.pruner_enabled,
            "freeze_chi": self.freeze_chi,
        }
        if self.fusion_depth is not None:
            d["fusion_depth"] = self.fusion_depth
        if self.initial_psi is not None:
            d["initial_psi"] = self.initial_psi.astype(np.float32).flatten().tolist()
        if self.initial_chi is not None:
            # gzip+base64: ~134 MB float list → ~200 KB for typical chi fields
            d["initial_chi_b64gz"] = base64.b64encode(
                gzip.compress(self.initial_chi.astype(np.float32).tobytes(), compresslevel=1)
            ).decode()
        return d


@dataclass
class Snapshot:
    """Decoded snapshot from a simulation response."""

    step: int
    psi: Optional[np.ndarray] = None  # shape (N,N,N) or (N,N) when 2D slice
    chi: Optional[np.ndarray] = None

    @classmethod
    def from_response_dict(cls, d: Dict[str, Any], N: int) -> "Snapshot":
        stride = int(d.get("psi_downsample_stride", 1))
        eff_N = N // stride  # effective grid size after spatial downsampling

        def _decode(
            b64: Optional[str], is_2d: bool = False, downsampled: bool = False
        ) -> Optional[np.ndarray]:
            if b64 is None:
                return None
            raw = base64.b64decode(b64)
            if raw[:2] == b"\x1f\x8b":  # gzip magic — server sent compressed snapshot
                raw = gzip.decompress(raw)
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            if is_2d:
                return arr.reshape(N, N)  # legacy 2D slice in original coords
            n = eff_N if downsampled else N
            return arr.reshape(n, n, n)

        is_2d = bool(d.get("psi_is_2d_slice", False))
        return cls(
            step=d["step"],
            psi=_decode(d.get("psi_b64"), is_2d=is_2d, downsampled=(stride > 1 and not is_2d)),
            chi=_decode(d.get("chi_b64"), downsampled=(stride > 1)),
        )


@dataclass
class JobResult:
    """Decoded response from POST /v1/simulate_job."""

    job_id: str
    steps_completed: int
    elapsed_ms: int
    fusion_depth: int
    active_fraction: float
    pruning_efficiency: str
    backend: str
    energy_initial: Optional[float]
    energy_final: Optional[float]
    energy_drift_pct: Optional[float]
    snapshots: List[Snapshot] = field(default_factory=list)
    detector_patterns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response_dict(cls, d: Dict[str, Any], N: int) -> "JobResult":
        ki = d.get("kernel_info", {})
        metrics = d.get("metrics", {})
        snapshots = [Snapshot.from_response_dict(s, N) for s in d.get("snapshots", [])]
        return cls(
            job_id=d["job_id"],
            steps_completed=d["steps_completed"],
            elapsed_ms=d["elapsed_ms"],
            fusion_depth=ki.get("fusion_depth", 1),
            active_fraction=ki.get("active_fraction", 1.0),
            pruning_efficiency=ki.get("pruning_efficiency", "0%"),
            backend=ki.get("backend", "unknown"),
            energy_initial=metrics.get("energy_initial"),
            energy_final=metrics.get("energy_final"),
            energy_drift_pct=metrics.get("energy_drift_pct"),
            snapshots=snapshots,
            detector_patterns=d.get("detector_patterns", {}),
        )

    @property
    def psi_final(self) -> Optional[np.ndarray]:
        """Return psi from the last snapshot, or None."""
        for s in reversed(self.snapshots):
            if s.psi is not None:
                return s.psi
        return None

    @property
    def chi_final(self) -> Optional[np.ndarray]:
        """Return chi from the last snapshot, or None."""
        for s in reversed(self.snapshots):
            if s.chi is not None:
                return s.chi
        return None
