"""Strict evidence policy for unified-force LFM validation.

This module does not define force dynamics. It prevents structural identities,
reduced models, external comparators, or mixed actions from being promoted as
one live four-force result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class BenchmarkStatus(str, Enum):
    """Outcome of one frozen benchmark."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class EvidenceClass(str, Enum):
    """Scientific role of a benchmark result."""

    LIVE = "LIVE"
    STRUCTURAL = "STRUCTURAL"
    DIAGNOSTIC = "DIAGNOSTIC"
    COMPARATOR = "COMPARATOR"


class ForceSector(str, Enum):
    """Force or cross-cutting sector."""

    CORE = "CORE"
    GRAVITY = "GRAVITY"
    EM = "EM"
    WEAK = "WEAK"
    STRONG = "STRONG"
    UNIFIED = "UNIFIED"


class GateTier(str, Enum):
    """Evidence maturity tier."""

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"


class ReadoutFrame(str, Enum):
    """Reference system used by a benchmark measurement."""

    STRUCTURAL = "STRUCTURAL"
    INTERNAL_OPERATIONAL = "INTERNAL_OPERATIONAL"
    EXTERNAL_GRID = "EXTERNAL_GRID"
    CONTINUUM_COMPARATOR = "CONTINUUM_COMPARATOR"


GLOBAL_FORBIDDEN_MECHANISMS = frozenset(
    {
        "external_force",
        "prescribed_trajectory",
        "prescribed_potential",
        "frozen_source",
        "negative_energy_mode",
        "per_sector_tuning",
        "target_law_in_evolution",
        "fundamental_metric_register",
        "fundamental_affine_frame",
        "fundamental_u1_links",
        "target_theory_field",
        "external_grid_only_readout",
    }
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def stable_fingerprint(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-like data."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExecutionIdentity:
    """Identity of the live equations and numerics used by a result."""

    action_id: str
    register_id: str
    parameter_fingerprint: str
    gov01_stencil: str
    gov02_stencil: str
    boundary_policy: str
    implementation_fingerprint: str

    @classmethod
    def build(
        cls,
        *,
        action_id: str,
        register_id: str,
        parameters: Mapping[str, Any],
        gov01_stencil: str,
        gov02_stencil: str,
        boundary_policy: str,
        implementation: Mapping[str, Any],
    ) -> ExecutionIdentity:
        """Construct an identity from explicit action and implementation data."""
        return cls(
            action_id=action_id,
            register_id=register_id,
            parameter_fingerprint=stable_fingerprint(parameters),
            gov01_stencil=gov01_stencil,
            gov02_stencil=gov02_stencil,
            boundary_policy=boundary_policy,
            implementation_fingerprint=stable_fingerprint(implementation),
        )

    @property
    def fingerprint(self) -> str:
        """Return the full execution fingerprint."""
        return stable_fingerprint(asdict(self))


@dataclass(frozen=True)
class BenchmarkSpec:
    """Frozen contract for one benchmark."""

    benchmark_id: str
    sector: ForceSector
    tier: GateTier
    description: str
    required_for_sector: bool
    promotion_eligible: bool
    accepted_evidence: frozenset[EvidenceClass]
    dependencies: tuple[str, ...] = ()
    forbidden_mechanisms: frozenset[str] = frozenset()
    requires_shared_identity: bool = False
    readout_frame: ReadoutFrame = ReadoutFrame.STRUCTURAL
    substrate_evolution: str = ""
    internal_observable: str = ""
    continuum_interpretation: str = ""
    operational_readout_required: bool = False

    def __post_init__(self) -> None:
        if not self.benchmark_id or any(char.isspace() for char in self.benchmark_id):
            raise ValueError("benchmark_id must be non-empty and contain no whitespace")
        if self.required_for_sector and not self.promotion_eligible:
            raise ValueError("a required sector gate must be promotion eligible")
        if not self.accepted_evidence:
            raise ValueError("accepted_evidence cannot be empty")
        if self.operational_readout_required:
            if self.readout_frame is not ReadoutFrame.INTERNAL_OPERATIONAL:
                raise ValueError(
                    "an operational gate must use an INTERNAL_OPERATIONAL readout"
                )
            missing = [
                name
                for name, value in (
                    ("substrate_evolution", self.substrate_evolution),
                    ("internal_observable", self.internal_observable),
                    ("continuum_interpretation", self.continuum_interpretation),
                )
                if not value.strip()
            ]
            if missing:
                raise ValueError(
                    "REPRESENTATION AUDIT INCOMPLETE: missing " + ", ".join(missing)
                )
            _validate_internal_observable_text(self.internal_observable)


def _validate_internal_observable_text(text: str) -> None:
    """Require operational gates to name an actual internal wave readout.

    The four-force harness is allowed to retain external lattice coordinates as
    diagnostics, but a promoting operational gate must be phrased as something
    an observer built from the evolved wave fields could measure: clocks,
    rulers, phases, currents, correlations, fluxes, probes, spectra, and
    similar E/chi-derived observables. This guard prevents the common false
    pass where a raw simulator coordinate is relabeled as an observation.
    """
    lowered = " ".join(text.lower().replace("_", " ").split())
    forbidden_fragments = (
        "external coordinate",
        "external grid",
        "external-grid",
        "fixed grid",
        "god eye",
        "god's eye",
        "grid coordinate",
        "lattice coordinate",
        "raw coordinate",
        "raw simulator",
        "simulator time",
    )
    if any(fragment in lowered for fragment in forbidden_fragments):
        raise ValueError(
            "REPRESENTATION AUDIT INCOMPLETE: operational readout uses a raw "
            "external coordinate diagnostic"
        )
    internal_tokens = (
        "e-wave",
        "wave",
        "clock",
        "ruler",
        "phase",
        "current",
        "correlation",
        "probe",
        "charge",
        "coherence",
        "curvature",
        "flux",
        "helicity",
        "chirality",
        "energy",
        "spectral",
        "scattering",
        "singlet",
        "nonsinglet",
        "internal",
    )
    if not any(token in lowered for token in internal_tokens):
        raise ValueError(
            "REPRESENTATION AUDIT INCOMPLETE: operational readout must name "
            "an internal E/chi wave observable"
        )


@dataclass(frozen=True)
class BenchmarkResult:
    """Evidence returned by one benchmark."""

    benchmark_id: str
    status: BenchmarkStatus
    evidence: EvidenceClass
    reason: str
    metrics: Mapping[str, JsonScalar] = field(default_factory=dict)
    mechanisms_used: frozenset[str] = frozenset()
    identity: ExecutionIdentity | None = None
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessReport:
    """Adjudicated immutable view of a harness run."""

    harness_version: str
    manifest_fingerprint: str
    results: tuple[BenchmarkResult, ...]
    sector_status: Mapping[ForceSector, BenchmarkStatus]
    unified_status: BenchmarkStatus
    policy_findings: tuple[str, ...]
    shared_identity_fingerprint: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable evidence ledger."""
        result_rows: list[JsonValue] = []
        for result in self.results:
            identity = asdict(result.identity) if result.identity is not None else None
            result_rows.append(
                {
                    "benchmark_id": result.benchmark_id,
                    "status": result.status.value,
                    "evidence": result.evidence.value,
                    "reason": result.reason,
                    "metrics": dict(result.metrics),
                    "mechanisms_used": sorted(result.mechanisms_used),
                    "identity": identity,
                    "identity_fingerprint": (
                        result.identity.fingerprint
                        if result.identity is not None
                        else None
                    ),
                    "artifacts": list(result.artifacts),
                }
            )
        return {
            "harness_version": self.harness_version,
            "manifest_fingerprint": self.manifest_fingerprint,
            "unified_status": self.unified_status.value,
            "sector_status": {
                sector.value: status.value
                for sector, status in self.sector_status.items()
            },
            "shared_identity_fingerprint": self.shared_identity_fingerprint,
            "policy_findings": list(self.policy_findings),
            "results": result_rows,
        }


class UnifiedForceHarness:
    """Adjudicate benchmark evidence under strict unified-force rules."""

    def __init__(
        self,
        specs: Iterable[BenchmarkSpec],
        *,
        version: str = "1.0",
    ) -> None:
        spec_list = tuple(specs)
        by_id = {spec.benchmark_id: spec for spec in spec_list}
        if len(by_id) != len(spec_list):
            raise ValueError("benchmark IDs must be unique")
        for spec in spec_list:
            missing = set(spec.dependencies) - set(by_id)
            if missing:
                raise ValueError(
                    f"{spec.benchmark_id} has unknown dependencies: {sorted(missing)}"
                )
        self.specs = spec_list
        self.by_id = by_id
        self.version = version
        self.manifest_fingerprint = stable_fingerprint(
            [
                {
                    **asdict(spec),
                    "sector": spec.sector.value,
                    "tier": spec.tier.value,
                    "readout_frame": spec.readout_frame.value,
                    "accepted_evidence": sorted(
                        evidence.value for evidence in spec.accepted_evidence
                    ),
                    "forbidden_mechanisms": sorted(spec.forbidden_mechanisms),
                }
                for spec in spec_list
            ]
        )

    def _result_map(
        self,
        results: Iterable[BenchmarkResult],
    ) -> dict[str, BenchmarkResult]:
        rows = tuple(results)
        result_map = {result.benchmark_id: result for result in rows}
        if len(result_map) != len(rows):
            raise ValueError("result benchmark IDs must be unique")
        unknown = set(result_map) - set(self.by_id)
        if unknown:
            raise ValueError(f"results contain unknown benchmark IDs: {sorted(unknown)}")
        return result_map

    @staticmethod
    def _blocked_missing(spec: BenchmarkSpec) -> BenchmarkResult:
        status = (
            BenchmarkStatus.BLOCKED
            if spec.required_for_sector
            else BenchmarkStatus.NOT_RUN
        )
        return BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            status=status,
            evidence=next(iter(spec.accepted_evidence)),
            reason="No result was supplied for this frozen benchmark.",
        )

    def evaluate(self, results: Iterable[BenchmarkResult]) -> HarnessReport:
        """Validate results and compute strict sector and unified statuses."""
        supplied = self._result_map(results)
        adjudicated: dict[str, BenchmarkResult] = {}
        findings: list[str] = []

        for spec in self.specs:
            result = supplied.get(spec.benchmark_id, self._blocked_missing(spec))
            forbidden = (
                GLOBAL_FORBIDDEN_MECHANISMS | spec.forbidden_mechanisms
            ) & result.mechanisms_used
            if forbidden:
                message = (
                    f"{spec.benchmark_id}: forbidden mechanisms used: "
                    f"{sorted(forbidden)}"
                )
                findings.append(message)
                result = replace(
                    result,
                    status=BenchmarkStatus.FAIL,
                    reason=message,
                )
            elif (
                result.status is BenchmarkStatus.PASS
                and result.evidence not in spec.accepted_evidence
            ):
                message = (
                    f"{spec.benchmark_id}: {result.evidence.value} evidence cannot "
                    "satisfy this gate"
                )
                findings.append(message)
                result = replace(
                    result,
                    status=BenchmarkStatus.FAIL,
                    reason=message,
                )
            elif (
                result.status is BenchmarkStatus.PASS
                and spec.requires_shared_identity
                and result.identity is None
            ):
                message = f"{spec.benchmark_id}: live pass lacks an execution identity"
                findings.append(message)
                result = replace(
                    result,
                    status=BenchmarkStatus.FAIL,
                    reason=message,
                )
            elif (
                result.status is BenchmarkStatus.PASS
                and spec.operational_readout_required
                and spec.readout_frame is not ReadoutFrame.INTERNAL_OPERATIONAL
            ):
                message = (
                    f"{spec.benchmark_id}: external-grid or comparator readout "
                    "cannot satisfy an internal operational gate"
                )
                findings.append(message)
                result = replace(
                    result,
                    status=BenchmarkStatus.FAIL,
                    reason=message,
                )

            dependency_statuses = {
                dependency: adjudicated[dependency].status
                for dependency in spec.dependencies
            }
            unsatisfied = {
                dependency: status
                for dependency, status in dependency_statuses.items()
                if status is not BenchmarkStatus.PASS
            }
            if result.status is BenchmarkStatus.PASS and unsatisfied:
                message = (
                    f"{spec.benchmark_id}: claimed PASS with unsatisfied dependencies "
                    f"{ {key: value.value for key, value in unsatisfied.items()} }"
                )
                findings.append(message)
                result = replace(
                    result,
                    status=BenchmarkStatus.FAIL,
                    reason=message,
                )
            adjudicated[spec.benchmark_id] = result

        required_identities = [
            adjudicated[spec.benchmark_id].identity
            for spec in self.specs
            if spec.required_for_sector
            and spec.requires_shared_identity
            and adjudicated[spec.benchmark_id].status is BenchmarkStatus.PASS
        ]
        identity_fingerprints = {
            identity.fingerprint
            for identity in required_identities
            if identity is not None
        }
        shared_identity = (
            next(iter(identity_fingerprints))
            if len(identity_fingerprints) == 1
            else None
        )
        identity_mismatch = len(identity_fingerprints) > 1
        if identity_mismatch:
            findings.append(
                "Required live gates used more than one execution identity; "
                "same-action promotion is forbidden."
            )

        sector_status: dict[ForceSector, BenchmarkStatus] = {}
        for sector in ForceSector:
            required = [
                adjudicated[spec.benchmark_id].status
                for spec in self.specs
                if spec.sector is sector and spec.required_for_sector
            ]
            if not required:
                sector_status[sector] = BenchmarkStatus.BLOCKED
            elif any(status is BenchmarkStatus.FAIL for status in required):
                sector_status[sector] = BenchmarkStatus.FAIL
            elif all(status is BenchmarkStatus.PASS for status in required):
                sector_status[sector] = BenchmarkStatus.PASS
            else:
                sector_status[sector] = BenchmarkStatus.BLOCKED

        required_sectors = (
            ForceSector.CORE,
            ForceSector.GRAVITY,
            ForceSector.EM,
            ForceSector.WEAK,
            ForceSector.STRONG,
            ForceSector.UNIFIED,
        )
        if identity_mismatch or any(
            sector_status[sector] is BenchmarkStatus.FAIL
            for sector in required_sectors
        ):
            unified_status = BenchmarkStatus.FAIL
        elif all(
            sector_status[sector] is BenchmarkStatus.PASS
            for sector in required_sectors
        ):
            unified_status = BenchmarkStatus.PASS
        else:
            unified_status = BenchmarkStatus.BLOCKED

        return HarnessReport(
            harness_version=self.version,
            manifest_fingerprint=self.manifest_fingerprint,
            results=tuple(
                adjudicated[spec.benchmark_id] for spec in self.specs
            ),
            sector_status=sector_status,
            unified_status=unified_status,
            policy_findings=tuple(findings),
            shared_identity_fingerprint=shared_identity,
        )
