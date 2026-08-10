"""Mutation tests for the unified-force evidence policy."""

from __future__ import annotations

import pytest

from lfm.validation.unified_force import (
    BenchmarkResult,
    BenchmarkSpec,
    BenchmarkStatus,
    EvidenceClass,
    ExecutionIdentity,
    ForceSector,
    GateTier,
    ReadoutFrame,
    UnifiedForceHarness,
)


def _identity(action: str = "action-a") -> ExecutionIdentity:
    return ExecutionIdentity.build(
        action_id=action,
        register_id="R2",
        parameters={"chi0": 19.0},
        gov01_stencil="19",
        gov02_stencil="19",
        boundary_policy="periodic",
        implementation={"module": "test"},
    )


def _spec(
    benchmark_id: str,
    sector: ForceSector,
    *,
    evidence: EvidenceClass = EvidenceClass.LIVE,
    dependencies: tuple[str, ...] = (),
) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id=benchmark_id,
        sector=sector,
        tier=GateTier.T2,
        description="test gate",
        required_for_sector=True,
        promotion_eligible=True,
        accepted_evidence=frozenset({evidence}),
        dependencies=dependencies,
        requires_shared_identity=evidence is EvidenceClass.LIVE,
    )


def _pass(
    benchmark_id: str,
    *,
    evidence: EvidenceClass = EvidenceClass.LIVE,
    identity: ExecutionIdentity | None = None,
    mechanisms: frozenset[str] = frozenset(),
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        status=BenchmarkStatus.PASS,
        evidence=evidence,
        reason="synthetic pass",
        mechanisms_used=mechanisms,
        identity=identity,
    )


def test_missing_required_gate_blocks() -> None:
    harness = UnifiedForceHarness([_spec("GR-LIVE", ForceSector.GRAVITY)])
    report = harness.evaluate([])
    assert report.sector_status[ForceSector.GRAVITY] is BenchmarkStatus.BLOCKED
    assert report.unified_status is BenchmarkStatus.BLOCKED


def test_diagnostic_cannot_substitute_for_live() -> None:
    harness = UnifiedForceHarness([_spec("EM-LIVE", ForceSector.EM)])
    report = harness.evaluate(
        [_pass("EM-LIVE", evidence=EvidenceClass.DIAGNOSTIC, identity=_identity())]
    )
    assert report.results[0].status is BenchmarkStatus.FAIL


def test_forbidden_mechanism_forces_failure() -> None:
    harness = UnifiedForceHarness([_spec("GR-LIVE", ForceSector.GRAVITY)])
    report = harness.evaluate(
        [
            _pass(
                "GR-LIVE",
                identity=_identity(),
                mechanisms=frozenset({"prescribed_potential"}),
            )
        ]
    )
    assert report.results[0].status is BenchmarkStatus.FAIL


def test_unsatisfied_dependency_rejects_claimed_pass() -> None:
    specs = [
        _spec("CORE", ForceSector.CORE, evidence=EvidenceClass.STRUCTURAL),
        _spec("EM-LIVE", ForceSector.EM, dependencies=("CORE",)),
    ]
    harness = UnifiedForceHarness(specs)
    report = harness.evaluate([_pass("EM-LIVE", identity=_identity())])
    assert report.results[1].status is BenchmarkStatus.FAIL


def test_mixed_actions_forbid_unified_promotion() -> None:
    sectors = (
        ForceSector.CORE,
        ForceSector.GRAVITY,
        ForceSector.EM,
        ForceSector.WEAK,
        ForceSector.STRONG,
        ForceSector.UNIFIED,
    )
    specs = [_spec(f"{sector.value}-LIVE", sector) for sector in sectors]
    results = [
        _pass(
            spec.benchmark_id,
            identity=_identity("action-b" if index == 4 else "action-a"),
        )
        for index, spec in enumerate(specs)
    ]
    report = UnifiedForceHarness(specs).evaluate(results)
    assert report.unified_status is BenchmarkStatus.FAIL
    assert report.shared_identity_fingerprint is None


def test_one_shared_action_can_pass_policy() -> None:
    sectors = (
        ForceSector.CORE,
        ForceSector.GRAVITY,
        ForceSector.EM,
        ForceSector.WEAK,
        ForceSector.STRONG,
        ForceSector.UNIFIED,
    )
    specs = [_spec(f"{sector.value}-LIVE", sector) for sector in sectors]
    identity = _identity()
    results = [_pass(spec.benchmark_id, identity=identity) for spec in specs]
    report = UnifiedForceHarness(specs).evaluate(results)
    assert report.unified_status is BenchmarkStatus.PASS
    assert report.shared_identity_fingerprint == identity.fingerprint


def test_external_grid_cannot_define_required_operational_gate() -> None:
    with pytest.raises(ValueError, match="INTERNAL_OPERATIONAL"):
        BenchmarkSpec(
            benchmark_id="GR-EXTERNAL",
            sector=ForceSector.GRAVITY,
            tier=GateTier.T2,
            description="invalid external readout",
            required_for_sector=True,
            promotion_eligible=True,
            accepted_evidence=frozenset({EvidenceClass.LIVE}),
            readout_frame=ReadoutFrame.EXTERNAL_GRID,
            substrate_evolution="GOV-01/GOV-02",
            internal_observable="grid coordinate position",
            continuum_interpretation="gravity",
            operational_readout_required=True,
        )


def test_operational_gate_requires_three_layer_mapping() -> None:
    with pytest.raises(ValueError, match="REPRESENTATION AUDIT INCOMPLETE"):
        BenchmarkSpec(
            benchmark_id="EM-INCOMPLETE",
            sector=ForceSector.EM,
            tier=GateTier.T2,
            description="missing internal measurement",
            required_for_sector=True,
            promotion_eligible=True,
            accepted_evidence=frozenset({EvidenceClass.LIVE}),
            readout_frame=ReadoutFrame.INTERNAL_OPERATIONAL,
            substrate_evolution="multicomponent GOV-01/GOV-02",
            internal_observable="",
            continuum_interpretation="electromagnetism",
            operational_readout_required=True,
        )


def test_internal_operational_gate_rejects_raw_coordinate_readout() -> None:
    with pytest.raises(ValueError, match="raw external coordinate"):
        BenchmarkSpec(
            benchmark_id="EM-RAW-READOUT",
            sector=ForceSector.EM,
            tier=GateTier.T3,
            description="invalid raw coordinate readout",
            required_for_sector=True,
            promotion_eligible=True,
            accepted_evidence=frozenset({EvidenceClass.LIVE}),
            readout_frame=ReadoutFrame.INTERNAL_OPERATIONAL,
            substrate_evolution="multicomponent GOV-01/GOV-02",
            internal_observable="raw simulator time and grid coordinate phase",
            continuum_interpretation="electromagnetism",
            operational_readout_required=True,
        )


def test_valid_internal_operational_gate_can_pass() -> None:
    spec = BenchmarkSpec(
        benchmark_id="GR-INTERNAL",
        sector=ForceSector.GRAVITY,
        tier=GateTier.T2,
        description="valid relational readout",
        required_for_sector=True,
        promotion_eligible=True,
        accepted_evidence=frozenset({EvidenceClass.LIVE}),
        requires_shared_identity=True,
        readout_frame=ReadoutFrame.INTERNAL_OPERATIONAL,
        substrate_evolution="local GOV-01/GOV-02",
        internal_observable="E-wave clock and ruler ratios",
        continuum_interpretation="effective free fall",
        operational_readout_required=True,
    )
    report = UnifiedForceHarness([spec]).evaluate(
        [_pass("GR-INTERNAL", identity=_identity())]
    )
    assert report.results[0].status is BenchmarkStatus.PASS
