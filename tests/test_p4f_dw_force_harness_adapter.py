from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from lfm.validation.unified_force import (
    BenchmarkStatus,
    ForceSector,
    UnifiedForceHarness,
)


REPO = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO / "paper_experiments" / "lfm_unified_force_harness_2026"
RUNNER = HARNESS_DIR / "run_p4f_dw_force_harness.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("p4f_dw_harness_adapter", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load P4F-DW harness adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_manifest_replaces_only_frame_specific_provenance() -> None:
    runner = _runner_module()
    assert "gravity_attraction_t2" in runner.ARTIFACTS
    _version, specs, payload = runner._candidate_specs()
    identifiers = {spec.benchmark_id for spec in specs}
    assert "GR-FRAME-PROVENANCE" not in identifiers
    assert "GR-CARRIER-PROVENANCE" in identifiers
    assert payload["candidate_changes"] == {
        "GR-FRAME-PROVENANCE": "GR-CARRIER-PROVENANCE"
    }
    strong_live = next(
        spec for spec in specs if spec.benchmark_id == "STRONG-CONFINEMENT-LIVE"
    )
    assert strong_live.required_for_sector
    assert "STRONG-ACTION-CLOSURE" in strong_live.dependencies


def test_candidate_ledger_cannot_promote_scoped_four_channel_pass() -> None:
    runner = _runner_module()
    version, specs, _payload = runner._candidate_specs()
    report = UnifiedForceHarness(specs, version=version).evaluate(
        runner._candidate_results(20260725)
    )
    assert report.unified_status is not BenchmarkStatus.PASS
    assert report.sector_status[ForceSector.STRONG] is BenchmarkStatus.BLOCKED
    passed_live_identities = {
        result.identity.fingerprint
        for result in report.results
        if result.status is BenchmarkStatus.PASS and result.identity is not None
    }
    assert len(passed_live_identities) <= 1


def test_unresolved_parameter_provenance_blocks_action_promotion() -> None:
    runner = _runner_module()
    results = {
        result.benchmark_id: result for result in runner._candidate_results(20260725)
    }
    assert results["WEAK-ACTION-CLOSURE"].status is BenchmarkStatus.BLOCKED
    assert results["WEAK-CHIRAL-LIVE-INTERACTION"].status is BenchmarkStatus.BLOCKED
    assert results["STRONG-ACTION-CLOSURE"].status is BenchmarkStatus.BLOCKED
