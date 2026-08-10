"""Regression tests for active versus legacy four-force entry points."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from lfm.validation.unified_force import (
    EvidenceClass,
    ForceSector,
    ReadoutFrame,
)

REPO = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO / "paper_experiments" / "lfm_unified_force_harness_2026"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_manifest_enforces_internal_operational_readouts() -> None:
    runner = _load_module(
        "operational_force_harness_runner",
        HARNESS_DIR / "run_unified_force_harness.py",
    )
    version, specs, payload = runner._load_manifest(runner.MANIFEST_PATH)
    assert version == "2.0.0-operational"
    assert payload["schema_version"] == "2.0"
    assert runner.MANIFEST_PATH.name == "operational_emergence_manifest.json"
    operational = [
        spec
        for spec in specs
        if spec.required_for_sector and spec.operational_readout_required
    ]
    assert operational
    assert all(
        spec.readout_frame is ReadoutFrame.INTERNAL_OPERATIONAL
        and spec.substrate_evolution
        and spec.internal_observable
        and spec.continuum_interpretation
        for spec in operational
    )
    forbidden_readout_fragments = (
        "external coordinate",
        "external grid",
        "fixed grid",
        "god eye",
        "grid coordinate",
        "lattice coordinate",
        "raw coordinate",
        "raw simulator",
        "simulator time",
    )
    assert all(
        not any(fragment in spec.internal_observable.lower() for fragment in forbidden_readout_fragments)
        for spec in operational
    )
    em_maxwell = next(
        spec for spec in specs if spec.benchmark_id == "EM-MAXWELL-CONTINUUM-CLOSURE"
    )
    assert "E-wave observers" in em_maxwell.internal_observable
    assert "chi-clock compensated" in em_maxwell.internal_observable
    required_ids = {
        spec.benchmark_id for spec in specs if spec.required_for_sector
    }
    assert "GR-FRAME-PROVENANCE" not in required_ids
    assert "STRONG-GAUGE-INVARIANT-OBSERVABLE" not in required_ids


def test_v1_manifest_is_marked_legacy() -> None:
    payload = json.loads(
        (HARNESS_DIR / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["status"].startswith("LEGACY_V1")
    assert payload["superseded_by"] == "operational_emergence_manifest.json"


def test_pure_ab_entrypoint_cannot_promote_force_diagnostics() -> None:
    runner = _load_module(
        "pure_gov02_diagnostic_runner",
        HARNESS_DIR / "run_pure_gov02_ab_force_harness.py",
    )
    force_specs = [
        spec
        for spec in runner._specs()
        if spec.sector is not ForceSector.CORE
    ]
    assert force_specs
    assert all(not spec.required_for_sector for spec in force_specs)
    assert all(not spec.promotion_eligible for spec in force_specs)
    assert all(
        spec.accepted_evidence == frozenset({EvidenceClass.DIAGNOSTIC})
        for spec in force_specs
    )
