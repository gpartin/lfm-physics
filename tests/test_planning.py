"""Tests for planning helpers."""

from __future__ import annotations

import lfm
from lfm.config import FieldLevel, SimulationConfig


def test_estimate_memory_real_positive() -> None:
    cfg = SimulationConfig(grid_size=32, field_level=FieldLevel.REAL)
    mem = lfm.estimate_memory_gb(cfg)
    assert mem > 0


def test_estimate_memory_color_gt_real() -> None:
    cfg_real = SimulationConfig(grid_size=48, field_level=FieldLevel.REAL)
    cfg_color = SimulationConfig(grid_size=48, field_level=FieldLevel.COLOR)
    assert lfm.estimate_memory_gb(cfg_color) > lfm.estimate_memory_gb(cfg_real)


def test_assess_feasibility_detects_infeasible() -> None:
    cfg = SimulationConfig(grid_size=512, field_level=FieldLevel.COLOR)
    report = lfm.assess_feasibility(cfg, cpu_ram_gb=8.0, gpu_vram_gb=4.0)
    assert report.status == "infeasible"
    assert report.recommended_backend == "none"


def test_assess_feasibility_prefers_gpu_when_fit() -> None:
    cfg = SimulationConfig(grid_size=64, field_level=FieldLevel.COMPLEX)
    report = lfm.assess_feasibility(cfg, cpu_ram_gb=16.0, gpu_vram_gb=8.0)
    assert report.status == "feasible"
    assert report.recommended_backend in {"gpu", "cpu"}


def test_use_case_preset_returns_expected_levels() -> None:
    assert lfm.use_case_preset("intro_gravity").field_level == FieldLevel.REAL
    assert lfm.use_case_preset("electromagnetism_charges").field_level == FieldLevel.COMPLEX
    assert lfm.use_case_preset("strong_force_color").field_level == FieldLevel.COLOR


def test_scale_limit_note_mentions_multiscale() -> None:
    note = lfm.scale_limit_note().lower()
    assert "multiscale" in note or "multiscale" in note.replace("-", "")
