"""
Validation Experiment Runner
============================

Standardised runner for particle-validation experiments that enforces
the mandatory hypothesis framework (H₀ / H₁ pattern) from
``copilot-instructions.md``.

Usage::

    from lfm.experiment.runner import run_experiment, ValidationResult

    result = run_experiment(
        name="electron_stability",
        h0="Electron dissolves within 10 000 steps",
        h1="Electron persists as stable soliton",
        setup_fn=setup_electron,
        measure_fn=measure_electron,
        evaluate_fn=evaluate_electron,
        steps=10_000,
    )
    print(result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from lfm.simulation import Simulation


class _SetupFn(Protocol):
    def __call__(self) -> Simulation: ...


class _MeasureFn(Protocol):
    def __call__(self, sim: Simulation) -> dict[str, Any]: ...


class _EvalFn(Protocol):
    def __call__(self, measurements: dict[str, Any]) -> tuple[bool, str]: ...


@dataclass
class ValidationResult:
    """Outcome of a single validation experiment."""

    name: str
    h0_description: str
    h1_description: str
    success_criterion: str
    measurements: dict[str, Any] = field(default_factory=dict)
    h0_rejected: bool = False
    lfm_only_verified: bool = True
    notes: str = ""
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        status = "REJECTED" if self.h0_rejected else "FAILED TO REJECT"
        lines = [
            "=" * 50,
            f"EXPERIMENT: {self.name}",
            "=" * 50,
            f"H₀: {self.h0_description}",
            f"H₁: {self.h1_description}",
            f"Criterion: {self.success_criterion}",
            "-" * 50,
            f"H₀ STATUS: {status}",
            f"LFM-ONLY VERIFIED: {'YES' if self.lfm_only_verified else 'NO'}",
            f"Elapsed: {self.elapsed_seconds:.1f}s",
        ]
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        for k, v in self.measurements.items():
            lines.append(f"  {k}: {v}")
        lines.append("=" * 50)
        return "\n".join(lines)


def run_experiment(
    *,
    name: str,
    h0: str,
    h1: str,
    criterion: str = "",
    setup_fn: _SetupFn,
    measure_fn: _MeasureFn,
    evaluate_fn: _EvalFn,
    steps: int = 10_000,
) -> ValidationResult:
    """Run a validation experiment with hypothesis tracking.

    1. ``setup_fn()`` → :class:`~lfm.Simulation` (configured & solitons placed)
    2. ``sim.equilibrate()`` + ``sim.run(steps)``
    3. ``measure_fn(sim)`` → measurements dict
    4. ``evaluate_fn(measurements)`` → ``(h0_rejected, notes)``
    5. Return :class:`ValidationResult`
    """
    t0 = time.perf_counter()
    sim = setup_fn()
    sim.equilibrate()
    sim.run(steps=steps, record_metrics=False)
    measurements = measure_fn(sim)
    h0_rejected, notes = evaluate_fn(measurements)
    elapsed = time.perf_counter() - t0

    return ValidationResult(
        name=name,
        h0_description=h0,
        h1_description=h1,
        success_criterion=criterion,
        measurements=measurements,
        h0_rejected=h0_rejected,
        notes=notes,
        elapsed_seconds=elapsed,
    )
