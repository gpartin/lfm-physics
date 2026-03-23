"""Simulation planning utilities.

Helpers for answering practical UX questions before running a simulation:
- Which config should I start with for a given physics goal?
- Will this configuration fit on my hardware?
- Is a request infeasible because of physics scale or current library limits?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import KAPPA, KAPPA_C, KAPPA_STRING, LAMBDA_H

UseCaseName = Literal[
    "intro_gravity",
    "electromagnetism_charges",
    "strong_force_color",
    "cosmic_structure",
    "matter_creation_resonance",
    "rotating_galaxy",
    "particle_collision",
    "string_tension",
]


@dataclass(frozen=True)
class FeasibilityReport:
    """Summary of hardware feasibility for a candidate simulation.

    Notes
    -----
    Memory estimates are conservative, backend-level storage estimates for
    core evolution arrays only. Analysis arrays, plotting buffers, and Python
    overhead are not fully modeled.
    """

    estimated_memory_gb: float
    recommended_backend: str
    fits_cpu: bool
    fits_gpu: bool | None
    status: str
    reason: str


def _psi_component_factor(field_level: FieldLevel, n_colors: int) -> int:
    if field_level == FieldLevel.REAL:
        return 1
    if field_level == FieldLevel.COMPLEX:
        return 1
    return n_colors


def estimate_memory_gb(config: SimulationConfig) -> float:
    """Estimate core evolver memory footprint in GiB.

    The estimate mirrors array allocation in ``lfm.core.evolver.Evolver``:
    - 4 real-psi buffers (current + prev, double-buffered)
    - +4 imag-psi buffers for COMPLEX/COLOR
    - 4 chi buffers (current + prev, double-buffered)
    - 1 boundary mask (bool)

    Parameters
    ----------
    config : SimulationConfig
        Candidate simulation configuration.
    """
    n = config.grid_size
    cells = n**3
    bytes_f32 = 4

    psi_factor = _psi_component_factor(config.field_level, config.n_colors)

    psi_real_bytes = 4 * psi_factor * cells * bytes_f32
    psi_imag_bytes = 0
    if config.field_level != FieldLevel.REAL:
        psi_imag_bytes = 4 * psi_factor * cells * bytes_f32

    chi_bytes = 4 * cells * bytes_f32
    mask_bytes = cells  # bool mask

    total_bytes = psi_real_bytes + psi_imag_bytes + chi_bytes + mask_bytes
    gib = total_bytes / (1024**3)
    return float(gib)


def assess_feasibility(
    config: SimulationConfig,
    *,
    cpu_ram_gb: float = 32.0,
    gpu_vram_gb: float | None = 8.0,
) -> FeasibilityReport:
    """Assess whether a run is feasible on available hardware.

    Parameters
    ----------
    config : SimulationConfig
        Candidate simulation configuration.
    cpu_ram_gb : float
        Usable system memory budget for simulation arrays.
    gpu_vram_gb : float | None
        Usable VRAM budget. Set ``None`` to indicate no GPU available.
    """
    mem = estimate_memory_gb(config)

    # Reserve headroom for analysis buffers and runtime overhead.
    cpu_limit = 0.60 * cpu_ram_gb
    fits_cpu = mem <= cpu_limit

    if gpu_vram_gb is None:
        fits_gpu: bool | None = None
    else:
        gpu_limit = 0.70 * gpu_vram_gb
        fits_gpu = mem <= gpu_limit

    if not fits_cpu and (fits_gpu is False or fits_gpu is None):
        return FeasibilityReport(
            estimated_memory_gb=mem,
            recommended_backend="none",
            fits_cpu=False,
            fits_gpu=fits_gpu,
            status="infeasible",
            reason=(
                "Configuration exceeds safe memory headroom for available hardware. "
                "Reduce grid_size, simplify field_level, or split the problem into staged runs."
            ),
        )

    if fits_gpu:
        return FeasibilityReport(
            estimated_memory_gb=mem,
            recommended_backend="gpu",
            fits_cpu=fits_cpu,
            fits_gpu=fits_gpu,
            status="feasible",
            reason="Fits GPU VRAM headroom; GPU is recommended for throughput.",
        )

    return FeasibilityReport(
        estimated_memory_gb=mem,
        recommended_backend="cpu",
        fits_cpu=fits_cpu,
        fits_gpu=fits_gpu,
        status="feasible",
        reason="Fits CPU RAM headroom; prefer CPU or reduce size for GPU execution.",
    )


def use_case_preset(name: UseCaseName) -> SimulationConfig:
    """Return a ready-to-run configuration for common physics workflows."""
    if name == "intro_gravity":
        return SimulationConfig(
            grid_size=48,
            field_level=FieldLevel.REAL,
            boundary_type=BoundaryType.FROZEN,
            report_interval=500,
        )

    if name == "electromagnetism_charges":
        return SimulationConfig(
            grid_size=64,
            field_level=FieldLevel.COMPLEX,
            report_interval=500,
        )

    if name == "strong_force_color":
        return SimulationConfig(
            grid_size=64,
            field_level=FieldLevel.COLOR,
            kappa_c=1.0 / 189.0,
            epsilon_cc=2.0 / 17.0,
            report_interval=500,
        )

    if name == "cosmic_structure":
        return SimulationConfig(
            grid_size=128,
            field_level=FieldLevel.REAL,
            report_interval=2000,
        )

    if name == "matter_creation_resonance":
        return SimulationConfig(
            grid_size=64,
            field_level=FieldLevel.REAL,
            phase1_steps=10_000,
            phase1_omega=38.0,
            phase1_amplitude=0.3,
            report_interval=500,
        )

    if name == "rotating_galaxy":
        return SimulationConfig(
            grid_size=128,
            field_level=FieldLevel.REAL,
            boundary_type=BoundaryType.FROZEN,
            report_interval=2000,
        )

    if name == "particle_collision":
        return SimulationConfig(
            grid_size=64,
            field_level=FieldLevel.COMPLEX,
            boundary_type=BoundaryType.FROZEN,
            report_interval=250,
        )

    if name == "string_tension":
        return SimulationConfig(
            grid_size=64,
            field_level=FieldLevel.COLOR,
            kappa_c=KAPPA_C,
            kappa_tube=10.0 * KAPPA,
            kappa_string=KAPPA_STRING,
            lambda_self=LAMBDA_H,
            report_interval=500,
        )

    raise ValueError(f"Unknown use-case preset: {name}")


def scale_limit_note() -> str:
    """Return guidance for intrinsically infeasible one-grid requests."""
    return (
        "Single-grid all-scale simulation (subatomic to cosmic simultaneously) is not "
        "a practical target for current hardware or algorithms. Use staged multiscale "
        "workflows: local high-resolution runs + coarse cosmological runs + calibrated "
        "bridging observables."
    )


__all__ = [
    "FeasibilityReport",
    "UseCaseName",
    "assess_feasibility",
    "estimate_memory_gb",
    "scale_limit_note",
    "use_case_preset",
]
