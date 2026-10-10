"""Diagnostics for the unpromoted LFM temporal-link geometry candidate.

The functions in this module analyze a proposed positive local clock factor
``q = exp(varphi)`` coupled to the complete bare LFM Hamiltonian. They do not
add the candidate register to :class:`lfm.Simulation` and do not implement a
gravitational force or trajectory law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfm.constants import C_DEFAULT, CHI0, KAPPA
from lfm.core.stencils import (
    eigenvalue_19pt,
    eigenvalue_27pt,
    laplacian_19pt,
    laplacian_27pt,
)


@dataclass(frozen=True)
class ClockLinkParameters:
    """Positive parameters for the candidate clock-link quadratic sector."""

    inertia: float = CHI0 / KAPPA
    speed: float = C_DEFAULT

    def __post_init__(self) -> None:
        if not np.isfinite(self.inertia) or self.inertia <= 0.0:
            raise ValueError("clock-link inertia must be positive and finite")
        if not np.isfinite(self.speed) or self.speed <= 0.0:
            raise ValueError("clock-link speed must be positive and finite")


def clock_link_stiffness(
    stencil: str,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
) -> np.ndarray:
    """Return nonnegative clock-link stiffness for a labeled stencil."""
    if stencil == "19":
        eigenvalue = eigenvalue_19pt(kx, ky, kz)
    elif stencil == "27":
        eigenvalue = eigenvalue_27pt(kx, ky, kz)
    else:
        raise ValueError("stencil must be '19' or '27'")
    stiffness = -np.asarray(eigenvalue, dtype=np.float64)
    return np.maximum(stiffness, 0.0)


def clock_link_frequency_sq(
    stiffness: np.ndarray | float,
    parameters: ClockLinkParameters = ClockLinkParameters(),
) -> np.ndarray:
    """Return the candidate vacuum branch ``omega^2 = c_phi^2 K``."""
    values = np.asarray(stiffness, dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError("stiffness must be nonnegative")
    return parameters.speed**2 * values


def clock_link_static_response(
    stiffness: np.ndarray | float,
    source: np.ndarray | float = 1.0,
    parameters: ClockLinkParameters = ClockLinkParameters(),
) -> np.ndarray:
    """Return the nonzero-mode static response per candidate field equation."""
    values = np.asarray(stiffness, dtype=np.float64)
    source_values = np.asarray(source, dtype=np.float64)
    if np.any(values <= 0.0):
        raise ValueError("static response requires strictly positive stiffness")
    return -source_values / (parameters.inertia * parameters.speed**2 * values)


def clock_link_green_residue(
    parameters: ClockLinkParameters = ClockLinkParameters(),
) -> float:
    """Return ``K*varphi/rho`` for the static candidate response."""
    return -1.0 / (parameters.inertia * parameters.speed**2)


def clock_factor(varphi: np.ndarray | float) -> np.ndarray:
    """Return the positive local temporal-link factor ``exp(varphi)``."""
    return np.exp(np.asarray(varphi, dtype=np.float64))


def matter_frequency_sq(
    stiffness: np.ndarray | float,
    chi: np.ndarray | float,
    varphi: np.ndarray | float,
    *,
    matter_speed: float = C_DEFAULT,
) -> np.ndarray:
    """Return uniform-clock candidate GOV-01 dispersion."""
    stiffness_values = np.asarray(stiffness, dtype=np.float64)
    if np.any(stiffness_values < 0.0):
        raise ValueError("stiffness must be nonnegative")
    if not np.isfinite(matter_speed) or matter_speed <= 0.0:
        raise ValueError("matter_speed must be positive and finite")
    chi_values = np.asarray(chi, dtype=np.float64)
    return np.exp(2.0 * np.asarray(varphi, dtype=np.float64)) * (
        matter_speed**2 * stiffness_values + chi_values**2
    )


def matter_clock_sensitivity(
    stiffness: np.ndarray | float,
    chi: np.ndarray | float,
    *,
    matter_speed: float = C_DEFAULT,
) -> np.ndarray:
    """Return ``d omega^2/d varphi`` at the ordinary clock vacuum."""
    stiffness_values = np.asarray(stiffness, dtype=np.float64)
    if np.any(stiffness_values < 0.0):
        raise ValueError("stiffness must be nonnegative")
    chi_values = np.asarray(chi, dtype=np.float64)
    return 2.0 * (matter_speed**2 * stiffness_values + chi_values**2)


def solve_static_clock_link(
    source: np.ndarray,
    *,
    stencil: str = "19",
    parameters: ClockLinkParameters = ClockLinkParameters(),
    remove_mean: bool = True,
) -> np.ndarray:
    """Solve the candidate static clock-link equation on a periodic 3-D grid.

    The solved equation is

    ``c_phi^2 * Laplacian(varphi) = source / B_phi``.

    A periodic solution requires a zero-mean source. By default the uniform
    source mode is removed and the returned field has zero mean.
    """
    source_values = np.asarray(source, dtype=np.float64)
    if source_values.ndim != 3:
        raise ValueError("source must be a three-dimensional array")
    if not np.all(np.isfinite(source_values)):
        raise ValueError("source must contain only finite values")
    effective_source = (
        source_values - float(np.mean(source_values)) if remove_mean else source_values.copy()
    )
    if not remove_mean and abs(float(np.mean(effective_source))) > 1.0e-14:
        raise ValueError("periodic static source must have zero mean")

    shape = effective_source.shape
    kx = np.fft.fftfreq(shape[0]) * 2.0 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2.0 * np.pi
    kz = np.fft.fftfreq(shape[2]) * 2.0 * np.pi
    grid_kx, grid_ky, grid_kz = np.meshgrid(
        kx,
        ky,
        kz,
        indexing="ij",
        sparse=True,
    )
    stiffness = clock_link_stiffness(
        stencil,
        grid_kx,
        grid_ky,
        grid_kz,
    )
    source_hat = np.fft.fftn(effective_source)
    field_hat = np.zeros_like(source_hat, dtype=np.complex128)
    nonzero = stiffness > 1.0e-14
    field_hat[nonzero] = -source_hat[nonzero] / (
        parameters.inertia * parameters.speed**2 * stiffness[nonzero]
    )
    field = np.fft.ifftn(field_hat).real
    field -= float(np.mean(field))
    return field


def static_clock_link_residual(
    field: np.ndarray,
    source: np.ndarray,
    *,
    stencil: str = "19",
    parameters: ClockLinkParameters = ClockLinkParameters(),
    remove_mean: bool = True,
) -> np.ndarray:
    """Return the real-space residual of the candidate static equation."""
    field_values = np.asarray(field, dtype=np.float64)
    source_values = np.asarray(source, dtype=np.float64)
    if field_values.shape != source_values.shape or field_values.ndim != 3:
        raise ValueError("field and source must have the same 3-D shape")
    effective_source = (
        source_values - float(np.mean(source_values)) if remove_mean else source_values
    )
    if stencil == "19":
        laplacian = laplacian_19pt(field_values)
    elif stencil == "27":
        laplacian = laplacian_27pt(field_values)
    else:
        raise ValueError("stencil must be '19' or '27'")
    return parameters.speed**2 * laplacian - effective_source / parameters.inertia
