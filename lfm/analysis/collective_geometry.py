"""Operational coarse-graining for the bare discrete LFM substrate.

The routines in this module do not define a metric or add evolution state.
They construct registered GOV-01 initial data and measure how an independently
tagged GOV-01 component propagates through the live GOV-01/GOV-02 substrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from lfm.analysis.energy_current import (
    BareLFMParameters,
    BareLFMState,
    LinkCurrentMap,
)
from lfm.core.stencils import eigenvalue_19pt, eigenvalue_27pt

if TYPE_CHECKING:
    from numpy.typing import NDArray

SOURCE_CASES = (
    "vacuum_probe",
    "static_sphere",
    "moving_plus",
    "moving_minus",
    "quadrupole_plus",
    "quadrupole_cross",
)


@dataclass(frozen=True)
class WeightedMoments:
    """Centroid, covariance, and total weight on a periodic cube."""

    centroid: np.ndarray
    covariance: np.ndarray
    total_weight: float


@dataclass(frozen=True)
class ContinuumFit:
    """Three-resolution fit y(h)=intercept+slope*h^2."""

    intercept: np.ndarray
    slope: np.ndarray
    intercept_standard_error: np.ndarray
    relative_standard_error: np.ndarray
    drop_one_relative_change: np.ndarray
    sign_consistent: np.ndarray


def minimum_image_mesh(
    size: int,
    length: float,
    center_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell-centered minimum-image coordinates about a shifted origin."""
    if size <= 0 or length <= 0.0:
        raise ValueError("size and length must be positive")
    spacing = length / size
    base = (np.arange(size, dtype=float) - size // 2) * spacing
    axes = []
    for shift in center_shift:
        coordinate = (base - shift + 0.5 * length) % length - 0.5 * length
        axes.append(coordinate)
    return cast(
        "tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]",
        np.meshgrid(*axes, indexing="ij"),
    )


def collective_initial_state(
    size: int,
    length: float,
    case: str,
    *,
    chi0: float = 19.0,
    source_amplitude: float = 20.0,
    probe_amplitude: float = 0.05,
    source_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    probe_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    motion_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    quadrupole_angle: float = 0.0,
) -> BareLFMState:
    """Construct the frozen six-real-component source/probe initial state."""
    if case not in SOURCE_CASES:
        raise ValueError(f"case must be one of {SOURCE_CASES}")
    if source_amplitude < 0.0 or probe_amplitude <= 0.0:
        raise ValueError("source amplitude must be nonnegative and probe positive")
    x, y, z = minimum_image_mesh(size, length, source_shift)
    radius_sq = x**2 + y**2 + z**2
    shape = (6, size, size, size)
    wave = np.zeros(shape, dtype=float)
    wave_momentum = np.zeros_like(wave)
    chi = np.full((size, size, size), chi0, dtype=float)
    chi_momentum = np.zeros_like(chi)

    if case != "vacuum_probe":
        sigma = 0.16 if case.startswith("quadrupole") else 0.12
        envelope = np.exp(-0.5 * radius_sq / sigma**2)
        omega = chi0
        if case == "moving_plus" or case == "moving_minus":
            sign = 1.0 if case == "moving_plus" else -1.0
            direction = np.asarray(motion_direction, dtype=float)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("motion_direction must be finite and nonzero")
            direction /= norm
            wave_number = sign * 4.0 * np.pi / length
            phase = wave_number * (direction[0] * x + direction[1] * y + direction[2] * z)
            wave[0] = source_amplitude * envelope * np.cos(phase)
            wave[1] = source_amplitude * envelope * np.sin(phase)
            omega = float(np.sqrt(chi0**2 + wave_number**2))
        elif case == "static_sphere":
            wave[0] = source_amplitude * envelope
        else:
            cosine = np.cos(quadrupole_angle)
            sine = np.sin(quadrupole_angle)
            rotated_x = cosine * x + sine * y
            rotated_y = -sine * x + cosine * y
            pattern = rotated_x**2 - rotated_y**2
            if case == "quadrupole_cross":
                pattern = 2.0 * rotated_x * rotated_y
            peak = float(np.max(np.abs(pattern * envelope)))
            if peak <= 0.0:
                raise ValueError("quadrupole pattern is unresolved")
            wave[0] = source_amplitude * pattern * envelope / peak
        wave_momentum[0] = omega * wave[1]
        wave_momentum[1] = -omega * wave[0]

    probe_x, probe_y, probe_z = minimum_image_mesh(size, length, probe_shift)
    probe_radius_sq = probe_x**2 + probe_y**2 + probe_z**2
    wave_momentum[4] = probe_amplitude * np.exp(-0.5 * probe_radius_sq / 0.08**2)
    return BareLFMState(
        wave=wave,
        wave_momentum=wave_momentum,
        chi=chi,
        chi_momentum=chi_momentum,
    )


def apply_momentum_sponge(
    state: BareLFMState,
    length: float,
    dt: float,
    *,
    start_fraction: float = 0.75,
    strength: float = 40.0,
) -> BareLFMState:
    """Apply an explicitly non-Hamiltonian absorbing boundary ablation.

    The sponge is a boundary-condition red-team tool, not part of GOV-01 or
    GOV-02.  It damps canonical momenta only in the outer cubical layer.
    """
    if not 0.0 < start_fraction < 1.0:
        raise ValueError("start_fraction must lie between zero and one")
    if length <= 0.0 or dt <= 0.0 or strength < 0.0:
        raise ValueError("length and dt must be positive and strength nonnegative")
    size = state.chi.shape[0]
    x, y, z = minimum_image_mesh(size, length)
    radial_fraction = np.maximum.reduce((np.abs(x), np.abs(y), np.abs(z))) / (0.5 * length)
    ramp = np.clip(
        (radial_fraction - start_fraction) / (1.0 - start_fraction),
        0.0,
        1.0,
    )
    damping = np.exp(-strength * ramp**2 * dt)
    return BareLFMState(
        wave=state.wave,
        wave_momentum=state.wave_momentum * damping[np.newaxis, ...],
        chi=state.chi,
        chi_momentum=state.chi_momentum * damping,
    )


def periodic_weighted_moments(
    weights: np.ndarray,
    length: float,
) -> WeightedMoments:
    """Measure centroid and covariance without a periodic-boundary seam."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 3 or len(set(values.shape)) != 1:
        raise ValueError("weights must have cubic shape (N,N,N)")
    if not np.all(np.isfinite(values)) or np.min(values) < -1.0e-14:
        raise ValueError("weights must be finite and nonnegative")
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    size = values.shape[0]
    phase = 2.0 * np.pi * np.arange(size, dtype=float) / size
    centroid_index = np.empty(3, dtype=float)
    for axis in range(3):
        marginal_axes = tuple(candidate for candidate in range(3) if candidate != axis)
        marginal = np.sum(values, axis=marginal_axes)
        phasor = np.sum(marginal * np.exp(1j * phase))
        angle = float(np.angle(phasor)) % (2.0 * np.pi)
        centroid_index[axis] = angle * size / (2.0 * np.pi)

    index = np.arange(size, dtype=float)
    delta_axes = []
    for center in centroid_index:
        delta_index = (index - center + 0.5 * size) % size - 0.5 * size
        delta_axes.append(delta_index * length / size)
    dx, dy, dz = np.meshgrid(*delta_axes, indexing="ij")
    deltas = (dx, dy, dz)
    covariance = np.empty((3, 3), dtype=float)
    for row in range(3):
        for column in range(3):
            covariance[row, column] = float(np.sum(values * deltas[row] * deltas[column]) / total)
    centroid = ((centroid_index - size // 2 + 0.5 * size) % size - 0.5 * size) * length / size
    return WeightedMoments(
        centroid=centroid,
        covariance=covariance,
        total_weight=total,
    )


def block_average(values: np.ndarray, factor: int) -> np.ndarray:
    """Average scalar/vector/tensor data over nonoverlapping cubic blocks."""
    array = np.asarray(values)
    if array.ndim < 3 or len(set(array.shape[-3:])) != 1:
        raise ValueError("the final three axes must form a cubic lattice")
    size = array.shape[-1]
    if factor <= 0 or size % factor != 0:
        raise ValueError("factor must divide the lattice size")
    blocks = size // factor
    prefix = array.shape[:-3]
    reshaped = array.reshape(prefix + (blocks, factor, blocks, factor, blocks, factor))
    offset = len(prefix)
    return np.mean(reshaped, axis=(offset + 1, offset + 3, offset + 5))


def energy_current_vector_and_tensor(
    currents: LinkCurrentMap,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert exact link currents into local vector and direction tensor."""
    if not currents or spacing <= 0.0:
        raise ValueError("currents must be nonempty and spacing positive")
    sample = next(iter(currents.values()))
    vector = np.zeros((3,) + sample.shape, dtype=float)
    tensor = np.zeros((3, 3) + sample.shape, dtype=float)
    magnitude = np.zeros(sample.shape, dtype=float)
    for offset, current in currents.items():
        direction = np.asarray(offset, dtype=float)
        unit = direction / np.linalg.norm(direction)
        vector += (
            0.5
            * current[np.newaxis, ...]
            * direction[:, np.newaxis, np.newaxis, np.newaxis]
            * spacing
        )
        absolute = 0.5 * np.abs(current)
        magnitude += absolute
        tensor += (
            np.outer(unit, unit)[:, :, np.newaxis, np.newaxis, np.newaxis]
            * absolute[np.newaxis, np.newaxis, ...]
        )
    tensor /= np.maximum(magnitude, np.finfo(float).tiny)[np.newaxis, np.newaxis, ...]
    return vector, tensor


def traceless(tensor: np.ndarray) -> np.ndarray:
    """Return the traceless part of arrays whose first axes are 3 by 3."""
    values = np.asarray(tensor, dtype=float)
    if values.shape[:2] != (3, 3):
        raise ValueError("tensor must begin with shape (3,3)")
    trace = np.trace(values, axis1=0, axis2=1) / 3.0
    result = values.copy()
    for axis in range(3):
        result[axis, axis] -= trace
    return result


def continuum_fit(spacings: np.ndarray, values: np.ndarray) -> ContinuumFit:
    """Fit three or more matched measurements to y(h)=a+b*h^2."""
    h = np.asarray(spacings, dtype=float)
    y = np.asarray(values, dtype=float)
    if h.ndim != 1 or h.size < 3 or y.shape[0] != h.size:
        raise ValueError("need at least three values with resolution on axis zero")
    design = np.column_stack((np.ones_like(h), h**2))
    flat = y.reshape(h.size, -1)
    coefficients, _, _, _ = np.linalg.lstsq(design, flat, rcond=None)
    fitted = design @ coefficients
    residual = flat - fitted
    degrees = h.size - 2
    variance = np.sum(residual**2, axis=0) / degrees
    covariance_factor = np.linalg.inv(design.T @ design)[0, 0]
    standard_error = np.sqrt(np.maximum(variance * covariance_factor, 0.0))
    intercept = coefficients[0]
    slope = coefficients[1]

    drop_changes = np.zeros_like(intercept)
    for dropped in range(h.size):
        keep = np.arange(h.size) != dropped
        reduced_design = design[keep]
        reduced_coefficients, _, _, _ = np.linalg.lstsq(
            reduced_design,
            flat[keep],
            rcond=None,
        )
        change = np.abs(reduced_coefficients[0] - intercept) / np.maximum(
            np.abs(intercept),
            np.finfo(float).tiny,
        )
        drop_changes = np.maximum(drop_changes, change)
    nonzero = np.abs(flat) > 100.0 * np.finfo(float).eps
    positive = np.all((flat > 0.0) | ~nonzero, axis=0)
    negative = np.all((flat < 0.0) | ~nonzero, axis=0)
    output_shape = y.shape[1:]
    return ContinuumFit(
        intercept=intercept.reshape(output_shape),
        slope=slope.reshape(output_shape),
        intercept_standard_error=standard_error.reshape(output_shape),
        relative_standard_error=(
            standard_error / np.maximum(np.abs(intercept), np.finfo(float).tiny)
        ).reshape(output_shape),
        drop_one_relative_change=drop_changes.reshape(output_shape),
        sign_consistent=(positive | negative).reshape(output_shape),
    )


def dispersion_shell_metrics(
    stencil: str,
    spacing: float,
    physical_wave_number: float,
    *,
    mass: float = 19.0,
    wave_speed: float = 1.0,
) -> dict[str, float]:
    """Compare equal-|k| axis, face-diagonal, and body-diagonal modes."""
    if stencil == "19":
        eigenvalue = eigenvalue_19pt
    elif stencil == "27":
        eigenvalue = eigenvalue_27pt
    else:
        raise ValueError("stencil must be '19' or '27'")
    directions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0),
            (1.0 / np.sqrt(3.0),) * 3,
        )
    )
    dimensionless = physical_wave_number * spacing * directions
    lambdas = np.asarray(
        [eigenvalue(*wave_vector) for wave_vector in dimensionless],
        dtype=float,
    )
    frequencies = np.sqrt(mass**2 - wave_speed**2 * lambdas / spacing**2)
    continuum = float(np.sqrt(mass**2 + wave_speed**2 * physical_wave_number**2))
    relative = (frequencies - continuum) / continuum
    return {
        "axis_relative_error": float(relative[0]),
        "face_relative_error": float(relative[1]),
        "body_relative_error": float(relative[2]),
        "directional_anisotropy": float(
            (np.max(frequencies) - np.min(frequencies)) / np.mean(frequencies)
        ),
    }


def analytic_leapfrog_limit(
    parameters: BareLFMParameters,
    *,
    symbol_samples: int = 65,
) -> dict[str, float]:
    """Return the conservative vacuum Verlet limit from both lattice symbols."""
    wave_numbers = np.linspace(-np.pi, np.pi, symbol_samples)
    kx, ky, kz = np.meshgrid(wave_numbers, wave_numbers, wave_numbers, indexing="ij")
    eigenvalues = {"19": eigenvalue_19pt, "27": eigenvalue_27pt}
    wave_mu = float(-np.min(eigenvalues[parameters.gov01_stencil](kx, ky, kz)))
    chi_mu = float(-np.min(eigenvalues[parameters.gov02_stencil](kx, ky, kz)))
    wave_omega_sq = parameters.chi0**2 + parameters.wave_speed**2 * wave_mu / parameters.spacing**2
    chi_mass_sq = (
        8.0 * parameters.lambda_h * parameters.chi0**2
        if parameters.chi_potential == "quartic"
        else 0.0
    )
    chi_omega_sq = chi_mass_sq + parameters.wave_speed**2 * chi_mu / parameters.spacing**2
    maximum_omega = float(np.sqrt(max(wave_omega_sq, chi_omega_sq)))
    return {
        "wave_symbol_max": wave_mu,
        "chi_symbol_max": chi_mu,
        "maximum_vacuum_omega": maximum_omega,
        "maximum_dt": 2.0 / maximum_omega,
        "maximum_courant": 2.0 / (maximum_omega * parameters.spacing),
    }
