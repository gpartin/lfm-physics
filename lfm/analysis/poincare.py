"""Poincare-emergence diagnostics for the LFM cubic substrate.

The finite lattice has exact integer translations and cubic rotations, not the
continuous Poincare group.  This module quantifies how the exact lattice
dispersion approaches the continuum mass shell and how rotation and boost
defects vanish in the long-wavelength limit.

The implementation is spectral.  It uses the exact symbols of the 19-point
and 27-point cubic stencils and the exact leapfrog time symbol, so it does not
duplicate the production GOV-01 or GOV-02 update loops.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CubicStencil:
    """Weights for a center/face/edge/corner cubic Laplacian."""

    name: str
    face_weight: float
    edge_weight: float
    corner_weight: float


STENCIL_19 = CubicStencil(
    name="19",
    face_weight=1.0 / 3.0,
    edge_weight=1.0 / 6.0,
    corner_weight=0.0,
)

STENCIL_27 = CubicStencil(
    name="27",
    face_weight=4.0 / 9.0,
    edge_weight=1.0 / 9.0,
    corner_weight=1.0 / 36.0,
)

STENCILS = {"19": STENCIL_19, "27": STENCIL_27}


def get_stencil(stencil: str | CubicStencil) -> CubicStencil:
    """Return a validated stencil specification."""

    if isinstance(stencil, CubicStencil):
        return stencil
    try:
        return STENCILS[str(stencil)]
    except KeyError as exc:
        raise ValueError(f"unknown stencil {stencil!r}; expected '19' or '27'") from exc


def _wavevectors(k: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(k, dtype=np.float64)
    if arr.shape == (3,):
        return arr
    if arr.ndim < 1 or arr.shape[-1] != 3:
        raise ValueError("wavevectors must have shape (3,) or (..., 3)")
    return arr


def stencil_symbol(
    k: ArrayLike,
    *,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> NDArray[np.float64]:
    """Return the exact Laplacian symbol lambda(k), including 1/spacing^2."""

    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    spec = get_stencil(stencil)
    wave = _wavevectors(k)
    q = wave * spacing
    cos_q = np.cos(q)
    face_sum = np.sum(cos_q, axis=-1)
    edge_sum = (
        cos_q[..., 0] * cos_q[..., 1]
        + cos_q[..., 0] * cos_q[..., 2]
        + cos_q[..., 1] * cos_q[..., 2]
    )
    corner_product = np.prod(cos_q, axis=-1)
    symbol = (
        2.0 * spec.face_weight * (face_sum - 3.0)
        + 4.0 * spec.edge_weight * (edge_sum - 3.0)
        + 8.0 * spec.corner_weight * (corner_product - 1.0)
    )
    return np.asarray(symbol / (spacing * spacing), dtype=np.float64)


def lattice_k_squared(
    k: ArrayLike,
    *,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> NDArray[np.float64]:
    """Return the nonnegative lattice momentum squared, -lambda(k)."""

    value = -stencil_symbol(k, spacing=spacing, stencil=stencil)
    return np.maximum(value, 0.0)


def discrete_omega(
    k: ArrayLike,
    *,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> NDArray[np.float64]:
    """Return the exact positive-frequency leapfrog branch."""

    if dt <= 0.0 or c <= 0.0 or mass < 0.0:
        raise ValueError("dt and c must be positive and mass must be nonnegative")
    k2 = lattice_k_squared(k, spacing=spacing, stencil=stencil)
    frequency_sq = c * c * k2 + mass * mass
    argument = 0.5 * dt * np.sqrt(frequency_sq)
    if np.any(argument > 1.0 + 1.0e-13):
        raise ValueError("requested mode is outside the stable leapfrog branch")
    return 2.0 * np.arcsin(np.clip(argument, 0.0, 1.0)) / dt


def _lattice_k_squared_gradient(
    k: ArrayLike,
    *,
    spacing: float,
    stencil: str | CubicStencil,
) -> NDArray[np.float64]:
    spec = get_stencil(stencil)
    wave = _wavevectors(k)
    q = wave * spacing
    sin_q = np.sin(q)
    cos_q = np.cos(q)
    gradients = np.empty_like(wave, dtype=np.float64)
    for axis in range(3):
        other = [candidate for candidate in range(3) if candidate != axis]
        gradients[..., axis] = (
            sin_q[..., axis]
            * (
                2.0 * spec.face_weight
                + 4.0 * spec.edge_weight * (cos_q[..., other[0]] + cos_q[..., other[1]])
                + 8.0 * spec.corner_weight * cos_q[..., other[0]] * cos_q[..., other[1]]
            )
            / spacing
        )
    return gradients


def group_velocity(
    k: ArrayLike,
    *,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> NDArray[np.float64]:
    """Return the exact group-velocity vector for the leapfrog branch."""

    wave = _wavevectors(k)
    k2 = lattice_k_squared(wave, spacing=spacing, stencil=stencil)
    frequency_sq = c * c * k2 + mass * mass
    root = np.sqrt(frequency_sq)
    phase_factor_sq = 1.0 - 0.25 * dt * dt * frequency_sq
    if np.any(phase_factor_sq <= 0.0):
        raise ValueError("group velocity is undefined at or above the leapfrog branch edge")
    denominator = 2.0 * root * np.sqrt(phase_factor_sq)
    gradient = _lattice_k_squared_gradient(wave, spacing=spacing, stencil=stencil)
    velocity = np.zeros_like(gradient)
    np.divide(
        c * c * gradient,
        np.expand_dims(denominator, axis=-1),
        out=velocity,
        where=np.expand_dims(denominator > 0.0, axis=-1),
    )
    return velocity


def fibonacci_sphere(count: int) -> NDArray[np.float64]:
    """Return deterministic near-uniform unit vectors on the two-sphere."""

    if count < 6:
        raise ValueError("count must be at least 6")
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    phi = math.pi * (3.0 - math.sqrt(5.0)) * index
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def shell_metrics(
    k_magnitude: float,
    *,
    directions: ArrayLike | None = None,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> dict[str, float]:
    """Measure dispersion and directional errors on one physical k-shell."""

    if k_magnitude <= 0.0:
        raise ValueError("k_magnitude must be positive")
    unit = fibonacci_sphere(512) if directions is None else _wavevectors(directions)
    norms = np.linalg.norm(unit, axis=-1)
    if np.any(norms <= 0.0):
        raise ValueError("directions must be nonzero")
    unit = unit / norms[..., None]
    wave = k_magnitude * unit
    omega = discrete_omega(
        wave,
        mass=mass,
        c=c,
        dt=dt,
        spacing=spacing,
        stencil=stencil,
    )
    continuum_omega = math.sqrt(c * c * k_magnitude * k_magnitude + mass * mass)
    dispersion_error = np.abs(omega / continuum_omega - 1.0)

    velocity = group_velocity(
        wave,
        mass=mass,
        c=c,
        dt=dt,
        spacing=spacing,
        stencil=stencil,
    )
    radial_velocity = np.sum(velocity * unit, axis=-1)
    continuum_velocity = c * c * k_magnitude / continuum_omega
    group_error = np.abs(radial_velocity / continuum_velocity - 1.0)
    anisotropy = (np.max(radial_velocity) - np.min(radial_velocity)) / np.mean(
        np.abs(radial_velocity)
    )
    transverse = np.linalg.norm(velocity - radial_velocity[..., None] * unit, axis=-1)

    return {
        "dispersion_error_mean": float(np.mean(dispersion_error)),
        "dispersion_error_max": float(np.max(dispersion_error)),
        "group_velocity_error_mean": float(np.mean(group_error)),
        "group_velocity_error_max": float(np.max(group_error)),
        "directional_anisotropy": float(anisotropy),
        "transverse_group_velocity_max_over_c": float(np.max(transverse) / c),
        "radial_velocity_mean_over_c": float(np.mean(radial_velocity) / c),
    }


def mass_shell_residual(
    omega: ArrayLike,
    k: ArrayLike,
    *,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> NDArray[np.float64]:
    """Evaluate the exact discrete mass-shell function."""

    omega_arr = np.asarray(omega, dtype=np.float64)
    temporal = 4.0 * np.sin(0.5 * omega_arr * dt) ** 2 / (dt * dt)
    spatial = c * c * lattice_k_squared(k, spacing=spacing, stencil=stencil)
    return temporal - spatial - mass * mass


def lorentz_boost_wavevector(
    omega: ArrayLike,
    k: ArrayLike,
    *,
    beta: float,
    axis: int = 0,
    c: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a continuum passive Lorentz boost to a frequency-wavevector pair."""

    if not 0 <= axis < 3:
        raise ValueError("axis must be 0, 1, or 2")
    if abs(beta) >= 1.0:
        raise ValueError("abs(beta) must be less than one")
    wave = np.array(_wavevectors(k), copy=True)
    omega_arr = np.asarray(omega, dtype=np.float64)
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    boosted_omega = gamma * (omega_arr - beta * c * wave[..., axis])
    boosted_axis = gamma * (wave[..., axis] - beta * omega_arr / c)
    wave[..., axis] = boosted_axis
    return boosted_omega, wave


def boost_covariance_metrics(
    k: ArrayLike,
    *,
    beta: float,
    axis: int = 0,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> dict[str, float]:
    """Measure mass-shell and velocity-addition defects under a continuum boost."""

    wave = _wavevectors(k)
    omega = discrete_omega(
        wave,
        mass=mass,
        c=c,
        dt=dt,
        spacing=spacing,
        stencil=stencil,
    )
    boosted_omega, boosted_wave = lorentz_boost_wavevector(
        omega,
        wave,
        beta=beta,
        axis=axis,
        c=c,
    )
    residual = mass_shell_residual(
        boosted_omega,
        boosted_wave,
        mass=mass,
        c=c,
        dt=dt,
        spacing=spacing,
        stencil=stencil,
    )
    normalization = np.maximum(boosted_omega * boosted_omega + mass * mass, 1.0e-30)

    velocity = group_velocity(
        wave,
        mass=mass,
        c=c,
        dt=dt,
        spacing=spacing,
        stencil=stencil,
    )
    actual = group_velocity(
        boosted_wave,
        mass=mass,
        c=c,
        dt=dt,
        spacing=spacing,
        stencil=stencil,
    )
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    denominator = 1.0 - beta * velocity[..., axis] / c
    expected = np.array(velocity, copy=True)
    expected[..., axis] = (velocity[..., axis] - beta * c) / denominator
    for component in range(3):
        if component != axis:
            expected[..., component] = velocity[..., component] / (gamma * denominator)
    addition_error = np.linalg.norm(actual - expected, axis=-1) / c

    return {
        "boosted_mass_shell_residual_mean": float(np.mean(np.abs(residual) / normalization)),
        "boosted_mass_shell_residual_max": float(np.max(np.abs(residual) / normalization)),
        "velocity_addition_error_mean_over_c": float(np.mean(addition_error)),
        "velocity_addition_error_max_over_c": float(np.max(addition_error)),
    }


def cubic_rotation_residual(
    k: ArrayLike,
    *,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> float:
    """Return the maximum symbol change over all 48 signed axis permutations."""

    wave = np.asarray(_wavevectors(k), dtype=np.float64)
    reference = stencil_symbol(wave, spacing=spacing, stencil=stencil)
    maximum = 0.0
    for permutation in itertools.permutations(range(3)):
        permuted = wave[..., permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            transformed = permuted * np.asarray(signs)
            value = stencil_symbol(transformed, spacing=spacing, stencil=stencil)
            maximum = max(maximum, float(np.max(np.abs(value - reference))))
    return maximum


def max_spatial_eigenvalue(
    *,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> float:
    """Return max(-lambda) over the Brillouin zone.

    The symbol is multilinear in cos(k_i spacing), so its extrema occur at
    the eight Brillouin-zone corners.
    """

    corners = np.asarray(list(itertools.product((0.0, math.pi / spacing), repeat=3)))
    return float(np.max(lattice_k_squared(corners, spacing=spacing, stencil=stencil)))


def leapfrog_stability_limit(
    *,
    mass: float = 0.0,
    c: float = 1.0,
    spacing: float = 1.0,
    stencil: str | CubicStencil = "19",
) -> float:
    """Return the exact linear leapfrog timestep bound."""

    maximum = max_spatial_eigenvalue(spacing=spacing, stencil=stencil)
    return 2.0 / math.sqrt(c * c * maximum + mass * mass)


def symanzik_coefficients(stencil: str | CubicStencil) -> dict[str, float]:
    """Return the small-spacing symbol coefficients through sixth order.

    The symbol is
      lambda = -k^2 + h^2 k^4/12
               - h^4[a sum(k_i^6)
                       + b sum_{i != j}(k_i^4 k_j^2)
                       + d k_x^2 k_y^2 k_z^2] + O(h^6).
    """

    spec = get_stencil(stencil)
    pure = spec.face_weight / 360.0 + spec.edge_weight / 90.0
    pure += spec.corner_weight / 90.0
    mixed_ordered = spec.edge_weight / 12.0 + spec.corner_weight / 6.0
    triple = spec.corner_weight
    return {
        "quadratic": 1.0,
        "quartic_pure": 1.0 / 12.0,
        "quartic_mixed": 1.0 / 6.0,
        "sixth_pure": pure,
        "sixth_mixed_ordered": mixed_ordered,
        "sixth_triple": triple,
        "isotropic_sixth_pure": 1.0 / 360.0,
        "isotropic_sixth_mixed_ordered": 1.0 / 120.0,
        "isotropic_sixth_triple": 1.0 / 60.0,
    }


def poincare_algebra_matrix_residual() -> dict[str, float]:
    """Verify a 5x5 affine representation of the Poincare Lie algebra."""

    rotations = []
    boosts = []
    translations = []
    for mu in range(4):
        generator = np.zeros((5, 5), dtype=np.float64)
        generator[mu, 4] = 1.0
        translations.append(generator)

    for axis in range(3):
        rotation = np.zeros((5, 5), dtype=np.float64)
        first = 1 + (axis + 1) % 3
        second = 1 + (axis + 2) % 3
        rotation[first, second] = -1.0
        rotation[second, first] = 1.0
        rotations.append(rotation)

        boost = np.zeros((5, 5), dtype=np.float64)
        boost[0, axis + 1] = 1.0
        boost[axis + 1, 0] = 1.0
        boosts.append(boost)

    def commutator(left: NDArray[np.float64], right: NDArray[np.float64]):
        return left @ right - right @ left

    epsilon = np.zeros((3, 3, 3), dtype=np.float64)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[1, 0, 2] = epsilon[2, 1, 0] = epsilon[0, 2, 1] = -1.0

    residuals: dict[str, float] = {}
    checks: dict[str, list[NDArray[np.float64]]] = {
        "P_P": [],
        "J_J": [],
        "J_K": [],
        "K_K": [],
        "J_P": [],
        "J_H": [],
        "K_H": [],
        "K_P": [],
    }
    for mu in range(4):
        for nu in range(4):
            checks["P_P"].append(commutator(translations[mu], translations[nu]))
    for i in range(3):
        checks["J_H"].append(commutator(rotations[i], translations[0]))
        checks["K_H"].append(commutator(boosts[i], translations[0]) - translations[i + 1])
        for j in range(3):
            expected_jj = sum(epsilon[i, j, k] * rotations[k] for k in range(3))
            expected_jk = sum(epsilon[i, j, k] * boosts[k] for k in range(3))
            expected_kk = -sum(epsilon[i, j, k] * rotations[k] for k in range(3))
            expected_jp = sum(epsilon[i, j, k] * translations[k + 1] for k in range(3))
            checks["J_J"].append(commutator(rotations[i], rotations[j]) - expected_jj)
            checks["J_K"].append(commutator(rotations[i], boosts[j]) - expected_jk)
            checks["K_K"].append(commutator(boosts[i], boosts[j]) - expected_kk)
            checks["J_P"].append(commutator(rotations[i], translations[j + 1]) - expected_jp)
            checks["K_P"].append(
                commutator(boosts[i], translations[j + 1]) - (translations[0] if i == j else 0.0)
            )

    for name, matrices in checks.items():
        residuals[name] = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    residuals["maximum"] = max(residuals.values())
    return residuals


def gaussian_packet(
    grid_size: int,
    *,
    length: float,
    center: Iterable[float],
    direction: Iterable[float],
    k_magnitude: float,
    sigma: float,
) -> NDArray[np.complex128]:
    """Construct a complex positive-frequency Gaussian packet on a 3D torus."""

    if grid_size < 8 or length <= 0.0 or sigma <= 0.0 or k_magnitude <= 0.0:
        raise ValueError("invalid packet geometry")
    center_arr = np.asarray(tuple(center), dtype=np.float64)
    direction_arr = np.asarray(tuple(direction), dtype=np.float64)
    if center_arr.shape != (3,) or direction_arr.shape != (3,):
        raise ValueError("center and direction must have three components")
    direction_arr = direction_arr / np.linalg.norm(direction_arr)
    coordinate = np.arange(grid_size, dtype=np.float64) * (length / grid_size)
    offsets = []
    for axis in range(3):
        delta = coordinate - center_arr[axis]
        delta = (delta + 0.5 * length) % length - 0.5 * length
        shape = [1, 1, 1]
        shape[axis] = grid_size
        offsets.append(delta.reshape(shape))
    radius_sq = offsets[0] ** 2 + offsets[1] ** 2 + offsets[2] ** 2
    phase = k_magnitude * sum(direction_arr[axis] * offsets[axis] for axis in range(3))
    return np.exp(-0.5 * radius_sq / (sigma * sigma) + 1j * phase)


def evolve_positive_frequency(
    field: NDArray[np.complexfloating],
    *,
    time: float,
    length: float,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    stencil: str | CubicStencil = "19",
) -> NDArray[np.complex128]:
    """Evolve a periodic 3D field with the exact positive-frequency branch."""

    array = np.asarray(field, dtype=np.complex128)
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise ValueError("field must be a cubic 3D array")
    grid_size = array.shape[0]
    spacing = length / grid_size
    frequencies = 2.0 * math.pi * np.fft.fftfreq(grid_size, d=spacing)
    kx, ky, kz = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    wave = np.stack((kx, ky, kz), axis=-1)
    omega = discrete_omega(
        wave,
        mass=mass,
        c=c,
        dt=dt,
        spacing=spacing,
        stencil=stencil,
    )
    spectrum = np.fft.fftn(array)
    evolved = np.fft.ifftn(spectrum * np.exp(-1j * omega * time))
    return np.asarray(evolved, dtype=np.complex128)


def evolve_continuum_positive_frequency(
    field: NDArray[np.complexfloating],
    *,
    time: float,
    length: float,
    mass: float = 0.0,
    c: float = 1.0,
) -> NDArray[np.complex128]:
    """Evolve the same periodic data with the continuum Klein-Gordon symbol."""

    array = np.asarray(field, dtype=np.complex128)
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise ValueError("field must be a cubic 3D array")
    grid_size = array.shape[0]
    spacing = length / grid_size
    frequencies = 2.0 * math.pi * np.fft.fftfreq(grid_size, d=spacing)
    kx, ky, kz = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    omega = np.sqrt(c * c * (kx * kx + ky * ky + kz * kz) + mass * mass)
    spectrum = np.fft.fftn(array)
    evolved = np.fft.ifftn(spectrum * np.exp(-1j * omega * time))
    return np.asarray(evolved, dtype=np.complex128)


def periodic_centroid(
    field: NDArray[np.complexfloating],
    *,
    length: float,
) -> NDArray[np.float64]:
    """Return the circular center of |field|^2 along each periodic axis."""

    density = np.abs(np.asarray(field)) ** 2
    total = float(np.sum(density))
    if total <= 0.0:
        raise ValueError("field has zero norm")
    coordinate = np.arange(density.shape[0], dtype=np.float64) * length / density.shape[0]
    phase = np.exp(2j * math.pi * coordinate / length)
    centroid = np.empty(3, dtype=np.float64)
    for axis in range(3):
        reduce_axes = tuple(candidate for candidate in range(3) if candidate != axis)
        marginal = np.sum(density, axis=reduce_axes)
        moment = np.sum(marginal * phase) / total
        centroid[axis] = (np.angle(moment) % (2.0 * math.pi)) * length / (2.0 * math.pi)
    return centroid


def periodic_displacement(
    final: ArrayLike,
    initial: ArrayLike,
    *,
    length: float,
) -> NDArray[np.float64]:
    """Return the minimum-image displacement on a periodic cube."""

    delta = np.asarray(final, dtype=np.float64) - np.asarray(initial, dtype=np.float64)
    return (delta + 0.5 * length) % length - 0.5 * length


def packet_propagation_metrics(
    grid_size: int,
    *,
    length: float,
    center: Iterable[float],
    direction: Iterable[float],
    k_magnitude: float,
    sigma: float,
    propagation_time: float,
    mass: float = 0.0,
    c: float = 1.0,
    courant: float = 0.2,
    stencil: str | CubicStencil = "19",
) -> dict[str, float | list[float]]:
    """Propagate one Gaussian packet and measure its centroid velocity."""

    spacing = length / grid_size
    dt = courant * spacing / c
    initial = gaussian_packet(
        grid_size,
        length=length,
        center=center,
        direction=direction,
        k_magnitude=k_magnitude,
        sigma=sigma,
    )
    evolved = evolve_positive_frequency(
        initial,
        time=propagation_time,
        length=length,
        mass=mass,
        c=c,
        dt=dt,
        stencil=stencil,
    )
    continuum_evolved = evolve_continuum_positive_frequency(
        initial,
        time=propagation_time,
        length=length,
        mass=mass,
        c=c,
    )
    center_initial = periodic_centroid(initial, length=length)
    center_final = periodic_centroid(evolved, length=length)
    center_continuum = periodic_centroid(continuum_evolved, length=length)
    displacement = periodic_displacement(center_final, center_initial, length=length)
    continuum_displacement = periodic_displacement(
        center_continuum,
        center_initial,
        length=length,
    )
    velocity = displacement / propagation_time
    continuum_velocity = continuum_displacement / propagation_time
    unit = np.asarray(tuple(direction), dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    radial = float(np.dot(velocity, unit))
    continuum_radial = float(np.dot(continuum_velocity, unit))
    transverse = float(np.linalg.norm(velocity - radial * unit))
    continuum_transverse = float(np.linalg.norm(continuum_velocity - continuum_radial * unit))
    return {
        "grid_size": grid_size,
        "spacing": spacing,
        "dt": dt,
        "velocity": velocity.tolist(),
        "radial_velocity_over_c": radial / c,
        "transverse_velocity_over_c": transverse / c,
        "continuum_velocity": continuum_velocity.tolist(),
        "continuum_radial_velocity_over_c": continuum_radial / c,
        "continuum_transverse_velocity_over_c": continuum_transverse / c,
        "radial_error": abs(radial / continuum_radial - 1.0),
        "norm_initial": float(np.sum(np.abs(initial) ** 2)),
        "norm_final": float(np.sum(np.abs(evolved) ** 2)),
        "continuum_norm_final": float(np.sum(np.abs(continuum_evolved) ** 2)),
    }


def translation_equivariance_residual(
    field: NDArray[np.complexfloating],
    *,
    shift: tuple[int, int, int],
    time: float,
    length: float,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    stencil: str | CubicStencil = "19",
) -> float:
    """Measure periodic integer-translation equivariance of spectral evolution."""

    evolved = evolve_positive_frequency(
        field,
        time=time,
        length=length,
        mass=mass,
        c=c,
        dt=dt,
        stencil=stencil,
    )
    shifted = np.roll(field, shift=shift, axis=(0, 1, 2))
    evolved_shifted = evolve_positive_frequency(
        shifted,
        time=time,
        length=length,
        mass=mass,
        c=c,
        dt=dt,
        stencil=stencil,
    )
    expected = np.roll(evolved, shift=shift, axis=(0, 1, 2))
    return float(np.linalg.norm(evolved_shifted - expected) / np.linalg.norm(expected))


def time_composition_residual(
    field: NDArray[np.complexfloating],
    *,
    time_1: float,
    time_2: float,
    length: float,
    mass: float = 0.0,
    c: float = 1.0,
    dt: float = 0.02,
    stencil: str | CubicStencil = "19",
) -> float:
    """Measure composition of the positive-frequency lattice evolution."""

    direct = evolve_positive_frequency(
        field,
        time=time_1 + time_2,
        length=length,
        mass=mass,
        c=c,
        dt=dt,
        stencil=stencil,
    )
    first = evolve_positive_frequency(
        field,
        time=time_1,
        length=length,
        mass=mass,
        c=c,
        dt=dt,
        stencil=stencil,
    )
    composed = evolve_positive_frequency(
        first,
        time=time_2,
        length=length,
        mass=mass,
        c=c,
        dt=dt,
        stencil=stencil,
    )
    return float(np.linalg.norm(direct - composed) / np.linalg.norm(direct))
