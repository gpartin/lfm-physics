"""Static Cartan-sector variational tools for the R4 color dielectric.

The routines minimize the existing R4 electric-plus-chi Hamiltonian at fixed
external color charge. They do not add a potential, flux path, or string
tension. The electric field is obtained from the minimum-energy periodic
Gauss constraint for the current local dielectric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from lfm.core.stencils import laplacian_19pt, laplacian_27pt
from lfm.foundations.r3_link_frame_live import _link_table
from lfm.foundations.r4_unified_live import (
    R4Parameters,
    color_dielectric,
)


@dataclass
class R4ColorStaticState:
    """One relaxed R4 Cartan electric/chi configuration."""

    chi: np.ndarray
    potential: np.ndarray
    electric: np.ndarray
    charge: np.ndarray
    gauss_residual: float
    energy: float
    energy_parts: dict[str, float]
    iterations: int


@dataclass
class R4FixedColorElectricState:
    """R4 chi minimum for a fixed divergence-free color electric field."""

    chi: np.ndarray
    electric: np.ndarray
    gauss_residual: float
    energy: float
    energy_parts: dict[str, float]
    effective_g_squared: float
    iterations: int


def r4_color_vacuum_instability_flux_sq(
    parameters: R4Parameters = R4Parameters(),
) -> float:
    """Return the local incident-flux threshold for chi=chi0 instability."""

    chi0 = parameters.r3.chi0
    kappa = parameters.r3.kappa
    return float(
        4.0
        * parameters.r3.frame_inertia
        * parameters.r3.lambda_h
        * kappa**2
        * chi0**4
        / (1.0 - kappa)
    )


def _laplacian(values: np.ndarray, stencil: str) -> np.ndarray:
    if stencil == "19":
        return laplacian_19pt(values)
    if stencil == "27":
        return laplacian_27pt(values)
    raise ValueError("stencil must be '19' or '27'")


def _link_dielectric(
    chi: np.ndarray,
    parameters: R4Parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilon, derivative = color_dielectric(chi, parameters)
    unique, _ = _link_table(parameters.stencil)
    links = np.stack(
        [
            0.5
            * (
                epsilon
                + np.roll(
                    epsilon,
                    shift=tuple(-value for value in offset),
                    axis=(0, 1, 2),
                )
            )
            for offset, _ in unique
        ],
        axis=-1,
    )
    return epsilon, derivative, links


def _electric_from_potential(
    potential: np.ndarray,
    link_epsilon: np.ndarray,
    parameters: R4Parameters,
) -> np.ndarray:
    unique, _ = _link_table(parameters.stencil)
    electric = np.empty(potential.shape + (len(unique),))
    for index, (offset, _) in enumerate(unique):
        neighbor = np.roll(
            potential,
            shift=tuple(-value for value in offset),
            axis=(0, 1, 2),
        )
        electric[..., index] = (
            link_epsilon[..., index] * (potential - neighbor)
        )
    return electric


def color_gauss_divergence(
    electric: np.ndarray,
    parameters: R4Parameters,
) -> np.ndarray:
    """Return periodic outgoing-minus-incoming Cartan electric flux."""

    unique, _ = _link_table(parameters.stencil)
    expected = electric.shape[:-1] + (len(unique),)
    if electric.shape != expected:
        raise ValueError("electric link count does not match stencil")
    divergence = np.zeros(electric.shape[:-1], dtype=np.float64)
    for index, (offset, _) in enumerate(unique):
        outgoing = electric[..., index]
        incoming = np.roll(
            outgoing,
            shift=offset,
            axis=(0, 1, 2),
        )
        divergence += outgoing - incoming
    return divergence


def r4_color_incident_flux_sq(
    electric: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
) -> np.ndarray:
    """Return sum of squared outgoing and incoming Cartan link flux."""

    unique, _ = _link_table(parameters.stencil)
    if electric.ndim != 4 or electric.shape[-1] != len(unique):
        raise ValueError("electric must match the R4 link graph")
    result = np.sum(electric**2, axis=-1)
    for index, (offset, _) in enumerate(unique):
        result += np.roll(
            electric[..., index] ** 2,
            shift=offset,
            axis=(0, 1, 2),
        )
    return result


def solve_color_gauss_minimum(
    chi: np.ndarray,
    charge: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
    *,
    tolerance: float = 1.0e-10,
    initial_potential: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the minimum-electric-energy field satisfying Gauss charge."""

    chi_values = np.asarray(chi, dtype=np.float64)
    charge_values = np.asarray(charge, dtype=np.float64)
    if chi_values.ndim != 3 or charge_values.shape != chi_values.shape:
        raise ValueError("chi and charge must share a 3D shape")
    if abs(float(np.sum(charge_values))) > 1.0e-10:
        raise ValueError("periodic color charge must sum to zero")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    _, _, link_epsilon = _link_dielectric(
        chi_values,
        parameters,
    )
    shape = chi_values.shape
    count = int(np.prod(shape))

    def matvec(vector: np.ndarray) -> np.ndarray:
        potential = vector.reshape(shape)
        electric = _electric_from_potential(
            potential,
            link_epsilon,
            parameters,
        )
        result = color_gauss_divergence(
            electric,
            parameters,
        )
        result += np.mean(potential)
        return result.reshape(-1)

    operator = LinearOperator(
        (count, count),
        matvec=matvec,
        dtype=np.float64,
    )
    guess = (
        np.zeros(shape, dtype=np.float64)
        if initial_potential is None
        else np.asarray(initial_potential, dtype=np.float64)
    )
    if guess.shape != shape:
        raise ValueError("initial_potential must match chi")
    solution, info = cg(
        operator,
        charge_values.reshape(-1),
        x0=guess.reshape(-1),
        rtol=tolerance,
        atol=0.0,
        maxiter=20 * count,
    )
    if info != 0:
        raise RuntimeError(f"color Gauss iteration did not converge: {info}")
    potential = solution.reshape(shape)
    potential -= np.mean(potential)
    electric = _electric_from_potential(
        potential,
        link_epsilon,
        parameters,
    )
    residual = color_gauss_divergence(
        electric,
        parameters,
    ) - charge_values
    scale = max(float(np.max(np.abs(charge_values))), 1.0)
    residual_norm = float(np.max(np.abs(residual)) / scale)
    return potential, electric, residual_norm


def r4_color_static_energy(
    chi: np.ndarray,
    electric: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
) -> tuple[float, dict[str, float], np.ndarray]:
    """Return constrained R4 Cartan energy and its site density."""

    chi_values = np.asarray(chi, dtype=np.float64)
    _, _, link_epsilon = _link_dielectric(
        chi_values,
        parameters,
    )
    if electric.shape != link_epsilon.shape:
        raise ValueError("electric must match the R4 link graph")
    link_density = electric**2 / (2.0 * link_epsilon)
    electric_energy = float(np.sum(link_density))
    radial_density = (
        parameters.r3.frame_inertia
        * parameters.r3.lambda_h
        * (chi_values**2 - parameters.r3.chi0**2) ** 2
    )
    radial_energy = float(np.sum(radial_density))
    laplacian = _laplacian(chi_values, parameters.stencil)
    gradient_density = (
        -0.5
        * parameters.r3.frame_stiffness
        * chi_values
        * laplacian
    )
    gradient_energy = float(np.sum(gradient_density))
    site_electric = r4_color_electric_site_density(
        chi_values,
        electric,
        parameters,
    )
    site_density = site_electric + radial_density + gradient_density
    parts = {
        "color_electric": electric_energy,
        "chi_radial": radial_energy,
        "chi_gradient": gradient_energy,
    }
    return float(sum(parts.values())), parts, site_density


def r4_color_electric_site_density(
    chi: np.ndarray,
    electric: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
) -> np.ndarray:
    """Assign half of each R4 Cartan link energy to either endpoint."""

    chi_values = np.asarray(chi, dtype=np.float64)
    _, _, link_epsilon = _link_dielectric(
        chi_values,
        parameters,
    )
    if electric.shape != link_epsilon.shape:
        raise ValueError("electric must match the R4 link graph")
    link_density = electric**2 / (2.0 * link_epsilon)
    site_electric = 0.5 * np.sum(link_density, axis=-1)
    unique, _ = _link_table(parameters.stencil)
    for index, (offset, _) in enumerate(unique):
        site_electric += 0.5 * np.roll(
            link_density[..., index],
            shift=offset,
            axis=(0, 1, 2),
        )
    return site_electric


def r4_color_flux_observables(
    state: R4ColorStaticState,
    parameters: R4Parameters = R4Parameters(),
) -> dict[str, float]:
    """Return path-independent point-pair flux/dielectric observables."""

    nonzero = np.argwhere(np.abs(state.charge) > 0.0)
    if nonzero.shape != (2, 3):
        raise ValueError("flux observables require exactly two point charges")
    difference = nonzero[1] - nonzero[0]
    axes = np.flatnonzero(difference != 0)
    if axes.size != 1:
        raise ValueError("point charges must differ along one lattice axis")
    longitudinal_axis = int(axes[0])
    transverse_axes = [
        axis for axis in range(3) if axis != longitudinal_axis
    ]
    density = r4_color_electric_site_density(
        state.chi,
        state.electric,
        parameters,
    )
    coordinates = np.indices(state.chi.shape, dtype=np.float64)
    center = 0.5 * (nonzero[0] + nonzero[1])
    transverse_sq = np.zeros_like(state.chi)
    for axis in transverse_axes:
        displacement = np.abs(coordinates[axis] - center[axis])
        displacement = np.minimum(
            displacement,
            state.chi.shape[axis] - displacement,
        )
        transverse_sq += displacement**2
    total = max(float(np.sum(density)), 1.0e-30)
    transverse_rms = float(
        np.sqrt(np.sum(density * transverse_sq) / total)
    )
    epsilon, _ = color_dielectric(state.chi, parameters)
    density_flat = density.reshape(-1)
    epsilon_flat = epsilon.reshape(-1)
    if np.std(density_flat) == 0.0 or np.std(epsilon_flat) == 0.0:
        correlation = 0.0
    else:
        correlation = float(
            np.corrcoef(density_flat, epsilon_flat)[0, 1]
        )
    return {
        "transverse_flux_rms": transverse_rms,
        "flux_dielectric_correlation": correlation,
        "epsilon_min": float(np.min(epsilon)),
        "epsilon_max": float(np.max(epsilon)),
        "chi_min": float(np.min(state.chi)),
        "chi_max": float(np.max(state.chi)),
    }


def _chi_energy_gradient(
    chi: np.ndarray,
    electric: np.ndarray,
    parameters: R4Parameters,
) -> np.ndarray:
    epsilon, epsilon_derivative, link_epsilon = _link_dielectric(
        chi,
        parameters,
    )
    del epsilon
    gradient = (
        4.0
        * parameters.r3.frame_inertia
        * parameters.r3.lambda_h
        * chi
        * (chi**2 - parameters.r3.chi0**2)
        - parameters.r3.frame_stiffness
        * _laplacian(chi, parameters.stencil)
    )
    unique, _ = _link_table(parameters.stencil)
    for index, (offset, _) in enumerate(unique):
        endpoint = (
            -0.25
            * electric[..., index] ** 2
            * epsilon_derivative
            / link_epsilon[..., index] ** 2
        )
        gradient += endpoint
        neighbor_endpoint = (
            -0.25
            * electric[..., index] ** 2
            * np.roll(
                epsilon_derivative,
                shift=tuple(-value for value in offset),
                axis=(0, 1, 2),
            )
            / link_epsilon[..., index] ** 2
        )
        gradient += np.roll(
            neighbor_endpoint,
            shift=offset,
            axis=(0, 1, 2),
        )
    return gradient


def relax_r4_color_static(
    charge: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
    *,
    seed: int,
    initial_chi_noise: float,
    chi_iterations: int,
    chi_step: float,
    gauss_tolerance: float,
    gauss_block: int = 10,
) -> R4ColorStaticState:
    """Relax chi from an unbiased seed while enforcing color Gauss law."""

    charge_values = np.asarray(charge, dtype=np.float64)
    if charge_values.ndim != 3:
        raise ValueError("charge must have a 3D shape")
    if chi_iterations < 1 or gauss_block < 1:
        raise ValueError("iteration counts must be positive")
    if initial_chi_noise < 0.0 or chi_step <= 0.0:
        raise ValueError("noise must be nonnegative and step positive")
    rng = np.random.default_rng(seed)
    chi = (
        parameters.r3.chi0
        + initial_chi_noise * rng.normal(size=charge_values.shape)
    )
    potential = np.zeros_like(charge_values)
    electric = np.zeros(
        charge_values.shape
        + (len(_link_table(parameters.stencil)[0]),),
        dtype=np.float64,
    )
    residual = float("inf")
    for iteration in range(chi_iterations):
        if iteration % gauss_block == 0:
            potential, electric, residual = solve_color_gauss_minimum(
                chi,
                charge_values,
                parameters,
                tolerance=gauss_tolerance,
                initial_potential=potential,
            )
        chi -= chi_step * _chi_energy_gradient(
            chi,
            electric,
            parameters,
        )
        if not np.all(np.isfinite(chi)):
            raise FloatingPointError("chi relaxation became non-finite")
    potential, electric, residual = solve_color_gauss_minimum(
        chi,
        charge_values,
        parameters,
        tolerance=gauss_tolerance,
        initial_potential=potential,
    )
    energy, parts, _ = r4_color_static_energy(
        chi,
        electric,
        parameters,
    )
    return R4ColorStaticState(
        chi=chi,
        potential=potential,
        electric=electric,
        charge=charge_values.copy(),
        gauss_residual=residual,
        energy=energy,
        energy_parts=parts,
        iterations=chi_iterations,
    )


def relax_r4_chi_at_fixed_color_electric(
    electric: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
    *,
    seed: int,
    initial_chi_noise: float,
    chi_iterations: int,
    chi_step: float,
) -> R4FixedColorElectricState:
    """Relax the existing R4 chi energy around a fixed color flux probe."""

    electric_values = np.asarray(electric, dtype=np.float64)
    expected_links = len(_link_table(parameters.stencil)[0])
    if electric_values.ndim != 4 or electric_values.shape[-1] != expected_links:
        raise ValueError("electric must match one 3D R4 link graph")
    if chi_iterations < 1 or chi_step <= 0.0:
        raise ValueError("iterations and chi_step must be positive")
    if initial_chi_noise < 0.0:
        raise ValueError("initial_chi_noise must be nonnegative")
    rng = np.random.default_rng(seed)
    chi = (
        parameters.r3.chi0
        + initial_chi_noise * rng.normal(size=electric_values.shape[:-1])
    )
    for _ in range(chi_iterations):
        chi -= chi_step * _chi_energy_gradient(
            chi,
            electric_values,
            parameters,
        )
        if not np.all(np.isfinite(chi)):
            raise FloatingPointError("fixed-flux chi relaxation became non-finite")
    energy, parts, _ = r4_color_static_energy(
        chi,
        electric_values,
        parameters,
    )
    _, _, link_epsilon = _link_dielectric(chi, parameters)
    electric_norm = float(np.sum(electric_values**2))
    effective_g_squared = float(
        np.sum(electric_values**2 / link_epsilon)
        / max(electric_norm, 1.0e-30)
    )
    gauss = color_gauss_divergence(electric_values, parameters)
    return R4FixedColorElectricState(
        chi=chi,
        electric=electric_values.copy(),
        gauss_residual=float(np.max(np.abs(gauss))),
        energy=energy,
        energy_parts=parts,
        effective_g_squared=effective_g_squared,
        iterations=chi_iterations,
    )
