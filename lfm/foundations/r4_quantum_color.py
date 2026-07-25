"""Quantum compact-link diagnostics for the experimental R4 color sector.

This module quantizes the SU(3) link and electric registers already present
in R4. It does not add a color potential, flux path, string tension, or
confinement register. The strong-coupling electric spectrum follows from
the R4 Hamiltonian coefficient and the SU(3) generators used by the live
evolution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from lfm.foundations.r3_link_frame_live import (
    _link_table,
    su3_generators,
    triangle_loops,
)
from lfm.foundations.r4_unified_live import (
    R4Parameters,
    R4State,
    color_dielectric,
)


@dataclass(frozen=True)
class R4QuantumColorCoefficients:
    """Vacuum coefficients of the compact SU(3) link Hamiltonian."""

    epsilon_vacuum: float
    electric_coefficient: float
    magnetic_coefficient: float
    g_squared_from_electric: float
    inverse_g_squared_from_magnetic: float
    fundamental_casimir: float
    fundamental_flux_slope: float


@dataclass(frozen=True)
class R4MagneticCompetitionBound:
    """Local norm bound on magnetic dressing of one flux link."""

    stencil: str
    max_weighted_loop_incidence: float
    su3_loop_range: float
    magnetic_bound_per_link: float
    electric_flux_slope: float
    residual_positive_slope: float


def local_su3_gauge_transform(
    state: R4State,
    transformations: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
) -> R4State:
    """Apply an arbitrary site-local SU(3) transformation to one R4 state."""

    gauge = np.asarray(transformations, dtype=np.complex128)
    expected = state.r3.chi.shape + (3, 3)
    if gauge.shape != expected:
        raise ValueError("transformations must provide one 3x3 matrix per site")
    identity = np.eye(3, dtype=np.complex128)
    unitary_error = float(
        np.max(
            np.abs(
                np.swapaxes(gauge.conj(), -1, -2) @ gauge - identity
            )
        )
    )
    determinant_error = float(
        np.max(np.abs(np.linalg.det(gauge) - 1.0))
    )
    if unitary_error > 1.0e-10 or determinant_error > 1.0e-10:
        raise ValueError("transformations must be site-local SU(3) matrices")
    transformed = state.copy()
    transformed.r3.matter = np.einsum(
        "...ij,...j->...i",
        gauge,
        state.r3.matter,
    )
    transformed.r3.matter_momentum = np.einsum(
        "...ij,...j->...i",
        gauge,
        state.r3.matter_momentum,
    )
    generators = su3_generators()
    unique, _ = _link_table(parameters.stencil)
    for index, (offset, _) in enumerate(unique):
        gauge_neighbor = np.roll(
            gauge,
            shift=tuple(-value for value in offset),
            axis=(0, 1, 2),
        )
        transformed.r3.color_links[..., index, :, :] = (
            gauge
            @ state.r3.color_links[..., index, :, :]
            @ np.swapaxes(gauge_neighbor.conj(), -1, -2)
        )
        electric_matrix = np.einsum(
            "...a,aij->...ij",
            state.r3.color_electric[..., index, :],
            generators,
        )
        rotated_electric = (
            gauge
            @ electric_matrix
            @ np.swapaxes(gauge.conj(), -1, -2)
        )
        transformed.r3.color_electric[..., index, :] = (
            2.0
            * np.einsum(
                "aij,...ji->...a",
                generators,
                rotated_electric,
            ).real
        )
    return transformed


def su3_fundamental_algebra_audit() -> dict[str, object]:
    """Audit generator normalization and derive the fundamental Casimir."""

    generators = su3_generators()
    gram = np.einsum(
        "aij,bji->ab",
        generators,
        generators,
    ).real
    casimir_matrix = np.einsum(
        "aij,ajk->ik",
        generators,
        generators,
    )
    eigenvalues = np.linalg.eigvalsh(casimir_matrix).real
    normalization_error = float(
        np.max(np.abs(gram - 0.5 * np.eye(generators.shape[0])))
    )
    casimir_spread = float(np.max(eigenvalues) - np.min(eigenvalues))
    return {
        "generator_count": int(generators.shape[0]),
        "normalization_error": normalization_error,
        "casimir_eigenvalues": [float(value) for value in eigenvalues],
        "casimir": float(np.mean(eigenvalues)),
        "casimir_spread": casimir_spread,
    }


def r4_quantum_color_coefficients(
    parameters: R4Parameters = R4Parameters(),
) -> R4QuantumColorCoefficients:
    """Read the quantum-link coefficients directly from the R4 vacuum."""

    epsilon, _ = color_dielectric(
        np.asarray(parameters.r3.chi0),
        parameters,
    )
    epsilon_vacuum = float(epsilon)
    electric_coefficient = 1.0 / (
        2.0 * parameters.r3.color_inertia * epsilon_vacuum
    )
    magnetic_coefficient = (
        parameters.r3.color_stiffness * epsilon_vacuum
    )
    g_squared = 2.0 * electric_coefficient
    inverse_g_squared = magnetic_coefficient
    casimir = float(su3_fundamental_algebra_audit()["casimir"])
    return R4QuantumColorCoefficients(
        epsilon_vacuum=epsilon_vacuum,
        electric_coefficient=electric_coefficient,
        magnetic_coefficient=magnetic_coefficient,
        g_squared_from_electric=g_squared,
        inverse_g_squared_from_magnetic=inverse_g_squared,
        fundamental_casimir=casimir,
        fundamental_flux_slope=electric_coefficient * casimir,
    )


def _reverse(offset: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(-value for value in offset)


def weighted_loop_incidence(
    stencil: str,
) -> dict[tuple[int, int, int], float]:
    """Return weighted R4 triangle-loop incidence for each link class."""

    unique, _ = _link_table(stencil)
    result: dict[tuple[int, int, int], float] = {}
    for offset, _ in unique:
        reverse = _reverse(offset)
        incidence = 0.0
        for first, second, third, weight in triangle_loops(stencil):
            incidence += weight * sum(
                edge == offset or edge == reverse
                for edge in (first, second, third)
            )
        result[offset] = float(incidence)
    return result


def r4_magnetic_competition_bound(
    parameters: R4Parameters = R4Parameters(),
) -> R4MagneticCompetitionBound:
    """Bound local magnetic dressing using the actual R4 loop inventory.

    For U in SU(3), Re Tr(U) is at least -3/2, so the range of the positive
    R4 loop operator 3-Re Tr(U) is 9/2. Multiplying that exact range by the
    weighted incidence gives the largest local magnetic energy change
    supported on one flux link.
    """

    coefficients = r4_quantum_color_coefficients(parameters)
    incidence = weighted_loop_incidence(parameters.stencil)
    max_incidence = max(incidence.values())
    su3_loop_range = 4.5
    magnetic_bound = (
        coefficients.magnetic_coefficient
        * su3_loop_range
        * max_incidence
    )
    return R4MagneticCompetitionBound(
        stencil=parameters.stencil,
        max_weighted_loop_incidence=max_incidence,
        su3_loop_range=su3_loop_range,
        magnetic_bound_per_link=magnetic_bound,
        electric_flux_slope=coefficients.fundamental_flux_slope,
        residual_positive_slope=(
            coefficients.fundamental_flux_slope - magnetic_bound
        ),
    )


def minimum_link_distance(
    displacement: tuple[int, int, int],
    stencil: str,
) -> int:
    """Return the graph distance using the live R4 link inventory."""

    target = tuple(int(value) for value in displacement)
    if target == (0, 0, 0):
        return 0
    if any(abs(value) > 64 for value in target):
        raise ValueError("displacement is too large for the exact audit")
    unique, _ = _link_table(stencil)
    moves = tuple(
        offset
        for base, _ in unique
        for offset in (base, _reverse(base))
    )
    margin = max(abs(value) for value in target) + 2
    lower = tuple(min(0, value) - margin for value in target)
    upper = tuple(max(0, value) + margin for value in target)
    queue: deque[tuple[tuple[int, int, int], int]] = deque(
        [((0, 0, 0), 0)]
    )
    visited = {(0, 0, 0)}
    while queue:
        site, distance = queue.popleft()
        for move in moves:
            neighbor = tuple(site[axis] + move[axis] for axis in range(3))
            if neighbor == target:
                return distance + 1
            if neighbor in visited:
                continue
            if not all(
                lower[axis] <= neighbor[axis] <= upper[axis]
                for axis in range(3)
            ):
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    raise RuntimeError("target was not reachable on the link graph")


def fundamental_flux_energy(
    displacement: tuple[int, int, int],
    parameters: R4Parameters = R4Parameters(),
) -> float:
    """Return the leading compact-link energy required by Gauss law."""

    distance = minimum_link_distance(displacement, parameters.stencil)
    slope = r4_quantum_color_coefficients(
        parameters
    ).fundamental_flux_slope
    return float(slope * distance)


def log_wilson_transfer(
    spatial_distance: int,
    euclidean_time: float,
    parameters: R4Parameters = R4Parameters(),
) -> float:
    """Return log W(R,T) in the controlled electric strong-coupling limit."""

    if spatial_distance < 0 or euclidean_time < 0.0:
        raise ValueError("Wilson extents must be nonnegative")
    energy = fundamental_flux_energy(
        (int(spatial_distance), 0, 0),
        parameters,
    )
    return float(-energy * euclidean_time)


def creutz_ratio_from_log_transfer(
    spatial_distance: int,
    euclidean_time: float,
    time_increment: float,
    parameters: R4Parameters = R4Parameters(),
) -> float:
    """Return the Creutz area coefficient without exponential underflow."""

    if spatial_distance < 1 or euclidean_time <= 0.0:
        raise ValueError("positive Wilson extents are required")
    if time_increment <= 0.0:
        raise ValueError("time_increment must be positive")
    r = int(spatial_distance)
    t = float(euclidean_time)
    dt = float(time_increment)
    log_ratio = (
        log_wilson_transfer(r + 1, t + dt, parameters)
        + log_wilson_transfer(r, t, parameters)
        - log_wilson_transfer(r + 1, t, parameters)
        - log_wilson_transfer(r, t + dt, parameters)
    )
    return float(-log_ratio / dt)
