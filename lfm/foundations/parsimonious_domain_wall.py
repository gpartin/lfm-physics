"""Local domain-wall weak completion of the experimental P4F action."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from lfm.foundations.parsimonious_four_force import (
    P4FParameters,
    inactive_frame_error,
    p4f_action_declaration,
)
from lfm.foundations.parsimonious_four_force import (
    total_hamiltonian as p4f_total_hamiltonian,
)
from lfm.foundations.parsimonious_four_force import (
    vacuum_state as p4f_vacuum_state,
)
from lfm.foundations.r3_link_frame_live import (
    _dagger,
    _link_and_shape_drift,
    _link_table,
    _neighbor,
    _oriented_link,
    _weighted_bare_kinetic_drift,
)
from lfm.foundations.r4_unified_live import (
    _extra_gauge_kinetic_drift,
    _weak_matter_kinetic_drift,
    su2_generators,
)
from lfm.foundations.r4_unified_live import (
    reverse_momenta as reverse_r4_momenta,
)
from lfm.foundations.r4_unified_live import (
    state_distance as r4_state_distance,
)
from lfm.foundations.r5_unified_live import _potential_kick

if TYPE_CHECKING:
    from lfm.foundations.r6_unified_live import R6State

P4F_DW_ACTION_ID = "LFM-P4F-DOMAIN-WALL-WEAK-EXPERIMENT-v1"
P4F_DW_REGISTER_ID = "P4F-DW=(P4F,DomainWallPhi_s,DomainWallPi_s)"


def _euclidean_spin_matrices() -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1.0j], [1.0j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    identity2 = np.eye(2, dtype=np.complex128)
    zero2 = np.zeros((2, 2), dtype=np.complex128)
    spatial = []
    for sigma in (sigma_x, sigma_y, sigma_z):
        spatial.append(
            np.block(
                [
                    [zero2, 1.0j * sigma],
                    [-1.0j * sigma, zero2],
                ]
            )
        )
    gamma5 = np.block([[identity2, zero2], [zero2, -identity2]])
    identity4 = np.eye(4, dtype=np.complex128)
    return (
        tuple(spatial),
        0.5 * (identity4 - gamma5),
        0.5 * (identity4 + gamma5),
    )


_SPATIAL_GAMMAS, _P_MINUS, _P_PLUS = _euclidean_spin_matrices()


@dataclass(frozen=True)
class P4FDWParameters:
    """Parameters for the local internal-chain completion."""

    p4f: P4FParameters = field(default_factory=P4FParameters)
    internal_depth: int = 19
    color_multiplicity: int = 3
    guard_coefficient: float = 1.0
    overlap_rho: float = 1.0
    single_wall_chiral: bool = True

    def __post_init__(self) -> None:
        if self.internal_depth < 2:
            raise ValueError("internal_depth must be at least two")
        if self.color_multiplicity != 3:
            raise ValueError("P4F-DW requires three color copies")
        if self.guard_coefficient != 1.0:
            raise ValueError("the action-derived guard coefficient is one")
        if self.overlap_rho != 1.0:
            raise ValueError("the audited overlap rho is one")
        if not self.single_wall_chiral:
            raise ValueError("P4F-DW requires the one-wall chiral register")
        if self.p4f.r6.r4.r3.stencil != "19":
            raise ValueError("P4F-DW v1 requires stencil19")


@dataclass
class P4FDWState:
    """P4F phase space plus a local domain-wall matter chain."""

    base: R6State
    domain_wall_field: np.ndarray
    domain_wall_momentum: np.ndarray

    def copy(self) -> P4FDWState:
        return P4FDWState(
            base=self.base.copy(),
            domain_wall_field=self.domain_wall_field.copy(),
            domain_wall_momentum=self.domain_wall_momentum.copy(),
        )


def vacuum_state(
    size: int,
    parameters: P4FDWParameters = P4FDWParameters(),
) -> P4FDWState:
    base = p4f_vacuum_state(size, parameters.p4f)
    shape = (
        size,
        size,
        size,
        parameters.internal_depth,
        4,
        2,
        parameters.color_multiplicity,
    )
    return P4FDWState(
        base=base,
        domain_wall_field=np.zeros(shape, dtype=np.complex128),
        domain_wall_momentum=np.zeros(shape, dtype=np.complex128),
    )


def _validate(
    state: P4FDWState,
    parameters: P4FDWParameters,
) -> None:
    sites = state.base.r3.chi.shape
    expected = sites + (
        parameters.internal_depth,
        4,
        2,
        parameters.color_multiplicity,
    )
    for name in ("domain_wall_field", "domain_wall_momentum"):
        values = np.asarray(getattr(state, name))
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains a non-finite value")
        removed_mirror = _apply_spin(
            _P_PLUS,
            values[..., -1:, :, :, :],
        )
        if float(np.max(np.abs(removed_mirror))) > 1.0e-12:
            raise ValueError(f"{name} contains the excluded right-wall mirror mode")
    if inactive_frame_error(state.base) != 0.0:
        raise ValueError("P4F-DW compatibility frame must remain vacuum")


def _apply_weak(
    link: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "...ij,...sajc->...saic",
        link,
        values,
        optimize=True,
    )


def _apply_spin(
    matrix: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "ab,...sbic->...saic",
        matrix,
        values,
        optimize=True,
    )


def project_domain_wall_register(values: np.ndarray) -> np.ndarray:
    """Project onto the one-wall chiral domain-wall phase space."""

    projected = np.asarray(values).copy()
    projected[..., -1:, :, :, :] = _apply_spin(
        _P_MINUS,
        projected[..., -1:, :, :, :],
    )
    return projected


def _spatial_parts(
    values: np.ndarray,
    weak_links: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique, _ = _link_table("19")
    guard = np.zeros_like(values)
    centered = np.zeros_like(values)
    for index, (offset, weight) in enumerate(unique):
        forward_link = weak_links[..., index, :, :]
        backward_link = _oriented_link(
            weak_links,
            tuple(-item for item in offset),
            complex_group=True,
        )
        forward = _apply_weak(
            forward_link,
            _neighbor(values, offset),
        )
        backward = _apply_weak(
            backward_link,
            _neighbor(values, tuple(-item for item in offset)),
        )
        guard += weight * (2.0 * values - forward - backward)
        if index < 3:
            centered += 0.5 * _apply_spin(
                _SPATIAL_GAMMAS[index],
                forward - backward,
            )
    return centered, guard


def apply_domain_wall(
    values: np.ndarray,
    weak_links: np.ndarray,
    parameters: P4FDWParameters,
) -> np.ndarray:
    """Apply the local domain-wall operator."""

    registered = project_domain_wall_register(values)
    centered, guard = _spatial_parts(registered, weak_links)
    result = (
        centered
        + parameters.guard_coefficient * guard
        + (1.0 - parameters.overlap_rho) * registered
    )
    result[..., :-1, :, :, :] -= _apply_spin(
        _P_MINUS,
        registered[..., 1:, :, :, :],
    )
    result[..., 1:, :, :, :] -= _apply_spin(
        _P_PLUS,
        registered[..., :-1, :, :, :],
    )
    return result


def apply_domain_wall_adjoint(
    values: np.ndarray,
    weak_links: np.ndarray,
    parameters: P4FDWParameters,
) -> np.ndarray:
    """Apply the exact adjoint of the local domain-wall operator."""

    centered, guard = _spatial_parts(values, weak_links)
    result = (
        -centered + parameters.guard_coefficient * guard + (1.0 - parameters.overlap_rho) * values
    )
    result[..., 1:, :, :, :] -= _apply_spin(
        _P_MINUS,
        values[..., :-1, :, :, :],
    )
    result[..., :-1, :, :, :] -= _apply_spin(
        _P_PLUS,
        values[..., 1:, :, :, :],
    )
    return project_domain_wall_register(result)


def domain_wall_potential_and_rates(
    state: P4FDWState,
    parameters: P4FDWParameters = P4FDWParameters(),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return domain-wall energy, matter rate, and weak-electric rate."""

    _validate(state, parameters)
    field = state.domain_wall_field
    if not np.any(field):
        return (
            0.0,
            np.zeros_like(field),
            np.zeros_like(state.base.weak_electric),
        )
    links = state.base.weak_links
    output = apply_domain_wall(field, links, parameters)
    energy = 0.5 * float(np.sum(np.abs(output) ** 2))
    matter_rate = -apply_domain_wall_adjoint(
        output,
        links,
        parameters,
    )
    electric_rate = np.zeros_like(state.base.weak_electric)
    unique, _ = _link_table("19")
    generators = su2_generators()
    for index, (offset, weight) in enumerate(unique):
        link = links[..., index, :, :]
        field_target = _neighbor(field, offset)
        output_target = _neighbor(output, offset)
        for generator_index, generator in enumerate(generators):
            link_variation = 1.0j * generator @ link
            dagger_variation = _dagger(link_variation)
            forward_variation = _apply_weak(
                link_variation,
                field_target,
            )
            backward_variation = _apply_weak(
                dagger_variation,
                field,
            )
            output_variation_at_base = -weight * forward_variation
            output_variation_at_target = -weight * backward_variation
            if index < 3:
                output_variation_at_base += 0.5 * _apply_spin(
                    _SPATIAL_GAMMAS[index],
                    forward_variation,
                )
                output_variation_at_target -= 0.5 * _apply_spin(
                    _SPATIAL_GAMMAS[index],
                    backward_variation,
                )
            derivative = np.real(
                np.sum(
                    np.conj(output) * output_variation_at_base,
                    axis=(-4, -3, -2, -1),
                )
                + np.sum(
                    np.conj(output_target) * output_variation_at_target,
                    axis=(-4, -3, -2, -1),
                )
            )
            electric_rate[..., index, generator_index] -= derivative
    return energy, matter_rate, electric_rate


def total_hamiltonian(
    state: P4FDWState,
    parameters: P4FDWParameters = P4FDWParameters(),
) -> tuple[float, dict[str, float]]:
    _validate(state, parameters)
    base_energy, parts = p4f_total_hamiltonian(
        state.base,
        parameters.p4f,
    )
    domain_wall_potential, _, _ = domain_wall_potential_and_rates(
        state,
        parameters,
    )
    domain_wall_kinetic = 0.5 * float(np.sum(np.abs(state.domain_wall_momentum) ** 2))
    components = dict(parts)
    components["domain_wall_potential"] = domain_wall_potential
    components["domain_wall_kinetic"] = domain_wall_kinetic
    return (
        base_energy + domain_wall_potential + domain_wall_kinetic,
        components,
    )


def _combined_kick(
    state: P4FDWState,
    duration: float,
    parameters: P4FDWParameters,
) -> None:
    _potential_kick(
        state.base,
        duration,
        parameters.p4f.r6.r5,
    )
    _, matter_rate, electric_rate = domain_wall_potential_and_rates(
        state,
        parameters,
    )
    state.domain_wall_momentum += duration * matter_rate
    state.domain_wall_momentum = project_domain_wall_register(state.domain_wall_momentum)
    state.base.weak_electric += duration * electric_rate


def step_p4f_domain_wall(
    state: P4FDWState,
    dt: float,
    parameters: P4FDWParameters = P4FDWParameters(),
) -> None:
    """Advance one symmetric local Hamiltonian split."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    _validate(state, parameters)
    base = state.base
    r4 = parameters.p4f.r6.r4
    half = 0.5 * dt
    _combined_kick(state, half, parameters)
    _link_and_shape_drift(base.r3, half, r4.r3)
    _extra_gauge_kinetic_drift(base, half, r4)
    _weighted_bare_kinetic_drift(base.r3, dt, r4.r3)
    _weak_matter_kinetic_drift(base, dt, r4)
    state.domain_wall_field += dt * state.domain_wall_momentum
    state.domain_wall_field = project_domain_wall_register(state.domain_wall_field)
    _extra_gauge_kinetic_drift(base, half, r4)
    _link_and_shape_drift(base.r3, half, r4.r3)
    _combined_kick(state, half, parameters)
    if inactive_frame_error(base) != 0.0:
        raise RuntimeError("inactive frame changed under P4F-DW evolution")


def reverse_momenta(state: P4FDWState) -> P4FDWState:
    reversed_state = state.copy()
    reverse_r4_momenta(reversed_state.base)
    reversed_state.domain_wall_momentum *= -1.0
    return reversed_state


def state_distance(
    first: P4FDWState,
    second: P4FDWState,
) -> float:
    numerator = r4_state_distance(first.base, second.base) ** 2
    denominator = 1.0
    for name in ("domain_wall_field", "domain_wall_momentum"):
        first_values = np.asarray(getattr(first, name))
        second_values = np.asarray(getattr(second, name))
        numerator += float(np.sum(np.abs(first_values - second_values) ** 2))
        denominator += float(np.sum(np.abs(first_values) ** 2))
    return float(np.sqrt(numerator / denominator))


def p4f_domain_wall_action_declaration(
    parameters: P4FDWParameters = P4FDWParameters(),
) -> dict[str, object]:
    return {
        "action_id": P4F_DW_ACTION_ID,
        "register_id": P4F_DW_REGISTER_ID,
        "canonical_status": "EXPERIMENT_ONLY_UNPROMOTED",
        "parameters": asdict(parameters),
        "retained_action": p4f_action_declaration(parameters.p4f),
        "added_registers": [
            "local_domain_wall_field",
            "local_domain_wall_conjugate_momentum",
        ],
        "added_terms": [
            "half_domain_wall_momentum_norm_squared",
            "half_domain_wall_operator_norm_squared",
            "reciprocal_SU2_link_current_from_action_variation",
            "one_wall_Pminus_chiral_phase_space_constraint",
        ],
        "locality": {
            "spatial": "stencil19_site_and_link_hops",
            "internal": "nearest_neighbor_open_chain",
            "mirror_removal": ("local_Pplus_constraint_at_terminal_internal_wall"),
            "inverse_solver": False,
            "target_force": False,
        },
        "paper_45_update_authorized": False,
    }


def p4f_domain_wall_action_fingerprint(
    parameters: P4FDWParameters = P4FDWParameters(),
) -> str:
    encoded = json.dumps(
        p4f_domain_wall_action_declaration(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
