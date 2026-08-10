"""Experimental R5 action with geometry-complete compact-link curvature.

R5 keeps the complete R4 register and adds no new field. It retains the
R4 triangle holonomies, which constrain diagonal and corner links, and
adds the three axial face-square holonomies required for independent
spatial curvature. Their coefficient is the directional link inertia
count of the selected stencil: 5 for the 19 graph and 9 for the 27 graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from lfm.foundations.r3_link_frame_live import (
    _accumulate_oriented_gradient,
    _accumulate_oriented_phase_gradient,
    _dagger,
    _frame_loop_energy_gradient,
    _link_and_shape_drift,
    _neighbor,
    _oriented_link,
    _scatter_gradient_to_base,
    _temporal_shape_projector,
    _tracefree_symmetric,
    _transpose,
    _weighted_bare_kinetic_drift,
    so4_generators,
    su3_generators,
)
from lfm.foundations.r4_gauge_spectrum import (
    directional_link_inertia,
    face_square_cycles,
)
from lfm.foundations.r4_unified_live import (
    R4Parameters,
    R4Rates,
    R4State,
    _extra_gauge_kinetic_drift,
    _validate_r4,
    _weak_matter_kinetic_drift,
    color_dielectric,
    group_constraint_errors,  # noqa: F401 - public R5 re-export
    reverse_momenta,  # noqa: F401 - public R5 re-export
    state_distance,  # noqa: F401 - public R5 re-export
    su2_generators,
)
from lfm.foundations.r4_unified_live import (
    kinetic_energy as r4_kinetic_energy,
)
from lfm.foundations.r4_unified_live import (
    potential_energy_and_rates as r4_potential_energy_and_rates,
)

R5_ACTION_ID = "LFM-R5-GEOMETRY-COMPLETE-CURVATURE-EXPERIMENT-v1"
R5_REGISTER_ID = "R5=R4(no_new_registers)"
R5State = R4State
R5Rates = R4Rates


@dataclass(frozen=True)
class R5Parameters:
    """R5 parameters, all inherited or counted from the R4 link graph."""

    r4: R4Parameters = field(default_factory=R4Parameters)

    @property
    def stencil(self) -> str:
        return self.r4.stencil

    @property
    def square_coefficient(self) -> float:
        return directional_link_inertia(self.stencil)


def vacuum_state(
    size: int,
    parameters: R5Parameters = R5Parameters(),
) -> R5State:
    """Return the exact R5 vacuum on the selected graph."""

    return R4State.vacuum(size, parameters.r4)


def _cycle_products(
    links: np.ndarray,
    cycle: tuple[tuple[int, int, int], ...],
    *,
    complex_group: bool,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[tuple[int, int, int], ...],
    np.ndarray,
]:
    factors = []
    base_shifts = []
    shift = (0, 0, 0)
    for offset in cycle:
        oriented = _oriented_link(
            links,
            offset,
            complex_group=complex_group,
        )
        factor = (
            oriented
            if shift == (0, 0, 0)
            else _neighbor(oriented, shift)
        )
        factors.append(factor)
        base_shifts.append(shift)
        shift = tuple(
            shift[axis] + offset[axis] for axis in range(3)
        )
    if shift != (0, 0, 0):
        raise ValueError("face-square cycle must close")
    holonomy = factors[0]
    for factor in factors[1:]:
        holonomy = (
            holonomy @ factor
            if holonomy.ndim >= 5
            else holonomy * factor
        )
    return tuple(factors), tuple(base_shifts), holonomy


def _matrix_cycle_gradients(
    factors: tuple[np.ndarray, ...],
    holonomy_gradient: np.ndarray,
    *,
    complex_group: bool,
) -> tuple[np.ndarray, ...]:
    gradients = []
    for selected in range(len(factors)):
        before = None
        for factor in factors[:selected]:
            before = factor if before is None else before @ factor
        after = None
        for factor in factors[selected + 1 :]:
            after = factor if after is None else after @ factor
        gradient = holonomy_gradient
        if before is not None:
            left = _dagger(before) if complex_group else _transpose(before)
            gradient = left @ gradient
        if after is not None:
            right = _dagger(after) if complex_group else _transpose(after)
            gradient = gradient @ right
        gradients.append(gradient)
    return tuple(gradients)


def _phase_cycle_gradients(
    factors: tuple[np.ndarray, ...],
    holonomy_gradient: np.ndarray,
) -> tuple[np.ndarray, ...]:
    gradients = []
    for selected in range(len(factors)):
        before = np.ones_like(holonomy_gradient)
        for factor in factors[:selected]:
            before *= factor
        after = np.ones_like(holonomy_gradient)
        for factor in factors[selected + 1 :]:
            after *= factor
        gradients.append(
            np.conj(before) * holonomy_gradient * np.conj(after)
        )
    return tuple(gradients)


def _active_square_sectors(
    state: R5State,
    parameters: R5Parameters,
) -> tuple[str, ...]:
    active = []
    if np.any(state.r3.phase_links != 1.0 + 0.0j):
        active.append("phase")
    identities = [
        ("color", state.r3.color_links, np.eye(3)),
        ("weak", state.weak_links, np.eye(2)),
    ]
    if parameters.r4.r3.frame_enabled:
        identities.append(("frame", state.r3.frame_links, np.eye(4)))
    for name, links, identity in identities:
        if np.any(links != identity):
            active.append(name)
    return tuple(active)


def _frame_weight(state: R5State, parameters: R5Parameters) -> np.ndarray:
    if parameters.r4.r3.frame_enabled:
        return np.exp(state.r3.shape[..., 0, 0])
    return np.ones_like(state.r3.chi)


def _add_single_face_square_curvature(
    state: R5State,
    parameters: R5Parameters,
    rates: R5Rates,
    components: dict[str, float],
    sector: str,
) -> float:
    """Fast exact square correction when only one compact group is active."""

    base = state.r3
    r4 = parameters.r4
    coefficient = parameters.square_coefficient
    q = _frame_weight(state, parameters)
    frame_enabled = r4.r3.frame_enabled
    names = {
        "phase": "phase_face_square",
        "color": "color_face_square",
        "frame": "frame_face_square",
        "weak": "weak_face_square",
    }
    for name in names.values():
        components[name] = 0.0
    if sector == "phase":
        gradient_total = np.zeros_like(base.phase_links)
        for cycle in face_square_cycles():
            factors, shifts, holonomy = _cycle_products(
                base.phase_links,
                cycle,
                complex_group=True,
            )
            bare = (
                r4.r3.phase_stiffness
                * coefficient
                * (1.0 - np.real(holonomy))
            )
            components[names[sector]] += float(np.sum(q * bare))
            if frame_enabled:
                rates.r3.shape -= (
                    q * bare
                )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()
            holonomy_gradient = (
                -q * r4.r3.phase_stiffness * coefficient
            ).astype(np.complex128)
            for offset, shift, gradient in zip(
                cycle,
                shifts,
                _phase_cycle_gradients(factors, holonomy_gradient),
                strict=True,
            ):
                _accumulate_oriented_phase_gradient(
                    gradient_total,
                    offset,
                    _scatter_gradient_to_base(gradient, shift),
                    stencil=parameters.stencil,
                )
        for index in range(base.phase_links.shape[3]):
            link = base.phase_links[..., index]
            rates.r3.phase_electric[..., index] -= np.real(
                np.conj(gradient_total[..., index]) * (1.0j * link)
            )
        if frame_enabled:
            rates.r3.shape = _tracefree_symmetric(rates.r3.shape)
        return float(components[names[sector]])

    if sector == "color":
        links = base.color_links
        gradient_total = np.zeros_like(links)
        epsilon, epsilon_derivative = color_dielectric(base.chi, r4)
        identity = np.eye(3, dtype=np.complex128)
        stiffness = r4.r3.color_stiffness
        generators = su3_generators()
        complex_group = True
    elif sector == "weak":
        links = state.weak_links
        gradient_total = np.zeros_like(links)
        epsilon = None
        epsilon_derivative = None
        identity = np.eye(2, dtype=np.complex128)
        stiffness = r4.weak_stiffness
        generators = su2_generators()
        complex_group = True
    elif sector == "frame":
        if not frame_enabled:
            raise ValueError("frame square sector is disabled")
        links = base.frame_links
        gradient_total = np.zeros_like(links)
        epsilon = None
        epsilon_derivative = None
        identity = np.eye(4, dtype=np.float64)
        stiffness = r4.r3.frame_stiffness
        generators = so4_generators()
        complex_group = False
    else:
        raise ValueError("unknown compact square sector")

    for cycle in face_square_cycles():
        factors, shifts, holonomy = _cycle_products(
            links,
            cycle,
            complex_group=complex_group,
        )
        if sector == "frame":
            energy_density, holonomy_gradient = (
                _frame_loop_energy_gradient(
                    holonomy,
                    stiffness * coefficient,
                    r4.r3.epsilon_w,
                )
            )
            components[names[sector]] += float(np.sum(energy_density))
        else:
            dimension = identity.shape[0]
            bare = (
                stiffness
                * coefficient
                * (
                    float(dimension)
                    - np.real(np.trace(holonomy, axis1=-2, axis2=-1))
                )
            )
            factor = q if sector == "weak" else q * epsilon
            components[names[sector]] += float(np.sum(factor * bare))
            if frame_enabled:
                rates.r3.shape -= (
                    factor * bare
                )[..., np.newaxis, np.newaxis] * (
                    _temporal_shape_projector()
                )
            if sector == "color":
                rates.r3.chi -= q * epsilon_derivative * bare
            holonomy_gradient = (
                -factor * stiffness * coefficient
            )[..., np.newaxis, np.newaxis] * identity
        for offset, shift, gradient in zip(
            cycle,
            shifts,
            _matrix_cycle_gradients(
                factors,
                holonomy_gradient,
                complex_group=complex_group,
            ),
            strict=True,
        ):
            _accumulate_oriented_gradient(
                gradient_total,
                offset,
                _scatter_gradient_to_base(gradient, shift),
                stencil=parameters.stencil,
                complex_group=complex_group,
            )
    for index in range(links.shape[3]):
        link = links[..., index, :, :]
        for generator_index, generator in enumerate(generators):
            variation = (
                1.0j * generator @ link
                if complex_group
                else generator @ link
            )
            derivative = np.sum(
                (
                    np.conj(gradient_total[..., index, :, :])
                    if complex_group
                    else gradient_total[..., index, :, :]
                )
                * variation,
                axis=(-2, -1),
            )
            derivative = np.real(derivative)
            if sector == "color":
                rates.r3.color_electric[
                    ..., index, generator_index
                ] -= derivative
            elif sector == "weak":
                rates.weak_electric[..., index, generator_index] -= derivative
            else:
                rates.r3.frame_electric[
                    ..., index, generator_index
                ] -= derivative
    if frame_enabled:
        rates.r3.shape = _tracefree_symmetric(rates.r3.shape)
    return float(components[names[sector]])


def _add_face_square_curvature(
    state: R5State,
    parameters: R5Parameters,
    rates: R5Rates,
    components: dict[str, float],
) -> float:
    base = state.r3
    r4 = parameters.r4
    coefficient = parameters.square_coefficient
    q = _frame_weight(state, parameters)
    frame_enabled = r4.r3.frame_enabled
    epsilon, epsilon_derivative = color_dielectric(base.chi, r4)
    phase_gradient = np.zeros_like(base.phase_links)
    color_gradient = np.zeros_like(base.color_links)
    frame_gradient = np.zeros_like(base.frame_links)
    weak_gradient = np.zeros_like(state.weak_links)
    identity3 = np.eye(3, dtype=np.complex128)
    identity2 = np.eye(2, dtype=np.complex128)
    square_energy = {
        "phase_face_square": 0.0,
        "color_face_square": 0.0,
        "frame_face_square": 0.0,
        "weak_face_square": 0.0,
    }
    for cycle in face_square_cycles():
        phase_factors, base_shifts, phase_holonomy = _cycle_products(
            base.phase_links,
            cycle,
            complex_group=True,
        )
        phase_bare = (
            r4.r3.phase_stiffness
            * coefficient
            * (1.0 - np.real(phase_holonomy))
        )
        square_energy["phase_face_square"] += float(
            np.sum(q * phase_bare)
        )
        if frame_enabled:
            rates.r3.shape -= (
                q * phase_bare
            )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()
        phase_h_gradient = (
            -q * r4.r3.phase_stiffness * coefficient
        ).astype(np.complex128)
        for offset, base_shift, gradient in zip(
            cycle,
            base_shifts,
            _phase_cycle_gradients(
                phase_factors,
                phase_h_gradient,
            ),
            strict=True,
        ):
            _accumulate_oriented_phase_gradient(
                phase_gradient,
                offset,
                _scatter_gradient_to_base(gradient, base_shift),
                stencil=parameters.stencil,
            )

        color_factors, _, color_holonomy = _cycle_products(
            base.color_links,
            cycle,
            complex_group=True,
        )
        color_bare = (
            r4.r3.color_stiffness
            * coefficient
            * (
                3.0
                - np.real(
                    np.trace(color_holonomy, axis1=-2, axis2=-1)
                )
            )
        )
        square_energy["color_face_square"] += float(
            np.sum(q * epsilon * color_bare)
        )
        if frame_enabled:
            rates.r3.shape -= (
                q * epsilon * color_bare
            )[..., np.newaxis, np.newaxis] * (
                _temporal_shape_projector()
            )
        rates.r3.chi -= q * epsilon_derivative * color_bare
        color_h_gradient = (
            -q * epsilon * r4.r3.color_stiffness * coefficient
        )[..., np.newaxis, np.newaxis] * identity3
        for offset, base_shift, gradient in zip(
            cycle,
            base_shifts,
            _matrix_cycle_gradients(
                color_factors,
                color_h_gradient,
                complex_group=True,
            ),
            strict=True,
        ):
            _accumulate_oriented_gradient(
                color_gradient,
                offset,
                _scatter_gradient_to_base(gradient, base_shift),
                stencil=parameters.stencil,
                complex_group=True,
            )

        if frame_enabled:
            frame_factors, _, frame_holonomy = _cycle_products(
                base.frame_links,
                cycle,
                complex_group=False,
            )
            frame_energy, frame_h_gradient = _frame_loop_energy_gradient(
                frame_holonomy,
                r4.r3.frame_stiffness * coefficient,
                r4.r3.epsilon_w,
            )
            square_energy["frame_face_square"] += float(
                np.sum(frame_energy)
            )
            for offset, base_shift, gradient in zip(
                cycle,
                base_shifts,
                _matrix_cycle_gradients(
                    frame_factors,
                    frame_h_gradient,
                    complex_group=False,
                ),
                strict=True,
            ):
                _accumulate_oriented_gradient(
                    frame_gradient,
                    offset,
                    _scatter_gradient_to_base(gradient, base_shift),
                    stencil=parameters.stencil,
                    complex_group=False,
                )

        weak_factors, _, weak_holonomy = _cycle_products(
            state.weak_links,
            cycle,
            complex_group=True,
        )
        weak_bare = (
            r4.weak_stiffness
            * coefficient
            * (
                2.0
                - np.real(
                    np.trace(weak_holonomy, axis1=-2, axis2=-1)
                )
            )
        )
        square_energy["weak_face_square"] += float(
            np.sum(q * weak_bare)
        )
        if frame_enabled:
            rates.r3.shape -= (
                q * weak_bare
            )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()
        weak_h_gradient = (
            -q * r4.weak_stiffness * coefficient
        )[..., np.newaxis, np.newaxis] * identity2
        for offset, base_shift, gradient in zip(
            cycle,
            base_shifts,
            _matrix_cycle_gradients(
                weak_factors,
                weak_h_gradient,
                complex_group=True,
            ),
            strict=True,
        ):
            _accumulate_oriented_gradient(
                weak_gradient,
                offset,
                _scatter_gradient_to_base(gradient, base_shift),
                stencil=parameters.stencil,
                complex_group=True,
            )

    for index in range(base.phase_links.shape[3]):
        phase = base.phase_links[..., index]
        rates.r3.phase_electric[..., index] -= np.real(
            np.conj(phase_gradient[..., index]) * (1.0j * phase)
        )
        color = base.color_links[..., index, :, :]
        for generator_index, generator in enumerate(su3_generators()):
            variation = 1.0j * generator @ color
            rates.r3.color_electric[..., index, generator_index] -= np.real(
                np.sum(
                    np.conj(color_gradient[..., index, :, :])
                    * variation,
                    axis=(-2, -1),
                )
            )
        if frame_enabled:
            frame = base.frame_links[..., index, :, :]
            for generator_index, generator in enumerate(so4_generators()):
                variation = generator @ frame
                rates.r3.frame_electric[
                    ..., index, generator_index
                ] -= np.sum(
                    frame_gradient[..., index, :, :] * variation,
                    axis=(-2, -1),
                )
        weak = state.weak_links[..., index, :, :]
        for generator_index, generator in enumerate(su2_generators()):
            variation = 1.0j * generator @ weak
            rates.weak_electric[..., index, generator_index] -= np.real(
                np.sum(
                    np.conj(weak_gradient[..., index, :, :])
                    * variation,
                    axis=(-2, -1),
                )
            )
    if frame_enabled:
        rates.r3.shape = _tracefree_symmetric(rates.r3.shape)
    components.update(square_energy)
    return float(sum(square_energy.values()))


def potential_energy_and_rates(
    state: R5State,
    parameters: R5Parameters = R5Parameters(),
) -> tuple[float, R5Rates, dict[str, float]]:
    """Return the complete R5 potential, forces, and component ledger."""

    energy, rates, components = r4_potential_energy_and_rates(
        state,
        parameters.r4,
    )
    active_sectors = _active_square_sectors(state, parameters)
    if not active_sectors:
        for name in (
            "phase_face_square",
            "color_face_square",
            "frame_face_square",
            "weak_face_square",
        ):
            components[name] = 0.0
        correction = 0.0
    elif len(active_sectors) == 1:
        correction = _add_single_face_square_curvature(
            state,
            parameters,
            rates,
            components,
            active_sectors[0],
        )
    else:
        correction = _add_face_square_curvature(
            state,
            parameters,
            rates,
            components,
        )
    return energy + correction, rates, components


def kinetic_energy(
    state: R5State,
    parameters: R5Parameters = R5Parameters(),
) -> tuple[float, dict[str, float]]:
    return r4_kinetic_energy(state, parameters.r4)


def total_hamiltonian(
    state: R5State,
    parameters: R5Parameters = R5Parameters(),
) -> tuple[float, dict[str, float]]:
    kinetic, kinetic_parts = kinetic_energy(state, parameters)
    potential, _, potential_parts = potential_energy_and_rates(
        state,
        parameters,
    )
    return kinetic + potential, {**kinetic_parts, **potential_parts}


def _potential_kick(
    state: R5State,
    duration: float,
    parameters: R5Parameters,
) -> None:
    _, rates, _ = potential_energy_and_rates(state, parameters)
    base = state.r3
    base.matter_momentum += duration * rates.r3.matter
    base.chi_momentum += duration * rates.r3.chi
    if parameters.r4.r3.frame_enabled:
        base.shape_momentum += duration * rates.r3.shape
    base.phase_electric += duration * rates.r3.phase_electric
    base.color_electric += duration * rates.r3.color_electric
    if parameters.r4.r3.frame_enabled:
        base.frame_electric += duration * rates.r3.frame_electric
    state.weak_momentum += duration * rates.weak_matter
    state.weak_electric += duration * rates.weak_electric
    state.higgs_electric += duration * rates.higgs_electric


def step_r5(
    state: R5State,
    dt: float,
    parameters: R5Parameters = R5Parameters(),
) -> None:
    """Advance one symmetric second-order R5 Hamiltonian split."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    _validate_r4(state, parameters.r4)
    half = 0.5 * dt
    _potential_kick(state, half, parameters)
    _link_and_shape_drift(state.r3, half, parameters.r4.r3)
    _extra_gauge_kinetic_drift(state, half, parameters.r4)
    _weighted_bare_kinetic_drift(state.r3, dt, parameters.r4.r3)
    _weak_matter_kinetic_drift(state, dt, parameters.r4)
    _extra_gauge_kinetic_drift(state, half, parameters.r4)
    _link_and_shape_drift(state.r3, half, parameters.r4.r3)
    _potential_kick(state, half, parameters)


def r5_action_declaration(
    parameters: R5Parameters = R5Parameters(),
) -> dict[str, object]:
    return {
        "action_id": R5_ACTION_ID,
        "register_id": R5_REGISTER_ID,
        "canonical_status": "EXPERIMENT_ONLY_UNPROMOTED",
        "parameters": asdict(parameters),
        "derived_parameters": {
            "face_square_coefficient": parameters.square_coefficient,
            "coefficient_rule": "sum_unique_links_offset_axis_squared",
        },
        "retained_terms": ["complete_R4_action"],
        "added_terms": [
            "geometry_normalized_U1_face_square_curvature",
            "geometry_normalized_SU2_face_square_curvature",
            "geometry_normalized_SU3_face_square_curvature",
            "geometry_normalized_SO4_face_square_curvature",
        ],
        "new_registers": [],
        "forbidden_mechanisms_used": [],
        "paper_45_update_authorized": False,
    }


def r5_action_fingerprint(
    parameters: R5Parameters = R5Parameters(),
) -> str:
    encoded = json.dumps(
        r5_action_declaration(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
