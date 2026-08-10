"""Live Hamiltonian evolution for the experimental R3 link-frame register.

The autonomous local Hamiltonian evolves complex three-color matter, radial
chi, a traceless symmetric frame shape, compact U(1)/SU(3)/SO(4) links, and
all conjugate momenta. It contains no target-force update, inverse
Laplacian, prescribed trajectory, or canonical promotion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from lfm.analysis.energy_current import Offset, stencil_links
from lfm.constants import C_DEFAULT, CHI0, EPSILON_W, KAPPA, LAMBDA_H

R3_LIVE_ACTION_ID = "LFM-R3-LINK-FRAME-LIVE-EXPERIMENT-v2"
R3_LIVE_REGISTER_ID = (
    "R3Live=(Psi_a,Pi_a,chi,p_chi,S_AB,P_AB,"
    "UFrame,EFrame,U1,E1,U3,E3)"
)


def _neighbor(values: np.ndarray, offset: Offset) -> np.ndarray:
    return np.roll(
        values,
        shift=tuple(-value for value in offset),
        axis=(0, 1, 2),
    )


def _scatter_from_base(values: np.ndarray, offset: Offset) -> np.ndarray:
    return np.roll(values, shift=offset, axis=(0, 1, 2))


def _dagger(values: np.ndarray) -> np.ndarray:
    return np.swapaxes(values.conj(), -1, -2)


def _transpose(values: np.ndarray) -> np.ndarray:
    return np.swapaxes(values, -1, -2)


def _tracefree_symmetric(values: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (values + _transpose(values))
    trace = np.trace(symmetric, axis1=-2, axis2=-1) / 4.0
    return symmetric - trace[..., np.newaxis, np.newaxis] * np.eye(4)


def _temporal_shape_projector() -> np.ndarray:
    source = np.zeros((4, 4), dtype=np.float64)
    source[0, 0] = 1.0
    return _tracefree_symmetric(source)


@lru_cache(maxsize=1)
def su3_generators() -> np.ndarray:
    """Return Hermitian traceless generators T_a=lambda_a/2."""

    zero = np.zeros((3, 3), dtype=np.complex128)
    generators: list[np.ndarray] = []

    def add(entries: tuple[tuple[int, int, complex], ...]) -> None:
        matrix = zero.copy()
        for row, column, value in entries:
            matrix[row, column] = value
        generators.append(0.5 * matrix)

    add(((0, 1, 1.0), (1, 0, 1.0)))
    add(((0, 1, -1.0j), (1, 0, 1.0j)))
    add(((0, 0, 1.0), (1, 1, -1.0)))
    add(((0, 2, 1.0), (2, 0, 1.0)))
    add(((0, 2, -1.0j), (2, 0, 1.0j)))
    add(((1, 2, 1.0), (2, 1, 1.0)))
    add(((1, 2, -1.0j), (2, 1, 1.0j)))
    add(
        (
            (0, 0, 1.0 / np.sqrt(3.0)),
            (1, 1, 1.0 / np.sqrt(3.0)),
            (2, 2, -2.0 / np.sqrt(3.0)),
        )
    )
    return np.stack(generators)


@lru_cache(maxsize=1)
def so4_generators() -> np.ndarray:
    """Return six unit-Frobenius antisymmetric SO(4) generators."""

    generators = []
    scale = 1.0 / np.sqrt(2.0)
    for first in range(4):
        for second in range(first + 1, 4):
            matrix = np.zeros((4, 4), dtype=np.float64)
            matrix[first, second] = scale
            matrix[second, first] = -scale
            generators.append(matrix)
    return np.stack(generators)


@lru_cache(maxsize=4)
def _link_table(stencil: str) -> tuple[
    tuple[tuple[Offset, float], ...],
    dict[Offset, tuple[int, bool]],
]:
    unique = stencil_links(stencil, oriented=False)
    table: dict[Offset, tuple[int, bool]] = {}
    for index, (offset, _) in enumerate(unique):
        table[offset] = (index, False)
        reverse = tuple(-value for value in offset)
        table[reverse] = (index, True)
    return unique, table


@lru_cache(maxsize=2)
def triangle_loops(
    stencil: str,
) -> tuple[tuple[Offset, Offset, Offset, float], ...]:
    """Return one orientation of every local three-link loop type."""

    unique, table = _link_table(stencil)
    weights = {
        offset: weight for offset, weight in stencil_links(stencil)
    }
    offsets = tuple(table)
    candidates: set[tuple[Offset, Offset, Offset]] = set()
    for first in offsets:
        for second in offsets:
            third = tuple(
                -(first[axis] + second[axis]) for axis in range(3)
            )
            if third not in table:
                continue
            triple = tuple(sorted((first, second, third)))
            reverse = tuple(
                sorted(
                    tuple(-value for value in item)
                    for item in triple
                )
            )
            candidates.add(min(triple, reverse))
    loops = []
    for triple in sorted(candidates):
        first, second, third = triple
        weight = (
            weights[first] * weights[second] * weights[third]
        ) ** (1.0 / 3.0)
        loops.append((first, second, third, weight))
    if not unique or not loops:
        raise RuntimeError("stencil must contain links and local loops")
    return tuple(loops)


@dataclass(frozen=True)
class R3LiveParameters:
    """Frozen parameters of the live experimental action."""

    chi0: float = CHI0
    kappa: float = KAPPA
    lambda_h: float = LAMBDA_H
    wave_speed: float = C_DEFAULT
    epsilon_w: float = EPSILON_W
    phase_stiffness: float = 1.0
    color_stiffness: float = 1.0
    phase_inertia: float = 1.0
    color_inertia: float = 1.0
    stencil: str = "19"
    frame_enabled: bool = True
    chi_potential: str = "quartic"

    @property
    def frame_inertia(self) -> float:
        return self.chi0 / self.kappa

    @property
    def frame_stiffness(self) -> float:
        return self.frame_inertia * self.wave_speed**2

    def __post_init__(self) -> None:
        positive = (
            self.chi0,
            self.kappa,
            self.lambda_h,
            self.wave_speed,
            self.phase_stiffness,
            self.color_stiffness,
            self.phase_inertia,
            self.color_inertia,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("live R3 parameters must be positive and finite")
        if not np.isfinite(self.epsilon_w) or abs(self.epsilon_w) >= 1.0:
            raise ValueError("epsilon_w must satisfy abs(epsilon_w)<1")
        if self.chi_potential not in {"quartic", "flat_octic"}:
            raise ValueError(
                "chi_potential must be 'quartic' or 'flat_octic'"
            )
        _link_table(self.stencil)
        triangle_loops(self.stencil)


@dataclass
class R3LiveState:
    """Complete site and oriented-link phase space of the live experiment."""

    matter: np.ndarray
    matter_momentum: np.ndarray
    chi: np.ndarray
    chi_momentum: np.ndarray
    shape: np.ndarray
    shape_momentum: np.ndarray
    phase_links: np.ndarray
    phase_electric: np.ndarray
    color_links: np.ndarray
    color_electric: np.ndarray
    frame_links: np.ndarray
    frame_electric: np.ndarray

    @classmethod
    def vacuum(
        cls,
        size: int,
        parameters: R3LiveParameters = R3LiveParameters(),
    ) -> R3LiveState:
        """Return the exact periodic R3 vacuum."""

        if size < 2:
            raise ValueError("size must be at least two")
        link_count = len(_link_table(parameters.stencil)[0])
        sites = (size, size, size)
        return cls(
            matter=np.zeros(sites + (3,), dtype=np.complex128),
            matter_momentum=np.zeros(sites + (3,), dtype=np.complex128),
            chi=np.full(sites, parameters.chi0, dtype=np.float64),
            chi_momentum=np.zeros(sites, dtype=np.float64),
            shape=np.zeros(sites + (4, 4), dtype=np.float64),
            shape_momentum=np.zeros(sites + (4, 4), dtype=np.float64),
            phase_links=np.ones(sites + (link_count,), dtype=np.complex128),
            phase_electric=np.zeros(sites + (link_count,), dtype=np.float64),
            color_links=np.broadcast_to(
                np.eye(3, dtype=np.complex128),
                sites + (link_count, 3, 3),
            ).copy(),
            color_electric=np.zeros(
                sites + (link_count, 8),
                dtype=np.float64,
            ),
            frame_links=np.broadcast_to(
                np.eye(4, dtype=np.float64),
                sites + (link_count, 4, 4),
            ).copy(),
            frame_electric=np.zeros(
                sites + (link_count, 6),
                dtype=np.float64,
            ),
        )

    def copy(self) -> R3LiveState:
        return R3LiveState(
            **{
                name: np.asarray(getattr(self, name)).copy()
                for name in self.__dataclass_fields__
            }
        )


@dataclass
class R3MomentumRates:
    matter: np.ndarray
    chi: np.ndarray
    shape: np.ndarray
    phase_electric: np.ndarray
    color_electric: np.ndarray
    frame_electric: np.ndarray


def _validate_state(
    state: R3LiveState,
    parameters: R3LiveParameters,
) -> tuple[int, int, int]:
    sites = state.chi.shape
    if len(sites) != 3 or min(sites) < 2:
        raise ValueError("chi must have a three-dimensional lattice shape")
    link_count = len(_link_table(parameters.stencil)[0])
    expected = {
        "matter": sites + (3,),
        "matter_momentum": sites + (3,),
        "chi_momentum": sites,
        "shape": sites + (4, 4),
        "shape_momentum": sites + (4, 4),
        "phase_links": sites + (link_count,),
        "phase_electric": sites + (link_count,),
        "color_links": sites + (link_count, 3, 3),
        "color_electric": sites + (link_count, 8),
        "frame_links": sites + (link_count, 4, 4),
        "frame_electric": sites + (link_count, 6),
    }
    for name, shape in expected.items():
        values = np.asarray(getattr(state, name))
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains a non-finite value")
    return sites


def _stencil_for_links(links: np.ndarray) -> str:
    return "19" if links.shape[3] == 9 else "27"


def _oriented_link(
    links: np.ndarray,
    offset: Offset,
    *,
    complex_group: bool,
) -> np.ndarray:
    _, table = _link_table(_stencil_for_links(links))
    index, reverse = table[offset]
    selected = links[..., index, :, :] if links.ndim >= 6 else links[..., index]
    if not reverse:
        return selected
    shifted = _neighbor(selected, offset)
    if selected.ndim >= 5:
        return _dagger(shifted) if complex_group else _transpose(shifted)
    return shifted.conj()


def _accumulate_oriented_gradient(
    target: np.ndarray,
    offset: Offset,
    gradient: np.ndarray,
    *,
    stencil: str,
    complex_group: bool,
) -> None:
    _, table = _link_table(stencil)
    index, reverse = table[offset]
    if not reverse:
        target[..., index, :, :] += gradient
        return
    converted = _dagger(gradient) if complex_group else _transpose(gradient)
    target[..., index, :, :] += _scatter_from_base(converted, offset)


def _accumulate_oriented_phase_gradient(
    target: np.ndarray,
    offset: Offset,
    gradient: np.ndarray,
    *,
    stencil: str,
) -> None:
    _, table = _link_table(stencil)
    index, reverse = table[offset]
    if not reverse:
        target[..., index] += gradient
        return
    target[..., index] += _scatter_from_base(gradient.conj(), offset)


def _frame_loop_energy_gradient(
    holonomy: np.ndarray,
    coefficient: float,
    epsilon_w: float,
) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(4)
    symmetric = 0.5 * (holonomy + _transpose(holonomy)) - identity
    omega = 0.5 * (holonomy - _transpose(holonomy))
    temporal = np.stack(
        (omega[..., 0, 1], omega[..., 0, 2], omega[..., 0, 3]),
        axis=-1,
    )
    spatial = np.stack(
        (omega[..., 2, 3], omega[..., 3, 1], omega[..., 1, 2]),
        axis=-1,
    )
    plus = (temporal + spatial) / np.sqrt(2.0)
    minus = (temporal - spatial) / np.sqrt(2.0)
    energy = 0.5 * coefficient * (
        np.sum(symmetric**2, axis=(-2, -1))
        + (1.0 + epsilon_w) * np.sum(plus**2, axis=-1)
        + (1.0 - epsilon_w) * np.sum(minus**2, axis=-1)
    )
    gradient = coefficient * symmetric
    temporal_derivative = coefficient * (
        temporal + epsilon_w * spatial
    )
    spatial_derivative = coefficient * (
        spatial + epsilon_w * temporal
    )
    pairs = (
        (0, 1, temporal_derivative[..., 0]),
        (0, 2, temporal_derivative[..., 1]),
        (0, 3, temporal_derivative[..., 2]),
        (2, 3, spatial_derivative[..., 0]),
        (3, 1, spatial_derivative[..., 1]),
        (1, 2, spatial_derivative[..., 2]),
    )
    for first, second, value in pairs:
        gradient[..., first, second] += 0.5 * value
        gradient[..., second, first] -= 0.5 * value
    return energy, gradient


def _loop_products(
    links: np.ndarray,
    first: Offset,
    second: Offset,
    third: Offset,
    *,
    complex_group: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_link = _oriented_link(
        links,
        first,
        complex_group=complex_group,
    )
    second_link = _neighbor(
        _oriented_link(links, second, complex_group=complex_group),
        first,
    )
    first_second = tuple(
        first[axis] + second[axis] for axis in range(3)
    )
    third_link = _neighbor(
        _oriented_link(links, third, complex_group=complex_group),
        first_second,
    )
    if links.ndim >= 6:
        holonomy = first_link @ second_link @ third_link
    else:
        holonomy = first_link * second_link * third_link
    return first_link, second_link, third_link, holonomy


def _scatter_gradient_to_base(
    gradient: np.ndarray,
    base_shift: Offset,
) -> np.ndarray:
    return _scatter_from_base(gradient, base_shift)


def potential_energy_and_rates(
    state: R3LiveState,
    parameters: R3LiveParameters = R3LiveParameters(),
) -> tuple[float, R3MomentumRates, dict[str, float]]:
    """Return coordinate energy and all reciprocal momentum rates."""

    _validate_state(state, parameters)
    unique, _ = _link_table(parameters.stencil)
    sites = state.chi.shape
    matter_rate = np.zeros_like(state.matter)
    chi_rate = np.zeros_like(state.chi)
    shape_rate = np.zeros_like(state.shape)
    phase_rate = np.zeros_like(state.phase_electric)
    color_rate = np.zeros_like(state.color_electric)
    frame_rate = np.zeros_like(state.frame_electric)
    color_gradient = np.zeros_like(state.color_links)
    frame_gradient = np.zeros_like(state.frame_links)
    phase_gradient = np.zeros_like(state.phase_links)
    q = (
        np.exp(state.shape[..., 0, 0])
        if parameters.frame_enabled
        else np.ones(sites, dtype=np.float64)
    )
    potential_source_density = np.zeros(sites, dtype=np.float64)
    components = {
        "matter_gradient": 0.0,
        "chi_gradient": 0.0,
        "onsite": 0.0,
        "shape_gradient": 0.0,
        "phase_loop": 0.0,
        "color_loop": 0.0,
        "frame_loop": 0.0,
    }
    color_generators = su3_generators()
    frame_generators = so4_generators()

    for index, (offset, weight) in enumerate(unique):
        matter_j = _neighbor(state.matter, offset)
        q_j = _neighbor(q, offset)
        phase = state.phase_links[..., index]
        color = state.color_links[..., index, :, :]
        transported = phase[..., np.newaxis] * np.einsum(
            "...ab,...b->...a",
            color,
            matter_j,
        )
        difference = transported - state.matter
        norm_sq = np.sum(np.abs(difference) ** 2, axis=-1)
        source_half = (
            0.25 * parameters.wave_speed**2 * weight * norm_sq
        )
        energy_density = (q + q_j) * source_half
        components["matter_gradient"] += float(np.sum(energy_density))
        potential_source_density += source_half
        potential_source_density += _scatter_from_base(source_half, offset)
        force_scale = (
            0.5
            * parameters.wave_speed**2
            * weight
            * (q + q_j)
        )
        matter_rate += force_scale[..., np.newaxis] * difference
        neighbor_force = (
            -force_scale[..., np.newaxis]
            * np.conj(phase)[..., np.newaxis]
            * np.einsum(
                "...ab,...b->...a",
                _dagger(color),
                difference,
            )
        )
        matter_rate += _scatter_from_base(neighbor_force, offset)
        phase_derivative = force_scale * np.real(
            np.sum(
                np.conj(difference) * (1.0j * transported),
                axis=-1,
            )
        )
        phase_rate[..., index] -= phase_derivative
        for generator_index, generator in enumerate(color_generators):
            variation = 1.0j * np.einsum(
                "ab,...b->...a",
                generator,
                transported,
            )
            derivative = force_scale * np.real(
                np.sum(np.conj(difference) * variation, axis=-1)
            )
            color_rate[..., index, generator_index] -= derivative

        chi_j = _neighbor(state.chi, offset)
        chi_difference = chi_j - state.chi
        chi_source_half = (
            0.25
            * parameters.frame_inertia
            * parameters.wave_speed**2
            * weight
            * chi_difference**2
        )
        chi_energy_density = (q + q_j) * chi_source_half
        components["chi_gradient"] += float(np.sum(chi_energy_density))
        potential_source_density += chi_source_half
        potential_source_density += _scatter_from_base(
            chi_source_half,
            offset,
        )
        chi_force_scale = (
            0.5
            * parameters.frame_inertia
            * parameters.wave_speed**2
            * weight
            * (q + q_j)
        )
        chi_force = chi_force_scale * chi_difference
        chi_rate += chi_force
        chi_rate += _scatter_from_base(-chi_force, offset)

        if parameters.frame_enabled:
            shape_j = _neighbor(state.shape, offset)
            frame = state.frame_links[..., index, :, :]
            transported_shape = frame @ shape_j @ _transpose(frame)
            shape_difference = transported_shape - state.shape
            shape_norm_sq = np.sum(
                shape_difference**2,
                axis=(-2, -1),
            )
            shape_scale = parameters.frame_stiffness * weight
            components["shape_gradient"] += float(
                np.sum(0.5 * shape_scale * shape_norm_sq)
            )
            shape_rate += shape_scale * shape_difference
            neighbor_shape_force = (
                -shape_scale
                * (_transpose(frame) @ shape_difference @ frame)
            )
            shape_rate += _scatter_from_base(
                neighbor_shape_force,
                offset,
            )
            for generator_index, generator in enumerate(frame_generators):
                variation = (
                    generator @ transported_shape
                    - transported_shape @ generator
                )
                derivative = shape_scale * np.sum(
                    shape_difference * variation,
                    axis=(-2, -1),
                )
                frame_rate[..., index, generator_index] -= derivative

    matter_norm_sq = np.sum(np.abs(state.matter) ** 2, axis=-1)
    interaction = 0.5 * state.chi**2 * matter_norm_sq
    chi_delta = state.chi**2 - parameters.chi0**2
    if parameters.chi_potential == "quartic":
        radial = (
            parameters.frame_inertia
            * parameters.lambda_h
            * chi_delta**2
        )
        radial_force = (
            4.0
            * parameters.frame_inertia
            * parameters.lambda_h
            * state.chi
            * chi_delta
        )
    else:
        radial = (
            parameters.frame_inertia
            * parameters.lambda_h
            * chi_delta**4
            / parameters.chi0**4
        )
        radial_force = (
            8.0
            * parameters.frame_inertia
            * parameters.lambda_h
            * state.chi
            * chi_delta**3
            / parameters.chi0**4
        )
    onsite = interaction + radial
    components["onsite"] = float(np.sum(q * onsite))
    potential_source_density += onsite
    matter_rate -= (
        q * state.chi**2
    )[..., np.newaxis] * state.matter
    chi_rate -= q * (state.chi * matter_norm_sq + radial_force)
    if parameters.frame_enabled:
        shape_rate -= (
            q * potential_source_density
        )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()

    identity3 = np.eye(3, dtype=np.complex128)
    for first, second, third, loop_weight in triangle_loops(
        parameters.stencil
    ):
        phase_one, phase_two, phase_three, phase_holonomy = _loop_products(
            state.phase_links,
            first,
            second,
            third,
            complex_group=True,
        )
        phase_coefficient = parameters.phase_stiffness * loop_weight
        components["phase_loop"] += float(
            np.sum(
                phase_coefficient
                * (1.0 - np.real(phase_holonomy))
            )
        )
        phase_h_gradient = np.full(
            sites,
            -phase_coefficient,
            dtype=np.complex128,
        )
        phase_gradients = (
            phase_h_gradient * np.conj(phase_two * phase_three),
            np.conj(phase_one)
            * phase_h_gradient
            * np.conj(phase_three),
            np.conj(phase_one * phase_two) * phase_h_gradient,
        )
        base_shifts = (
            (0, 0, 0),
            first,
            tuple(first[axis] + second[axis] for axis in range(3)),
        )
        for offset, base_shift, gradient in zip(
            (first, second, third),
            base_shifts,
            phase_gradients,
            strict=True,
        ):
            gradient_at_base = _scatter_gradient_to_base(
                gradient,
                base_shift,
            )
            _accumulate_oriented_phase_gradient(
                phase_gradient,
                offset,
                gradient_at_base,
                stencil=parameters.stencil,
            )

        color_one, color_two, color_three, color_holonomy = _loop_products(
            state.color_links,
            first,
            second,
            third,
            complex_group=True,
        )
        color_coefficient = parameters.color_stiffness * loop_weight
        color_loop_density = color_coefficient * (
            3.0 - np.real(np.trace(color_holonomy, axis1=-2, axis2=-1))
        )
        components["color_loop"] += float(np.sum(color_loop_density))
        color_h_gradient = np.broadcast_to(
            -color_coefficient * identity3,
            color_holonomy.shape,
        )
        color_gradients = (
            color_h_gradient @ _dagger(color_two @ color_three),
            _dagger(color_one)
            @ color_h_gradient
            @ _dagger(color_three),
            _dagger(color_one @ color_two) @ color_h_gradient,
        )
        for offset, base_shift, gradient in zip(
            (first, second, third),
            base_shifts,
            color_gradients,
            strict=True,
        ):
            gradient_at_base = _scatter_gradient_to_base(
                gradient,
                base_shift,
            )
            _accumulate_oriented_gradient(
                color_gradient,
                offset,
                gradient_at_base,
                stencil=parameters.stencil,
                complex_group=True,
            )

        if parameters.frame_enabled:
            frame_one, frame_two, frame_three, frame_holonomy = (
                _loop_products(
                    state.frame_links,
                    first,
                    second,
                    third,
                    complex_group=False,
                )
            )
            frame_energy, frame_h_gradient = (
                _frame_loop_energy_gradient(
                    frame_holonomy,
                    parameters.frame_stiffness * loop_weight,
                    parameters.epsilon_w,
                )
            )
            components["frame_loop"] += float(np.sum(frame_energy))
            frame_gradients = (
                frame_h_gradient @ _transpose(frame_two @ frame_three),
                _transpose(frame_one)
                @ frame_h_gradient
                @ _transpose(frame_three),
                _transpose(frame_one @ frame_two) @ frame_h_gradient,
            )
            for offset, base_shift, gradient in zip(
                (first, second, third),
                base_shifts,
                frame_gradients,
                strict=True,
            ):
                gradient_at_base = _scatter_gradient_to_base(
                    gradient,
                    base_shift,
                )
                _accumulate_oriented_gradient(
                    frame_gradient,
                    offset,
                    gradient_at_base,
                    stencil=parameters.stencil,
                    complex_group=False,
                )

    for index in range(len(unique)):
        phase = state.phase_links[..., index]
        derivative = np.real(
            np.conj(phase_gradient[..., index]) * (1.0j * phase)
        )
        phase_rate[..., index] -= derivative
        color = state.color_links[..., index, :, :]
        for generator_index, generator in enumerate(color_generators):
            variation = 1.0j * generator @ color
            derivative = np.real(
                np.sum(
                    np.conj(color_gradient[..., index, :, :])
                    * variation,
                    axis=(-2, -1),
                )
            )
            color_rate[..., index, generator_index] -= derivative
        if parameters.frame_enabled:
            frame = state.frame_links[..., index, :, :]
            for generator_index, generator in enumerate(frame_generators):
                variation = generator @ frame
                derivative = np.sum(
                    frame_gradient[..., index, :, :] * variation,
                    axis=(-2, -1),
                )
                frame_rate[..., index, generator_index] -= derivative

    shape_rate = _tracefree_symmetric(shape_rate)
    potential = float(sum(components.values()))
    rates = R3MomentumRates(
        matter=matter_rate,
        chi=chi_rate,
        shape=shape_rate,
        phase_electric=phase_rate,
        color_electric=color_rate,
        frame_electric=frame_rate,
    )
    return potential, rates, components


def kinetic_energy(
    state: R3LiveState,
    parameters: R3LiveParameters = R3LiveParameters(),
) -> tuple[float, dict[str, float]]:
    """Return the exact momentum-dependent Hamiltonian pieces."""

    _validate_state(state, parameters)
    q = (
        np.exp(state.shape[..., 0, 0])
        if parameters.frame_enabled
        else np.ones_like(state.chi)
    )
    bare_density = (
        0.5 * np.sum(np.abs(state.matter_momentum) ** 2, axis=-1)
        + state.chi_momentum**2 / (2.0 * parameters.frame_inertia)
    )
    components = {
        "weighted_bare_kinetic": float(np.sum(q * bare_density)),
        "shape_kinetic": (
            float(
                np.sum(state.shape_momentum**2)
                / (2.0 * parameters.frame_inertia)
            )
            if parameters.frame_enabled
            else 0.0
        ),
        "phase_electric": float(
            np.sum(state.phase_electric**2)
            / (2.0 * parameters.phase_inertia)
        ),
        "color_electric": float(
            np.sum(state.color_electric**2)
            / (2.0 * parameters.color_inertia)
        ),
        "frame_electric": (
            float(
                np.sum(state.frame_electric**2)
                / (2.0 * parameters.frame_inertia)
            )
            if parameters.frame_enabled
            else 0.0
        ),
    }
    return float(sum(components.values())), components


def total_hamiltonian(
    state: R3LiveState,
    parameters: R3LiveParameters = R3LiveParameters(),
) -> tuple[float, dict[str, float]]:
    kinetic, kinetic_parts = kinetic_energy(state, parameters)
    potential, _, potential_parts = potential_energy_and_rates(
        state,
        parameters,
    )
    parts = {**kinetic_parts, **potential_parts}
    return kinetic + potential, parts


def _potential_kick(
    state: R3LiveState,
    duration: float,
    parameters: R3LiveParameters,
) -> None:
    _, rates, _ = potential_energy_and_rates(state, parameters)
    state.matter_momentum += duration * rates.matter
    state.chi_momentum += duration * rates.chi
    if parameters.frame_enabled:
        state.shape_momentum += duration * rates.shape
    state.phase_electric += duration * rates.phase_electric
    state.color_electric += duration * rates.color_electric
    if parameters.frame_enabled:
        state.frame_electric += duration * rates.frame_electric


def _link_and_shape_drift(
    state: R3LiveState,
    duration: float,
    parameters: R3LiveParameters,
) -> None:
    if parameters.frame_enabled:
        state.shape += (
            duration * state.shape_momentum / parameters.frame_inertia
        )
        state.shape = _tracefree_symmetric(state.shape)
    color_generators = su3_generators()
    frame_generators = so4_generators()
    link_count = state.phase_links.shape[3]
    color_is_live = bool(np.any(state.color_electric != 0.0))
    frame_is_live = (
        parameters.frame_enabled
        and bool(np.any(state.frame_electric != 0.0))
    )
    for index in range(link_count):
        state.phase_links[..., index] *= np.exp(
            1.0j
            * duration
            * state.phase_electric[..., index]
            / parameters.phase_inertia
        )
        for site in np.ndindex(state.chi.shape):
            if color_is_live:
                color_algebra = np.einsum(
                    "a,aij->ij",
                    state.color_electric[site + (index,)],
                    color_generators,
                )
                state.color_links[site + (index,)] = (
                    expm(
                        1.0j
                        * duration
                        * color_algebra
                        / parameters.color_inertia
                    )
                    @ state.color_links[site + (index,)]
                )
            if frame_is_live:
                frame_algebra = np.einsum(
                    "a,aij->ij",
                    state.frame_electric[site + (index,)],
                    frame_generators,
                )
                state.frame_links[site + (index,)] = (
                    expm(
                        duration
                        * frame_algebra
                        / parameters.frame_inertia
                    )
                    @ state.frame_links[site + (index,)]
                )


def _weighted_bare_kinetic_drift(
    state: R3LiveState,
    duration: float,
    parameters: R3LiveParameters,
) -> None:
    q = (
        np.exp(state.shape[..., 0, 0])
        if parameters.frame_enabled
        else np.ones_like(state.chi)
    )
    kinetic_density = (
        0.5 * np.sum(np.abs(state.matter_momentum) ** 2, axis=-1)
        + state.chi_momentum**2 / (2.0 * parameters.frame_inertia)
    )
    state.matter += (
        duration * q[..., np.newaxis] * state.matter_momentum
    )
    state.chi += (
        duration
        * q
        * state.chi_momentum
        / parameters.frame_inertia
    )
    if parameters.frame_enabled:
        state.shape_momentum -= (
            duration * q * kinetic_density
        )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()


def step_r3_live(
    state: R3LiveState,
    dt: float,
    parameters: R3LiveParameters = R3LiveParameters(),
) -> None:
    """Advance one reversible second-order Hamiltonian splitting step."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    _validate_state(state, parameters)
    half = 0.5 * dt
    _potential_kick(state, half, parameters)
    _link_and_shape_drift(state, half, parameters)
    _weighted_bare_kinetic_drift(state, dt, parameters)
    _link_and_shape_drift(state, half, parameters)
    _potential_kick(state, half, parameters)


def reverse_momenta(state: R3LiveState) -> None:
    """Apply the exact experimental time-reversal momentum involution."""

    state.matter_momentum *= -1.0
    state.chi_momentum *= -1.0
    state.shape_momentum *= -1.0
    state.phase_electric *= -1.0
    state.color_electric *= -1.0
    state.frame_electric *= -1.0


def group_constraint_errors(state: R3LiveState) -> dict[str, float]:
    """Return maximum compact-link and shape-constraint violations."""

    phase = float(np.max(np.abs(np.abs(state.phase_links) - 1.0)))
    color_identity = state.color_links @ _dagger(state.color_links)
    color_unitarity = float(
        np.max(np.abs(color_identity - np.eye(3)))
    )
    color_determinant = float(
        np.max(np.abs(np.linalg.det(state.color_links) - 1.0))
    )
    frame_identity = state.frame_links @ _transpose(state.frame_links)
    frame_orthogonality = float(
        np.max(np.abs(frame_identity - np.eye(4)))
    )
    frame_determinant = float(
        np.max(np.abs(np.linalg.det(state.frame_links) - 1.0))
    )
    shape_symmetry = float(
        np.max(np.abs(state.shape - _transpose(state.shape)))
    )
    shape_trace = float(
        np.max(np.abs(np.trace(state.shape, axis1=-2, axis2=-1)))
    )
    return {
        "phase_unit": phase,
        "color_unitarity": color_unitarity,
        "color_determinant": color_determinant,
        "frame_orthogonality": frame_orthogonality,
        "frame_determinant": frame_determinant,
        "shape_symmetry": shape_symmetry,
        "shape_trace": shape_trace,
    }


def state_distance(left: R3LiveState, right: R3LiveState) -> float:
    """Return a relative Euclidean distance over the complete live state."""

    numerator = 0.0
    denominator = 0.0
    for name in left.__dataclass_fields__:
        left_values = np.asarray(getattr(left, name))
        right_values = np.asarray(getattr(right, name))
        numerator += float(np.sum(np.abs(left_values - right_values) ** 2))
        denominator += float(np.sum(np.abs(left_values) ** 2))
    return float(np.sqrt(numerator / max(denominator, 1.0)))


def r3_live_action_declaration(
    parameters: R3LiveParameters = R3LiveParameters(),
) -> dict[str, object]:
    """Return a machine-readable declaration of the live experiment."""

    return {
        "action_id": R3_LIVE_ACTION_ID,
        "register_id": R3_LIVE_REGISTER_ID,
        "canonical_status": "EXPERIMENT_ONLY_UNPROMOTED",
        "parameters": asdict(parameters),
        "frame_inertia": parameters.frame_inertia,
        "frame_enabled": parameters.frame_enabled,
        "chi_potential": parameters.chi_potential,
        "site_terms": [
            "q_times_bare_GOV01_GOV02_Hamiltonian",
            "traceless_frame_shape_kinetic_and_gradient",
        ],
        "link_terms": [
            "compact_electric_kinetic",
            "covariant_matter_neighbor_energy",
            "positive_triangle_holonomy_energy",
        ],
        "integrator": "symmetric_Hpotential_Hlink_Hbare_split",
        "forbidden_mechanisms_used": [],
        "paper_45_update_authorized": False,
    }


def r3_live_action_fingerprint(
    parameters: R3LiveParameters = R3LiveParameters(),
) -> str:
    encoded = json.dumps(
        r3_live_action_declaration(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
