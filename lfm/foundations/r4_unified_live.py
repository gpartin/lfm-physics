"""Live R4 experiment extending R3 with weak isospin and color dielectric.

R4 retains the complete live R3 register and adds a left-isospin doublet,
compact SU(2) links, and an SU(2)-valued site orientation whose radial
magnitude is the existing chi field. The covariant orientation-alignment
energy generates a weak-link gap without inputting a mediator mass.

The color electric and magnetic energies use a positive chi-dependent
dielectric. This is a local variational confinement candidate; no target
potential or string tension is supplied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from lfm.core.stencils import laplacian_19pt, laplacian_27pt
from lfm.foundations.r3_link_frame_live import (
    R3LiveParameters,
    R3LiveState,
    R3MomentumRates,
    _accumulate_oriented_gradient,
    _accumulate_oriented_phase_gradient,
    _dagger,
    _frame_loop_energy_gradient,
    _link_and_shape_drift,
    _link_table,
    _loop_products,
    _neighbor,
    _scatter_from_base,
    _scatter_gradient_to_base,
    _temporal_shape_projector,
    _tracefree_symmetric,
    _transpose,
    _validate_state,
    _weighted_bare_kinetic_drift,
    group_constraint_errors as r3_group_constraint_errors,
    kinetic_energy as r3_kinetic_energy,
    potential_energy_and_rates as r3_potential_energy_and_rates,
    so4_generators,
    state_distance as r3_state_distance,
    su3_generators,
    triangle_loops,
)


R4_ACTION_ID = "LFM-R4-UNIFIED-LIVE-EXPERIMENT-v1"
R4_REGISTER_ID = (
    "R4=(R3Live,PsiL_s,PiL_s,W_ij,EW_ij,H_i,EH_i)"
)


@lru_cache(maxsize=1)
def su2_generators() -> np.ndarray:
    """Return Pauli generators sigma_a/2."""

    return 0.5 * np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=np.complex128,
    )


@dataclass(frozen=True)
class R4Parameters:
    """Parameters derived from the existing LFM constant set."""

    r3: R3LiveParameters = field(default_factory=R3LiveParameters)

    @property
    def weak_stiffness(self) -> float:
        return 1.0 / self.r3.epsilon_w

    @property
    def weak_inertia(self) -> float:
        return 1.0

    @property
    def higgs_inertia(self) -> float:
        return self.r3.frame_inertia

    @property
    def higgs_alignment(self) -> float:
        return self.r3.epsilon_w

    @property
    def stencil(self) -> str:
        return self.r3.stencil


@dataclass
class R4State:
    """Complete R4 phase space."""

    r3: R3LiveState
    weak_matter: np.ndarray
    weak_momentum: np.ndarray
    weak_links: np.ndarray
    weak_electric: np.ndarray
    higgs_orientation: np.ndarray
    higgs_electric: np.ndarray

    @classmethod
    def vacuum(
        cls,
        size: int,
        parameters: R4Parameters = R4Parameters(),
    ) -> R4State:
        base = R3LiveState.vacuum(size, parameters.r3)
        sites = base.chi.shape
        link_count = base.phase_links.shape[3]
        return cls(
            r3=base,
            weak_matter=np.zeros(sites + (2,), dtype=np.complex128),
            weak_momentum=np.zeros(sites + (2,), dtype=np.complex128),
            weak_links=np.broadcast_to(
                np.eye(2, dtype=np.complex128),
                sites + (link_count, 2, 2),
            ).copy(),
            weak_electric=np.zeros(
                sites + (link_count, 3),
                dtype=np.float64,
            ),
            higgs_orientation=np.broadcast_to(
                np.eye(2, dtype=np.complex128),
                sites + (2, 2),
            ).copy(),
            higgs_electric=np.zeros(sites + (3,), dtype=np.float64),
        )

    def copy(self) -> R4State:
        return R4State(
            r3=self.r3.copy(),
            weak_matter=self.weak_matter.copy(),
            weak_momentum=self.weak_momentum.copy(),
            weak_links=self.weak_links.copy(),
            weak_electric=self.weak_electric.copy(),
            higgs_orientation=self.higgs_orientation.copy(),
            higgs_electric=self.higgs_electric.copy(),
        )


@dataclass
class R4Rates:
    r3: R3MomentumRates
    weak_matter: np.ndarray
    weak_electric: np.ndarray
    higgs_electric: np.ndarray


@dataclass
class R4FrameScalarState:
    """Exact single-polarization invariant sector of the R4 frame shape."""

    shape_amplitude: np.ndarray
    shape_momentum: np.ndarray

    @classmethod
    def vacuum(cls, size: int) -> R4FrameScalarState:
        if not isinstance(size, int) or size < 2:
            raise ValueError("size must be an integer >= 2")
        shape = (size, size, size)
        return cls(
            shape_amplitude=np.zeros(shape, dtype=np.float64),
            shape_momentum=np.zeros(shape, dtype=np.float64),
        )

    def copy(self) -> R4FrameScalarState:
        return R4FrameScalarState(
            shape_amplitude=self.shape_amplitude.copy(),
            shape_momentum=self.shape_momentum.copy(),
        )


def _frame_scalar_laplacian(
    values: np.ndarray,
    stencil: str,
) -> np.ndarray:
    if stencil == "19":
        return laplacian_19pt(values)
    if stencil == "27":
        return laplacian_27pt(values)
    raise ValueError("stencil must be '19' or '27'")


def _validate_frame_scalar(
    state: R4FrameScalarState,
    source_density: np.ndarray | None,
) -> np.ndarray:
    amplitude = np.asarray(state.shape_amplitude)
    momentum = np.asarray(state.shape_momentum)
    if amplitude.ndim != 3 or momentum.shape != amplitude.shape:
        raise ValueError(
            "frame scalar amplitude and momentum must share a 3D shape"
        )
    if not np.all(np.isfinite(amplitude)) or not np.all(
        np.isfinite(momentum)
    ):
        raise ValueError("frame scalar state contains non-finite values")
    if source_density is None:
        return np.zeros_like(amplitude)
    source = np.asarray(source_density, dtype=np.float64)
    if source.shape != amplitude.shape:
        raise ValueError("source_density must match the scalar frame shape")
    if not np.all(np.isfinite(source)) or np.any(source < 0.0):
        raise ValueError(
            "source_density must be finite and nonnegative"
        )
    return source


def r4_frame_scalar_energy(
    state: R4FrameScalarState,
    parameters: R4Parameters = R4Parameters(),
    *,
    source_density: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Return the exact R4 energy on the scalar frame invariant sector."""

    source = _validate_frame_scalar(state, source_density)
    amplitude = state.shape_amplitude
    momentum = state.shape_momentum
    polarization_norm_sq = 3.0 / 4.0
    laplacian = _frame_scalar_laplacian(
        amplitude,
        parameters.stencil,
    )
    kinetic = float(
        polarization_norm_sq
        * np.sum(momentum**2)
        / (2.0 * parameters.r3.frame_inertia)
    )
    gradient = float(
        -0.5
        * polarization_norm_sq
        * parameters.r3.frame_stiffness
        * np.sum(amplitude * laplacian)
    )
    source_energy = float(
        np.sum(source * np.exp(0.75 * amplitude))
    )
    parts = {
        "frame_scalar_kinetic": kinetic,
        "frame_scalar_gradient": gradient,
        "fixed_positive_source": source_energy,
    }
    return float(sum(parts.values())), parts


def step_r4_frame_scalar(
    state: R4FrameScalarState,
    dt: float,
    parameters: R4Parameters = R4Parameters(),
    *,
    source_density: np.ndarray | None = None,
) -> None:
    """Advance the exact scalar frame sector with local R4 leapfrog.

    A supplied source is a fixed positive-energy response probe. It uses the
    same exp(S00) coupling as R4 but is not an unsupported dynamical body.
    """

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    source = _validate_frame_scalar(state, source_density)
    half = 0.5 * dt

    def kick(duration: float) -> None:
        state.shape_momentum += duration * (
            parameters.r3.frame_stiffness
            * _frame_scalar_laplacian(
                state.shape_amplitude,
                parameters.stencil,
            )
            - source * np.exp(0.75 * state.shape_amplitude)
        )

    kick(half)
    state.shape_amplitude += (
        dt
        * state.shape_momentum
        / parameters.r3.frame_inertia
    )
    kick(half)


def _validate_r4(state: R4State, parameters: R4Parameters) -> None:
    sites = _validate_state(state.r3, parameters.r3)
    link_count = state.r3.phase_links.shape[3]
    expected = {
        "weak_matter": sites + (2,),
        "weak_momentum": sites + (2,),
        "weak_links": sites + (link_count, 2, 2),
        "weak_electric": sites + (link_count, 3),
        "higgs_orientation": sites + (2, 2),
        "higgs_electric": sites + (3,),
    }
    for name, shape in expected.items():
        values = np.asarray(getattr(state, name))
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")


def color_dielectric(
    chi: np.ndarray,
    parameters: R4Parameters = R4Parameters(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive color permittivity and its chi derivative."""

    ratio = np.asarray(chi, dtype=np.float64) / parameters.r3.chi0
    displacement = 1.0 - ratio**2
    floor = parameters.r3.kappa
    epsilon = floor + (1.0 - floor) * displacement**2
    derivative = (
        -4.0
        * (1.0 - floor)
        * np.asarray(chi, dtype=np.float64)
        * displacement
        / parameters.r3.chi0**2
    )
    return epsilon, derivative


def _link_average(values: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    return 0.5 * (values + _neighbor(values, offset))


def _add_phase_color_weight_corrections(
    state: R4State,
    parameters: R4Parameters,
    base_rates: R3MomentumRates,
    components: dict[str, float],
) -> float:
    """Weight R3 phase/color loop energy by q and color dielectric."""

    base = state.r3
    q = np.exp(base.shape[..., 0, 0])
    epsilon, epsilon_derivative = color_dielectric(base.chi, parameters)
    sites = base.chi.shape
    phase_gradient = np.zeros_like(base.phase_links)
    color_gradient = np.zeros_like(base.color_links)
    phase_correction = 0.0
    color_correction = 0.0
    color_generators = su3_generators()
    identity3 = np.eye(3, dtype=np.complex128)

    for first, second, third, loop_weight in triangle_loops(
        parameters.stencil
    ):
        phase_one, phase_two, phase_three, phase_holonomy = _loop_products(
            base.phase_links,
            first,
            second,
            third,
            complex_group=True,
        )
        phase_bare = (
            parameters.r3.phase_stiffness
            * loop_weight
            * (1.0 - np.real(phase_holonomy))
        )
        phase_factor = q
        phase_correction += float(np.sum((phase_factor - 1.0) * phase_bare))
        base_rates.shape -= (
            q * phase_bare
        )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()
        phase_h_gradient = (
            -(phase_factor - 1.0)
            * parameters.r3.phase_stiffness
            * loop_weight
        ).astype(np.complex128)
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
            _accumulate_oriented_phase_gradient(
                phase_gradient,
                offset,
                _scatter_gradient_to_base(gradient, base_shift),
                stencil=parameters.stencil,
            )

        color_one, color_two, color_three, color_holonomy = _loop_products(
            base.color_links,
            first,
            second,
            third,
            complex_group=True,
        )
        color_bare = (
            parameters.r3.color_stiffness
            * loop_weight
            * (
                3.0
                - np.real(
                    np.trace(color_holonomy, axis1=-2, axis2=-1)
                )
            )
        )
        color_factor = q * epsilon
        color_correction += float(np.sum((color_factor - 1.0) * color_bare))
        base_rates.shape -= (
            q * epsilon * color_bare
        )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()
        base_rates.chi -= q * epsilon_derivative * color_bare
        color_h_gradient = (
            -(color_factor - 1.0)
            * parameters.r3.color_stiffness
            * loop_weight
        )[..., np.newaxis, np.newaxis] * identity3
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
            _accumulate_oriented_gradient(
                color_gradient,
                offset,
                _scatter_gradient_to_base(gradient, base_shift),
                stencil=parameters.stencil,
                complex_group=True,
            )

    for index in range(base.phase_links.shape[3]):
        phase = base.phase_links[..., index]
        derivative = np.real(
            np.conj(phase_gradient[..., index]) * (1.0j * phase)
        )
        base_rates.phase_electric[..., index] -= derivative
        color = base.color_links[..., index, :, :]
        for generator_index, generator in enumerate(color_generators):
            variation = 1.0j * generator @ color
            derivative = np.real(
                np.sum(
                    np.conj(color_gradient[..., index, :, :])
                    * variation,
                    axis=(-2, -1),
                )
            )
            base_rates.color_electric[..., index, generator_index] -= (
                derivative
            )
    components["phase_loop_weight_correction"] = phase_correction
    components["color_loop_dielectric_correction"] = color_correction
    return phase_correction + color_correction


def potential_energy_and_rates(
    state: R4State,
    parameters: R4Parameters = R4Parameters(),
) -> tuple[float, R4Rates, dict[str, float]]:
    """Return the complete R4 coordinate energy and momentum rates."""

    _validate_r4(state, parameters)
    base_energy, base_rates, base_parts = r3_potential_energy_and_rates(
        state.r3,
        parameters.r3,
    )
    weak_rate = np.zeros_like(state.weak_matter)
    weak_electric_rate = np.zeros_like(state.weak_electric)
    higgs_electric_rate = np.zeros_like(state.higgs_electric)
    q = np.exp(state.r3.shape[..., 0, 0])
    weak_source_density = np.zeros_like(q)
    components = dict(base_parts)
    correction = _add_phase_color_weight_corrections(
        state,
        parameters,
        base_rates,
        components,
    )
    weak_gradient_energy = 0.0
    weak_onsite_energy = 0.0
    higgs_alignment_energy = 0.0
    weak_loop_energy = 0.0
    weak_generators = su2_generators()
    unique, _ = _link_table(parameters.stencil)

    for index, (offset, weight) in enumerate(unique):
        matter_j = _neighbor(state.weak_matter, offset)
        q_j = _neighbor(q, offset)
        weak_link = state.weak_links[..., index, :, :]
        phase = state.r3.phase_links[..., index]
        transported = phase[..., np.newaxis] * np.einsum(
            "...ab,...b->...a",
            weak_link,
            matter_j,
        )
        difference = transported - state.weak_matter
        norm_sq = np.sum(np.abs(difference) ** 2, axis=-1)
        source_half = (
            0.25 * parameters.r3.wave_speed**2 * weight * norm_sq
        )
        weak_gradient_energy += float(np.sum((q + q_j) * source_half))
        weak_source_density += source_half
        weak_source_density += _scatter_from_base(source_half, offset)
        force_scale = (
            0.5
            * parameters.r3.wave_speed**2
            * weight
            * (q + q_j)
        )
        weak_rate += force_scale[..., np.newaxis] * difference
        neighbor_force = (
            -force_scale[..., np.newaxis]
            * np.conj(phase)[..., np.newaxis]
            * np.einsum(
                "...ab,...b->...a",
                _dagger(weak_link),
                difference,
            )
        )
        weak_rate += _scatter_from_base(neighbor_force, offset)
        phase_derivative = force_scale * np.real(
            np.sum(
                np.conj(difference) * (1.0j * transported),
                axis=-1,
            )
        )
        base_rates.phase_electric[..., index] -= phase_derivative
        for generator_index, generator in enumerate(weak_generators):
            variation = 1.0j * np.einsum(
                "ab,...b->...a",
                generator,
                transported,
            )
            derivative = force_scale * np.real(
                np.sum(np.conj(difference) * variation, axis=-1)
            )
            weak_electric_rate[..., index, generator_index] -= derivative

        higgs_j = _neighbor(state.higgs_orientation, offset)
        chi_j = _neighbor(state.r3.chi, offset)
        transported_higgs = weak_link @ higgs_j
        higgs_difference = transported_higgs - state.higgs_orientation
        higgs_norm_sq = np.sum(
            np.abs(higgs_difference) ** 2,
            axis=(-2, -1),
        )
        chi_sq_sum = state.r3.chi**2 + chi_j**2
        q_sum = q + q_j
        alignment_coefficient = (
            0.125 * parameters.higgs_alignment * q_sum * chi_sq_sum
        )
        alignment_density = alignment_coefficient * higgs_norm_sq
        higgs_alignment_energy += float(np.sum(alignment_density))
        alignment_force_scale = (
            0.25 * parameters.higgs_alignment * q_sum * chi_sq_sum
        )
        for generator_index, generator in enumerate(weak_generators):
            left_variation = -1.0j * generator @ state.higgs_orientation
            left_derivative = alignment_force_scale * np.real(
                np.sum(
                    np.conj(higgs_difference) * left_variation,
                    axis=(-2, -1),
                )
            )
            higgs_electric_rate[..., generator_index] -= left_derivative
            right_variation = (
                weak_link @ (1.0j * generator @ higgs_j)
            )
            right_derivative = alignment_force_scale * np.real(
                np.sum(
                    np.conj(higgs_difference) * right_variation,
                    axis=(-2, -1),
                )
            )
            higgs_electric_rate[..., generator_index] += _scatter_from_base(
                -right_derivative,
                offset,
            )
            link_variation = (
                1.0j * generator @ transported_higgs
            )
            link_derivative = alignment_force_scale * np.real(
                np.sum(
                    np.conj(higgs_difference) * link_variation,
                    axis=(-2, -1),
                )
            )
            weak_electric_rate[..., index, generator_index] -= (
                link_derivative
            )
        chi_derivative = (
            0.25
            * parameters.higgs_alignment
            * q_sum
            * state.r3.chi
            * higgs_norm_sq
        )
        base_rates.chi -= chi_derivative
        neighbor_chi_derivative = (
            0.25
            * parameters.higgs_alignment
            * q_sum
            * chi_j
            * higgs_norm_sq
        )
        base_rates.chi += _scatter_from_base(
            -neighbor_chi_derivative,
            offset,
        )
        endpoint_source = (
            0.125
            * parameters.higgs_alignment
            * chi_sq_sum
            * higgs_norm_sq
        )
        weak_source_density += endpoint_source
        weak_source_density += _scatter_from_base(endpoint_source, offset)

    weak_norm_sq = np.sum(np.abs(state.weak_matter) ** 2, axis=-1)
    weak_onsite = 0.5 * state.r3.chi**2 * weak_norm_sq
    weak_onsite_energy = float(np.sum(q * weak_onsite))
    weak_source_density += weak_onsite
    weak_rate -= (
        q * state.r3.chi**2
    )[..., np.newaxis] * state.weak_matter
    base_rates.chi -= q * state.r3.chi * weak_norm_sq

    weak_gradient = np.zeros_like(state.weak_links)
    identity2 = np.eye(2, dtype=np.complex128)
    for first, second, third, loop_weight in triangle_loops(
        parameters.stencil
    ):
        first_link, second_link, third_link, holonomy = _loop_products(
            state.weak_links,
            first,
            second,
            third,
            complex_group=True,
        )
        bare_density = (
            parameters.weak_stiffness
            * loop_weight
            * (
                2.0
                - np.real(np.trace(holonomy, axis1=-2, axis2=-1))
            )
        )
        weak_loop_energy += float(np.sum(q * bare_density))
        base_rates.shape -= (
            q * bare_density
        )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()
        holonomy_gradient = (
            -q * parameters.weak_stiffness * loop_weight
        )[..., np.newaxis, np.newaxis] * identity2
        gradients = (
            holonomy_gradient @ _dagger(second_link @ third_link),
            _dagger(first_link)
            @ holonomy_gradient
            @ _dagger(third_link),
            _dagger(first_link @ second_link) @ holonomy_gradient,
        )
        base_shifts = (
            (0, 0, 0),
            first,
            tuple(first[axis] + second[axis] for axis in range(3)),
        )
        for offset, base_shift, gradient in zip(
            (first, second, third),
            base_shifts,
            gradients,
            strict=True,
        ):
            _accumulate_oriented_gradient(
                weak_gradient,
                offset,
                _scatter_gradient_to_base(gradient, base_shift),
                stencil=parameters.stencil,
                complex_group=True,
            )
    for index in range(state.weak_links.shape[3]):
        link = state.weak_links[..., index, :, :]
        for generator_index, generator in enumerate(weak_generators):
            variation = 1.0j * generator @ link
            derivative = np.real(
                np.sum(
                    np.conj(weak_gradient[..., index, :, :])
                    * variation,
                    axis=(-2, -1),
                )
            )
            weak_electric_rate[..., index, generator_index] -= derivative

    base_rates.shape -= (
        q * weak_source_density
    )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()
    base_rates.shape = _tracefree_symmetric(base_rates.shape)
    components.update(
        {
            "weak_matter_gradient": weak_gradient_energy,
            "weak_matter_onsite": weak_onsite_energy,
            "higgs_alignment": higgs_alignment_energy,
            "weak_loop": weak_loop_energy,
        }
    )
    total = (
        base_energy
        + correction
        + weak_gradient_energy
        + weak_onsite_energy
        + higgs_alignment_energy
        + weak_loop_energy
    )
    return (
        total,
        R4Rates(
            r3=base_rates,
            weak_matter=weak_rate,
            weak_electric=weak_electric_rate,
            higgs_electric=higgs_electric_rate,
        ),
        components,
    )


def _gauge_kinetic_replacements(
    state: R4State,
    parameters: R4Parameters,
) -> tuple[float, dict[str, float]]:
    base = state.r3
    q = np.exp(base.shape[..., 0, 0])
    epsilon, _ = color_dielectric(base.chi, parameters)
    unique, _ = _link_table(parameters.stencil)
    phase = 0.0
    color = 0.0
    frame = 0.0
    weak = 0.0
    for index, (offset, _) in enumerate(unique):
        q_link = _link_average(q, offset)
        epsilon_link = _link_average(epsilon, offset)
        phase += float(
            np.sum(
                q_link
                * base.phase_electric[..., index] ** 2
                / (2.0 * parameters.r3.phase_inertia)
            )
        )
        color += float(
            np.sum(
                q_link[..., np.newaxis]
                * base.color_electric[..., index, :] ** 2
                / (
                    2.0
                    * parameters.r3.color_inertia
                    * epsilon_link[..., np.newaxis]
                )
            )
        )
        frame += float(
            np.sum(
                q_link[..., np.newaxis]
                * base.frame_electric[..., index, :] ** 2
                / (
                    2.0
                    * parameters.r3.frame_inertia
                )
            )
        )
        weak += float(
            np.sum(
                q_link[..., np.newaxis]
                * state.weak_electric[..., index, :] ** 2
                / (2.0 * parameters.weak_inertia)
            )
        )
    higgs = float(
        np.sum(
            q[..., np.newaxis] * state.higgs_electric**2
        )
        / (2.0 * parameters.higgs_inertia)
    )
    return phase + color + frame + weak + higgs, {
        "phase_electric_weighted": phase,
        "color_electric_dielectric": color,
        "frame_electric_weighted": frame,
        "weak_electric_weighted": weak,
        "higgs_orientation_kinetic": higgs,
    }


def kinetic_energy(
    state: R4State,
    parameters: R4Parameters = R4Parameters(),
) -> tuple[float, dict[str, float]]:
    """Return the complete positive R4 kinetic energy."""

    _validate_r4(state, parameters)
    base_energy, base_parts = r3_kinetic_energy(state.r3, parameters.r3)
    q = np.exp(state.r3.shape[..., 0, 0])
    weak_matter = float(
        np.sum(
            q
            * 0.5
            * np.sum(np.abs(state.weak_momentum) ** 2, axis=-1)
        )
    )
    replacement, replacement_parts = _gauge_kinetic_replacements(
        state,
        parameters,
    )
    old_gauge = (
        base_parts["phase_electric"]
        + base_parts["color_electric"]
        + base_parts["frame_electric"]
    )
    parts = dict(base_parts)
    parts.update(replacement_parts)
    parts["weak_matter_kinetic"] = weak_matter
    parts["replaced_R3_gauge_electric"] = -old_gauge
    total = base_energy - old_gauge + replacement + weak_matter
    return total, parts


def total_hamiltonian(
    state: R4State,
    parameters: R4Parameters = R4Parameters(),
) -> tuple[float, dict[str, float]]:
    kinetic, kinetic_parts = kinetic_energy(state, parameters)
    potential, _, potential_parts = potential_energy_and_rates(
        state,
        parameters,
    )
    return kinetic + potential, {**kinetic_parts, **potential_parts}


def _potential_kick(
    state: R4State,
    duration: float,
    parameters: R4Parameters,
) -> None:
    _, rates, _ = potential_energy_and_rates(state, parameters)
    base = state.r3
    base.matter_momentum += duration * rates.r3.matter
    base.chi_momentum += duration * rates.r3.chi
    base.shape_momentum += duration * rates.r3.shape
    base.phase_electric += duration * rates.r3.phase_electric
    base.color_electric += duration * rates.r3.color_electric
    base.frame_electric += duration * rates.r3.frame_electric
    state.weak_momentum += duration * rates.weak_matter
    state.weak_electric += duration * rates.weak_electric
    state.higgs_electric += duration * rates.higgs_electric


def _extra_gauge_kinetic_drift(
    state: R4State,
    duration: float,
    parameters: R4Parameters,
) -> None:
    base = state.r3
    q = np.exp(base.shape[..., 0, 0])
    epsilon, epsilon_derivative = color_dielectric(base.chi, parameters)
    unique, _ = _link_table(parameters.stencil)
    weak_generators = su2_generators()
    color_generators = su3_generators()
    frame_generators = so4_generators()
    source_density = np.zeros_like(q)
    color_is_live = bool(np.any(base.color_electric != 0.0))
    frame_is_live = bool(np.any(base.frame_electric != 0.0))
    weak_is_live = bool(np.any(state.weak_electric != 0.0))

    for index, (offset, _) in enumerate(unique):
        q_link = _link_average(q, offset)
        epsilon_link = _link_average(epsilon, offset)
        epsilon_derivative_j = _neighbor(epsilon_derivative, offset)
        phase_coefficient = q_link - 1.0
        base.phase_links[..., index] *= np.exp(
            1.0j
            * duration
            * phase_coefficient
            * base.phase_electric[..., index]
            / parameters.r3.phase_inertia
        )
        phase_density = (
            base.phase_electric[..., index] ** 2
            / (4.0 * parameters.r3.phase_inertia)
        )
        source_density += phase_density
        source_density += _scatter_from_base(phase_density, offset)

        color_factor = q_link / epsilon_link
        color_density = np.sum(
            base.color_electric[..., index, :] ** 2,
            axis=-1,
        ) / (4.0 * parameters.r3.color_inertia * epsilon_link)
        source_density += color_density
        source_density += _scatter_from_base(color_density, offset)
        color_energy_sq = np.sum(
            base.color_electric[..., index, :] ** 2,
            axis=-1,
        )
        chi_rate_i = (
            q_link
            * color_energy_sq
            * epsilon_derivative
            / (
                4.0
                * parameters.r3.color_inertia
                * epsilon_link**2
            )
        )
        chi_rate_j = (
            q_link
            * color_energy_sq
            * epsilon_derivative_j
            / (
                4.0
                * parameters.r3.color_inertia
                * epsilon_link**2
            )
        )
        base.chi_momentum += duration * chi_rate_i
        base.chi_momentum += _scatter_from_base(
            duration * chi_rate_j,
            offset,
        )

        frame_density = np.sum(
            base.frame_electric[..., index, :] ** 2,
            axis=-1,
        ) / (4.0 * parameters.r3.frame_inertia)
        source_density += frame_density
        source_density += _scatter_from_base(frame_density, offset)

        weak_density = np.sum(
            state.weak_electric[..., index, :] ** 2,
            axis=-1,
        ) / (4.0 * parameters.weak_inertia)
        source_density += weak_density
        source_density += _scatter_from_base(weak_density, offset)

        if color_is_live:
            active_color_sites = np.argwhere(
                np.any(
                    base.color_electric[..., index, :] != 0.0,
                    axis=-1,
                )
            )
            for site_values in active_color_sites:
                site = tuple(int(value) for value in site_values)
                color_algebra = np.einsum(
                    "a,aij->ij",
                    base.color_electric[site + (index,)],
                    color_generators,
                )
                base.color_links[site + (index,)] = (
                    expm(
                        1.0j
                        * duration
                        * (color_factor[site] - 1.0)
                        * color_algebra
                        / parameters.r3.color_inertia
                    )
                    @ base.color_links[site + (index,)]
                )
        if frame_is_live:
            active_frame_sites = np.argwhere(
                np.any(
                    base.frame_electric[..., index, :] != 0.0,
                    axis=-1,
                )
            )
            for site_values in active_frame_sites:
                site = tuple(int(value) for value in site_values)
                frame_algebra = np.einsum(
                    "a,aij->ij",
                    base.frame_electric[site + (index,)],
                    frame_generators,
                )
                base.frame_links[site + (index,)] = (
                    expm(
                        duration
                        * (q_link[site] - 1.0)
                        * frame_algebra
                        / parameters.r3.frame_inertia
                    )
                    @ base.frame_links[site + (index,)]
                )
        if weak_is_live:
            active_weak_sites = np.argwhere(
                np.any(
                    state.weak_electric[..., index, :] != 0.0,
                    axis=-1,
                )
            )
            for site_values in active_weak_sites:
                site = tuple(int(value) for value in site_values)
                weak_algebra = np.einsum(
                    "a,aij->ij",
                    state.weak_electric[site + (index,)],
                    weak_generators,
                )
                state.weak_links[site + (index,)] = (
                    expm(
                        1.0j
                        * duration
                        * q_link[site]
                        * weak_algebra
                        / parameters.weak_inertia
                    )
                    @ state.weak_links[site + (index,)]
                )

    if np.any(state.higgs_electric != 0.0):
        for site in np.ndindex(base.chi.shape):
            higgs_algebra = np.einsum(
                "a,aij->ij",
                state.higgs_electric[site],
                weak_generators,
            )
            state.higgs_orientation[site] = (
                expm(
                    1.0j
                    * duration
                    * q[site]
                    * higgs_algebra
                    / parameters.higgs_inertia
                )
                @ state.higgs_orientation[site]
            )
    higgs_density = np.sum(state.higgs_electric**2, axis=-1) / (
        2.0 * parameters.higgs_inertia
    )
    source_density += higgs_density
    base.shape_momentum -= (
        duration * q * source_density
    )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()


def _weak_matter_kinetic_drift(
    state: R4State,
    duration: float,
    parameters: R4Parameters,
) -> None:
    q = np.exp(state.r3.shape[..., 0, 0])
    density = 0.5 * np.sum(
        np.abs(state.weak_momentum) ** 2,
        axis=-1,
    )
    state.weak_matter += (
        duration * q[..., np.newaxis] * state.weak_momentum
    )
    state.r3.shape_momentum -= (
        duration * q * density
    )[..., np.newaxis, np.newaxis] * _temporal_shape_projector()


def step_r4(
    state: R4State,
    dt: float,
    parameters: R4Parameters = R4Parameters(),
) -> None:
    """Advance one symmetric second-order R4 Hamiltonian split."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    _validate_r4(state, parameters)
    half = 0.5 * dt
    _potential_kick(state, half, parameters)
    _link_and_shape_drift(state.r3, half, parameters.r3)
    _extra_gauge_kinetic_drift(state, half, parameters)
    _weighted_bare_kinetic_drift(state.r3, dt, parameters.r3)
    _weak_matter_kinetic_drift(state, dt, parameters)
    _extra_gauge_kinetic_drift(state, half, parameters)
    _link_and_shape_drift(state.r3, half, parameters.r3)
    _potential_kick(state, half, parameters)


def reverse_momenta(state: R4State) -> None:
    """Reverse every R4 canonical momentum."""

    state.r3.matter_momentum *= -1.0
    state.r3.chi_momentum *= -1.0
    state.r3.shape_momentum *= -1.0
    state.r3.phase_electric *= -1.0
    state.r3.color_electric *= -1.0
    state.r3.frame_electric *= -1.0
    state.weak_momentum *= -1.0
    state.weak_electric *= -1.0
    state.higgs_electric *= -1.0


def group_constraint_errors(state: R4State) -> dict[str, float]:
    """Return compact-group errors for the complete R3 plus R4 register."""

    weak_identity = state.weak_links @ _dagger(state.weak_links)
    higgs_identity = state.higgs_orientation @ _dagger(
        state.higgs_orientation
    )
    return {
        **r3_group_constraint_errors(state.r3),
        "weak_unitarity": float(
            np.max(np.abs(weak_identity - np.eye(2)))
        ),
        "weak_determinant": float(
            np.max(np.abs(np.linalg.det(state.weak_links) - 1.0))
        ),
        "higgs_unitarity": float(
            np.max(np.abs(higgs_identity - np.eye(2)))
        ),
        "higgs_determinant": float(
            np.max(
                np.abs(np.linalg.det(state.higgs_orientation) - 1.0)
            )
        ),
    }


def state_distance(left: R4State, right: R4State) -> float:
    numerator = r3_state_distance(left.r3, right.r3) ** 2
    denominator = 1.0
    for name in (
        "weak_matter",
        "weak_momentum",
        "weak_links",
        "weak_electric",
        "higgs_orientation",
        "higgs_electric",
    ):
        left_values = np.asarray(getattr(left, name))
        right_values = np.asarray(getattr(right, name))
        numerator += float(np.sum(np.abs(left_values - right_values) ** 2))
        denominator += float(np.sum(np.abs(left_values) ** 2))
    return float(np.sqrt(numerator / denominator))


def r4_action_declaration(
    parameters: R4Parameters = R4Parameters(),
) -> dict[str, object]:
    return {
        "action_id": R4_ACTION_ID,
        "register_id": R4_REGISTER_ID,
        "canonical_status": "EXPERIMENT_ONLY_UNPROMOTED",
        "parameters": asdict(parameters),
        "derived_parameters": {
            "weak_stiffness": parameters.weak_stiffness,
            "weak_inertia": parameters.weak_inertia,
            "higgs_inertia": parameters.higgs_inertia,
            "higgs_alignment": parameters.higgs_alignment,
            "dielectric_floor": parameters.r3.kappa,
        },
        "added_terms": [
            "left_doublet_covariant_neighbor_energy",
            "positive_SU2_plaquette_energy",
            "positive_chi_Higgs_orientation_alignment",
            "positive_chi_color_dielectric",
            "total_added_energy_frame_source",
        ],
        "forbidden_mechanisms_used": [],
        "paper_45_update_authorized": False,
    }


def r4_action_fingerprint(
    parameters: R4Parameters = R4Parameters(),
) -> str:
    encoded = json.dumps(
        r4_action_declaration(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
