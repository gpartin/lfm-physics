"""Live Hamiltonian diagnostics for the unpromoted LFM clock-link candidate.

This module evolves the scalar bare GOV-01/GOV-02 Hamiltonian together with a
positive temporal-link factor ``q = exp(varphi)``. It contains no trajectory
law, quasi-static solve, or imported gravitational potential.

The implementation is intentionally separate from ``lfm.Simulation`` because
the clock link is a research candidate, not a canonical LFM register.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfm.constants import C_DEFAULT, CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import laplacian_19pt, laplacian_27pt

Offset = tuple[int, int, int]


def _offset3(values: tuple[int, ...]) -> Offset:
    return (values[0], values[1], values[2])


def _unique_links(stencil: str) -> tuple[tuple[tuple[int, int, int], float], ...]:
    if stencil == "19":
        return (
            ((1, 0, 0), 1.0 / 3.0),
            ((0, 1, 0), 1.0 / 3.0),
            ((0, 0, 1), 1.0 / 3.0),
            ((1, 1, 0), 1.0 / 6.0),
            ((1, -1, 0), 1.0 / 6.0),
            ((1, 0, 1), 1.0 / 6.0),
            ((1, 0, -1), 1.0 / 6.0),
            ((0, 1, 1), 1.0 / 6.0),
            ((0, 1, -1), 1.0 / 6.0),
        )
    if stencil == "27":
        return (
            ((1, 0, 0), 4.0 / 9.0),
            ((0, 1, 0), 4.0 / 9.0),
            ((0, 0, 1), 4.0 / 9.0),
            ((1, 1, 0), 1.0 / 9.0),
            ((1, -1, 0), 1.0 / 9.0),
            ((1, 0, 1), 1.0 / 9.0),
            ((1, 0, -1), 1.0 / 9.0),
            ((0, 1, 1), 1.0 / 9.0),
            ((0, 1, -1), 1.0 / 9.0),
            ((1, 1, 1), 1.0 / 36.0),
            ((1, 1, -1), 1.0 / 36.0),
            ((1, -1, 1), 1.0 / 36.0),
            ((1, -1, -1), 1.0 / 36.0),
        )
    raise ValueError("stencil must be '19' or '27'")


def _all_links(stencil: str) -> tuple[tuple[tuple[int, int, int], float], ...]:
    result: list[tuple[tuple[int, int, int], float]] = []
    for offset, weight in _unique_links(stencil):
        result.append((offset, weight))
        result.append((_offset3(tuple(-value for value in offset)), weight))
    return tuple(result)


def _shift(field: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    return np.roll(field, shift=offset, axis=(0, 1, 2))


def _laplacian(field: np.ndarray, stencil: str) -> np.ndarray:
    if stencil == "19":
        return laplacian_19pt(field)
    if stencil == "27":
        return laplacian_27pt(field)
    raise ValueError("stencil must be '19' or '27'")


@dataclass(frozen=True)
class LiveClockParameters:
    """Parameters and stencil assignments for the candidate live system."""

    chi0: float = CHI0
    kappa: float = KAPPA
    lambda_h: float = LAMBDA_H
    matter_speed: float = C_DEFAULT
    clock_inertia: float = CHI0 / KAPPA
    clock_speed: float = C_DEFAULT
    gov01_stencil: str = "19"
    gov02_stencil: str = "19"
    clock_stencil: str = "19"

    @property
    def chi_inertia(self) -> float:
        return self.chi0 / self.kappa

    def __post_init__(self) -> None:
        positive = (
            self.chi0,
            self.kappa,
            self.lambda_h,
            self.matter_speed,
            self.clock_inertia,
            self.clock_speed,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("live clock parameters must be positive and finite")
        for stencil in (
            self.gov01_stencil,
            self.gov02_stencil,
            self.clock_stencil,
        ):
            _unique_links(stencil)


@dataclass
class LiveClockState:
    """Canonical coordinates and momenta for a scalar live candidate run."""

    field: np.ndarray
    field_momentum: np.ndarray
    chi: np.ndarray
    chi_momentum: np.ndarray
    varphi: np.ndarray
    clock_momentum: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.field,
            self.field_momentum,
            self.chi,
            self.chi_momentum,
            self.varphi,
            self.clock_momentum,
        )
        shape = np.asarray(self.field).shape
        if len(shape) != 3 or any(np.asarray(value).shape != shape for value in arrays):
            raise ValueError("all live clock arrays must share one 3-D shape")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("live clock state must contain only finite values")

    def copy(self) -> LiveClockState:
        return LiveClockState(
            field=self.field.copy(),
            field_momentum=self.field_momentum.copy(),
            chi=self.chi.copy(),
            chi_momentum=self.chi_momentum.copy(),
            varphi=self.varphi.copy(),
            clock_momentum=self.clock_momentum.copy(),
        )


def positive_gradient_density(
    field: np.ndarray,
    *,
    coefficient: float,
    stencil: str,
) -> np.ndarray:
    """Return a positive site density with half of each link at each end."""
    values = np.asarray(field, dtype=np.float64)
    density = np.zeros_like(values)
    for offset, weight in _all_links(stencil):
        difference = _shift(values, offset) - values
        density += 0.25 * coefficient * weight * difference**2
    return density


def weighted_gradient_force(
    field: np.ndarray,
    clock_factor: np.ndarray,
    *,
    coefficient: float,
    stencil: str,
) -> np.ndarray:
    """Return minus the field derivative of the clock-weighted link energy."""
    force, _ = _weighted_gradient_force_density(
        field,
        clock_factor,
        coefficient=coefficient,
        stencil=stencil,
    )
    return force


def _weighted_gradient_force_density(
    field: np.ndarray,
    clock_factor: np.ndarray,
    *,
    coefficient: float,
    stencil: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted force and unweighted positive link density."""
    values = np.asarray(field, dtype=np.float64)
    q = np.asarray(clock_factor, dtype=np.float64)
    if values.shape != q.shape:
        raise ValueError("field and clock factor must have matching shapes")
    force = np.zeros_like(values)
    density = np.zeros_like(values)
    for offset, weight in _all_links(stencil):
        neighbor = _shift(values, offset)
        neighbor_q = _shift(q, offset)
        difference = neighbor - values
        force += 0.5 * coefficient * weight * (q + neighbor_q) * difference
        density += 0.25 * coefficient * weight * difference**2
    return force, density


def bare_kinetic_density(
    state: LiveClockState,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> np.ndarray:
    """Return the positive bare momentum density."""
    return 0.5 * state.field_momentum**2 + (state.chi_momentum**2 / (2.0 * parameters.chi_inertia))


def bare_potential_density(
    state: LiveClockState,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> np.ndarray:
    """Return the positive bare coordinate and neighbor-link density."""
    matter_gradient = positive_gradient_density(
        state.field,
        coefficient=parameters.matter_speed**2,
        stencil=parameters.gov01_stencil,
    )
    chi_gradient = positive_gradient_density(
        state.chi,
        coefficient=parameters.chi_inertia * parameters.matter_speed**2,
        stencil=parameters.gov02_stencil,
    )
    interaction = 0.5 * state.chi**2 * state.field**2
    radial = parameters.chi_inertia * parameters.lambda_h * (state.chi**2 - parameters.chi0**2) ** 2
    return matter_gradient + chi_gradient + interaction + radial


def bare_energy_density(
    state: LiveClockState,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> np.ndarray:
    """Return the complete positive scalar bare GOV-01/GOV-02 energy."""
    return bare_kinetic_density(state, parameters) + bare_potential_density(
        state,
        parameters,
    )


def clock_factor(state: LiveClockState) -> np.ndarray:
    """Return the positive temporal-link factor without clipping."""
    return np.exp(state.varphi)


def total_hamiltonian(
    state: LiveClockState,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> float:
    """Evaluate the autonomous live candidate Hamiltonian."""
    q = clock_factor(state)
    bare = bare_energy_density(state, parameters)
    clock_kinetic = state.clock_momentum**2 / (2.0 * parameters.clock_inertia)
    clock_gradient = positive_gradient_density(
        state.varphi,
        coefficient=parameters.clock_inertia * parameters.clock_speed**2,
        stencil=parameters.clock_stencil,
    )
    return float(np.sum(q * bare + clock_kinetic + clock_gradient))


def potential_momentum_rates(
    state: LiveClockState,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return momentum rates from coordinate-dependent Hamiltonian terms."""
    q = clock_factor(state)
    field_rate, matter_gradient = _weighted_gradient_force_density(
        state.field,
        q,
        coefficient=parameters.matter_speed**2,
        stencil=parameters.gov01_stencil,
    )
    field_rate -= q * state.chi**2 * state.field

    chi_rate, chi_gradient = _weighted_gradient_force_density(
        state.chi,
        q,
        coefficient=parameters.chi_inertia * parameters.matter_speed**2,
        stencil=parameters.gov02_stencil,
    )
    chi_rate -= q * (
        state.chi * state.field**2
        + 4.0
        * parameters.chi_inertia
        * parameters.lambda_h
        * state.chi
        * (state.chi**2 - parameters.chi0**2)
    )

    interaction = 0.5 * state.chi**2 * state.field**2
    radial = parameters.chi_inertia * parameters.lambda_h * (state.chi**2 - parameters.chi0**2) ** 2
    potential = matter_gradient + chi_gradient + interaction + radial
    clock_rate = (
        parameters.clock_inertia
        * parameters.clock_speed**2
        * _laplacian(state.varphi, parameters.clock_stencil)
        - q * potential
    )
    return field_rate, chi_rate, clock_rate


def clock_momentum_rate(
    state: LiveClockState,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> np.ndarray:
    """Return the complete instantaneous clock momentum rate."""
    q = clock_factor(state)
    return parameters.clock_inertia * parameters.clock_speed**2 * _laplacian(
        state.varphi, parameters.clock_stencil
    ) - q * bare_energy_density(state, parameters)


def _potential_kick(
    state: LiveClockState,
    duration: float,
    parameters: LiveClockParameters,
) -> None:
    field_rate, chi_rate, clock_rate = potential_momentum_rates(
        state,
        parameters,
    )
    state.field_momentum += duration * field_rate
    state.chi_momentum += duration * chi_rate
    state.clock_momentum += duration * clock_rate


def _clock_kinetic_drift(
    state: LiveClockState,
    duration: float,
    parameters: LiveClockParameters,
) -> None:
    state.varphi += duration * state.clock_momentum / parameters.clock_inertia


def _bare_kinetic_drift(
    state: LiveClockState,
    duration: float,
    parameters: LiveClockParameters,
) -> None:
    q = clock_factor(state)
    kinetic = bare_kinetic_density(state, parameters)
    state.field += duration * q * state.field_momentum
    state.chi += duration * q * state.chi_momentum / parameters.chi_inertia
    state.clock_momentum -= duration * q * kinetic


def step_live_clock(
    state: LiveClockState,
    dt: float,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> None:
    """Advance one symmetric second-order Hamiltonian-splitting step."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    half = 0.5 * dt
    _potential_kick(state, half, parameters)
    _clock_kinetic_drift(state, half, parameters)
    _bare_kinetic_drift(state, dt, parameters)
    _clock_kinetic_drift(state, half, parameters)
    _potential_kick(state, half, parameters)


def step_fixed_clock(
    state: LiveClockState,
    dt: float,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> None:
    """Advance bare GOV-01/GOV-02 with the candidate clock fixed to zero."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if np.max(np.abs(state.varphi)) > 0.0:
        raise ValueError("fixed-clock control requires varphi=0")
    half = 0.5 * dt
    q = np.ones_like(state.field)

    field_rate = (
        weighted_gradient_force(
            state.field,
            q,
            coefficient=parameters.matter_speed**2,
            stencil=parameters.gov01_stencil,
        )
        - state.chi**2 * state.field
    )
    chi_rate = weighted_gradient_force(
        state.chi,
        q,
        coefficient=parameters.chi_inertia * parameters.matter_speed**2,
        stencil=parameters.gov02_stencil,
    ) - (
        state.chi * state.field**2
        + 4.0
        * parameters.chi_inertia
        * parameters.lambda_h
        * state.chi
        * (state.chi**2 - parameters.chi0**2)
    )
    state.field_momentum += half * field_rate
    state.chi_momentum += half * chi_rate

    state.field += dt * state.field_momentum
    state.chi += dt * state.chi_momentum / parameters.chi_inertia

    field_rate = (
        weighted_gradient_force(
            state.field,
            q,
            coefficient=parameters.matter_speed**2,
            stencil=parameters.gov01_stencil,
        )
        - state.chi**2 * state.field
    )
    chi_rate = weighted_gradient_force(
        state.chi,
        q,
        coefficient=parameters.chi_inertia * parameters.matter_speed**2,
        stencil=parameters.gov02_stencil,
    ) - (
        state.chi * state.field**2
        + 4.0
        * parameters.chi_inertia
        * parameters.lambda_h
        * state.chi
        * (state.chi**2 - parameters.chi0**2)
    )
    state.field_momentum += half * field_rate
    state.chi_momentum += half * chi_rate


def make_traveling_packet(
    size: int,
    *,
    amplitude: float,
    width: float,
    carrier_index: int,
    parameters: LiveClockParameters = LiveClockParameters(),
) -> LiveClockState:
    """Construct an unsupported periodic scalar GOV-01 packet."""
    if size < 8:
        raise ValueError("size must be at least 8")
    if amplitude <= 0.0 or width <= 0.0 or carrier_index <= 0:
        raise ValueError("packet settings must be positive")
    coordinates = np.arange(size, dtype=np.float64) - size // 2
    x, y, z = np.meshgrid(
        coordinates,
        coordinates,
        coordinates,
        indexing="ij",
    )
    radius_sq = x**2 + y**2 + z**2
    wave_number = 2.0 * np.pi * carrier_index / size
    envelope = np.exp(-radius_sq / (2.0 * width**2))
    field = amplitude * envelope * np.cos(wave_number * x)

    frequencies = np.fft.fftfreq(size) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(
        frequencies,
        frequencies,
        frequencies,
        indexing="ij",
        sparse=True,
    )
    if parameters.gov01_stencil == "19":
        stiffness = -(
            (2.0 * np.cos(kx) - 2.0) / 3.0
            + (2.0 * np.cos(ky) - 2.0) / 3.0
            + (2.0 * np.cos(kz) - 2.0) / 3.0
            + (
                np.cos(kx + ky)
                + np.cos(kx - ky)
                + np.cos(kx + kz)
                + np.cos(kx - kz)
                + np.cos(ky + kz)
                + np.cos(ky - kz)
                - 6.0
            )
            / 3.0
        )
    else:
        stiffness = -(
            (8.0 / 9.0) * (np.cos(kx) + np.cos(ky) + np.cos(kz))
            + (4.0 / 9.0)
            * (np.cos(kx) * np.cos(ky) + np.cos(kx) * np.cos(kz) + np.cos(ky) * np.cos(kz))
            + (2.0 / 9.0) * np.cos(kx) * np.cos(ky) * np.cos(kz)
            - (38.0 / 9.0)
        )
    omega = np.sqrt(parameters.matter_speed**2 * np.maximum(stiffness, 0.0) + parameters.chi0**2)
    field_hat = np.fft.fftn(field)
    direction = np.sign(np.asarray(kx + np.zeros_like(ky) + np.zeros_like(kz)))
    momentum_hat = -1j * direction * omega * field_hat
    field_momentum = np.fft.ifftn(momentum_hat).real

    shape = (size, size, size)
    return LiveClockState(
        field=field,
        field_momentum=field_momentum,
        chi=np.full(shape, parameters.chi0, dtype=np.float64),
        chi_momentum=np.zeros(shape, dtype=np.float64),
        varphi=np.zeros(shape, dtype=np.float64),
        clock_momentum=np.zeros(shape, dtype=np.float64),
    )
