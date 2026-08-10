"""Experimental same-action temporal charge coupling for the C3 LFM field.

This module does not change canonical Simulation defaults. It implements a
local Hamiltonian candidate on the existing six real C3 coordinates and chi.
The update is implicit midpoint because the candidate is momentum dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfm.analysis.energy_current import (
    BareHamiltonRates,
    BareLFMParameters,
    BareLFMState,
    bare_hamilton_rates,
    bare_site_energy,
)


@dataclass(frozen=True)
class ChargeCouplingParameters:
    """Parameters for the experimental covariant temporal coupling."""

    bare: BareLFMParameters = BareLFMParameters(chi_potential="flat_octic")
    coupling: float = 0.0
    midpoint_tolerance: float = 1.0e-11
    midpoint_max_iterations: int = 20

    def __post_init__(self) -> None:
        if not np.isfinite(self.coupling):
            raise ValueError("coupling must be finite")
        if not np.isfinite(self.midpoint_tolerance) or self.midpoint_tolerance <= 0.0:
            raise ValueError("midpoint_tolerance must be positive and finite")
        if self.midpoint_max_iterations < 1:
            raise ValueError("midpoint_max_iterations must be positive")


@dataclass(frozen=True)
class ChargeCouplingStep:
    """One implicit-midpoint result and its convergence certificate."""

    state: BareLFMState
    iterations: int
    relative_residual: float


def _validated_charge_state(
    state: BareLFMState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wave = np.asarray(state.wave, dtype=np.float64)
    wave_p = np.asarray(state.wave_momentum, dtype=np.float64)
    chi = np.asarray(state.chi, dtype=np.float64)
    chi_p = np.asarray(state.chi_momentum, dtype=np.float64)
    if wave.ndim != 4 or wave.shape[0] < 2 or wave.shape[0] % 2 != 0:
        raise ValueError("wave must contain interleaved real/imaginary pairs")
    if wave_p.shape != wave.shape:
        raise ValueError("wave and wave_momentum shapes must match")
    if chi.ndim != 3 or chi_p.shape != chi.shape:
        raise ValueError("chi and chi_momentum must share a 3-D shape")
    if wave.shape[1:] != chi.shape:
        raise ValueError("wave and chi spatial shapes must match")
    return wave, wave_p, chi, chi_p


def charge_frequency(
    chi: np.ndarray,
    parameters: ChargeCouplingParameters,
) -> np.ndarray:
    """Return f(chi)=g_q(chi^2-chi0^2)/(2 chi0)."""

    chi_values = np.asarray(chi, dtype=np.float64)
    chi0 = parameters.bare.chi0
    return parameters.coupling * (chi_values**2 - chi0**2) / (2.0 * chi0)


def charge_frequency_derivative(
    chi: np.ndarray,
    parameters: ChargeCouplingParameters,
) -> np.ndarray:
    """Return df/dchi for the experimental coupling."""

    return parameters.coupling * np.asarray(chi, dtype=np.float64) / parameters.bare.chi0


def canonical_charge_density(state: BareLFMState) -> np.ndarray:
    """Return the global-phase Noether charge density in canonical variables."""

    wave, wave_p, _chi, _chi_p = _validated_charge_state(state)
    result = np.zeros(wave.shape[1:], dtype=np.float64)
    for component in range(0, wave.shape[0], 2):
        u = wave[component]
        v = wave[component + 1]
        p_u = wave_p[component]
        p_v = wave_p[component + 1]
        result += u * p_v - v * p_u
    return result


def total_canonical_charge(
    state: BareLFMState,
    parameters: ChargeCouplingParameters,
) -> float:
    """Return the physical-volume integral of canonical charge density."""

    return float(np.sum(canonical_charge_density(state))) * parameters.bare.spacing**3


def charge_coupled_rates(
    state: BareLFMState,
    parameters: ChargeCouplingParameters,
) -> BareHamiltonRates:
    """Return Hamilton's equations for H=H_bare-f(chi)Q."""

    wave, wave_p, chi, chi_p = _validated_charge_state(state)
    bare_rates = bare_hamilton_rates(
        wave,
        wave_p,
        chi,
        chi_p,
        parameters.bare,
    )
    wave_rate = np.asarray(bare_rates.wave, dtype=np.float64).copy()
    momentum_rate = np.asarray(bare_rates.wave_momentum, dtype=np.float64).copy()
    f_value = charge_frequency(chi, parameters)
    for component in range(0, wave.shape[0], 2):
        u = wave[component]
        v = wave[component + 1]
        p_u = wave_p[component]
        p_v = wave_p[component + 1]
        wave_rate[component] += f_value * v
        wave_rate[component + 1] -= f_value * u
        momentum_rate[component] += f_value * p_v
        momentum_rate[component + 1] -= f_value * p_u
    charge = canonical_charge_density(state)
    chi_momentum_rate = (
        np.asarray(bare_rates.chi_momentum, dtype=np.float64)
        + charge_frequency_derivative(chi, parameters) * charge
    )
    return BareHamiltonRates(
        wave=wave_rate,
        wave_momentum=momentum_rate,
        chi=np.asarray(bare_rates.chi, dtype=np.float64),
        chi_momentum=chi_momentum_rate,
    )


def charge_coupled_hamiltonian(
    state: BareLFMState,
    parameters: ChargeCouplingParameters,
) -> float:
    """Return the exact spatially discretized candidate Hamiltonian."""

    wave, wave_p, chi, chi_p = _validated_charge_state(state)
    density = bare_site_energy(
        wave,
        wave_p,
        chi,
        chi_p,
        parameters.bare,
    )
    density = density - charge_frequency(chi, parameters) * canonical_charge_density(state)
    return float(np.sum(density)) * parameters.bare.spacing**3


def _state_add_rates(
    state: BareLFMState,
    rates: BareHamiltonRates,
    factor: float,
) -> BareLFMState:
    wave, wave_p, chi, chi_p = _validated_charge_state(state)
    return BareLFMState(
        wave=wave + factor * rates.wave,
        wave_momentum=wave_p + factor * rates.wave_momentum,
        chi=chi + factor * rates.chi,
        chi_momentum=chi_p + factor * rates.chi_momentum,
    )


def _state_midpoint(left: BareLFMState, right: BareLFMState) -> BareLFMState:
    left_wave, left_p, left_chi, left_chi_p = _validated_charge_state(left)
    right_wave, right_p, right_chi, right_chi_p = _validated_charge_state(right)
    return BareLFMState(
        wave=0.5 * (left_wave + right_wave),
        wave_momentum=0.5 * (left_p + right_p),
        chi=0.5 * (left_chi + right_chi),
        chi_momentum=0.5 * (left_chi_p + right_chi_p),
    )


def _state_relative_difference(left: BareLFMState, right: BareLFMState) -> float:
    left_values = _validated_charge_state(left)
    right_values = _validated_charge_state(right)
    numerator = max(
        float(np.max(np.abs(a - b))) for a, b in zip(left_values, right_values, strict=False)
    )
    denominator = max(
        1.0,
        *(float(np.max(np.abs(value))) for value in right_values),
    )
    return numerator / denominator


def step_charge_coupled_lfm(
    state: BareLFMState,
    dt: float,
    parameters: ChargeCouplingParameters,
) -> ChargeCouplingStep:
    """Advance the local candidate with second-order implicit midpoint."""

    if not np.isfinite(dt) or dt == 0.0:
        raise ValueError("dt must be finite and nonzero")
    _validated_charge_state(state)
    guess = _state_add_rates(
        state,
        charge_coupled_rates(state, parameters),
        dt,
    )
    residual = float("inf")
    for iteration in range(1, parameters.midpoint_max_iterations + 1):
        midpoint = _state_midpoint(state, guess)
        candidate = _state_add_rates(
            state,
            charge_coupled_rates(midpoint, parameters),
            dt,
        )
        residual = _state_relative_difference(candidate, guess)
        guess = candidate
        if residual <= parameters.midpoint_tolerance:
            return ChargeCouplingStep(
                state=guess,
                iterations=iteration,
                relative_residual=residual,
            )
    raise RuntimeError(
        "implicit midpoint failed to converge: "
        f"residual={residual:.6e}, "
        f"iterations={parameters.midpoint_max_iterations}"
    )


__all__ = [
    "ChargeCouplingParameters",
    "ChargeCouplingStep",
    "canonical_charge_density",
    "charge_coupled_hamiltonian",
    "charge_coupled_rates",
    "charge_frequency",
    "charge_frequency_derivative",
    "step_charge_coupled_lfm",
    "total_canonical_charge",
]
