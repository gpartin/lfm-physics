"""Exact local energy continuity observables for bare GOV-01/GOV-02.

The conservative bare LFM Hamiltonian has a site energy obtained by assigning
half of every undirected stencil-link energy to each endpoint. Differentiating
that density with Hamilton's equations gives an exact oriented energy current.

This module adds no evolution register or force. It exposes observables already
fixed by the bare Hamiltonian for real, complex, and color-component fields.
The 19-point stencil is canonical; the 27-point option is an ablation label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfm.constants import C_DEFAULT, CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import laplacian_19pt, laplacian_27pt

Offset = tuple[int, int, int]
LinkCurrentMap = dict[Offset, np.ndarray]


def stencil_links(
    stencil: str,
    *,
    oriented: bool = True,
) -> tuple[tuple[Offset, float], ...]:
    """Return weighted cube links for a labeled cubic stencil.

    The unique list contains one representative of every undirected link.
    The oriented list appends its reverse with the same weight.
    """
    if stencil == "19":
        unique = (
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
    elif stencil == "27":
        unique = (
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
    else:
        raise ValueError("stencil must be '19' or '27'")
    if not oriented:
        return unique
    links: list[tuple[Offset, float]] = []
    for offset, weight in unique:
        links.append((offset, weight))
        links.append((tuple(-value for value in offset), weight))
    return tuple(links)


def _shift_scalar(values: np.ndarray, offset: Offset) -> np.ndarray:
    return np.roll(values, shift=offset, axis=(0, 1, 2))


def _shift_components(values: np.ndarray, offset: Offset) -> np.ndarray:
    return np.roll(values, shift=offset, axis=(1, 2, 3))


def _as_components(values: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(values)
    dtype = np.longdouble if source.dtype == np.dtype(np.longdouble) else np.float64
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 3:
        return array[np.newaxis, ...]
    if array.ndim == 4:
        return array
    raise ValueError(f"{name} must have shape (N,N,N) or (C,N,N,N)")


def _laplacian_scalar(values: np.ndarray, stencil: str) -> np.ndarray:
    if stencil == "19":
        return laplacian_19pt(values)
    if stencil == "27":
        return laplacian_27pt(values)
    raise ValueError("stencil must be '19' or '27'")


def _laplacian_components(values: np.ndarray, stencil: str) -> np.ndarray:
    return np.stack(
        [_laplacian_scalar(component, stencil) for component in values],
        axis=0,
    )


@dataclass(frozen=True)
class BareLFMParameters:
    """Parameters for the conservative bare GOV-01/GOV-02 Hamiltonian."""

    chi0: float = CHI0
    kappa: float = KAPPA
    lambda_h: float = LAMBDA_H
    wave_speed: float = C_DEFAULT
    background_norm_sq: float = 0.0
    gov01_stencil: str = "19"
    gov02_stencil: str = "19"

    @property
    def chi_inertia(self) -> float:
        """Return B=chi0/kappa fixed by the canonical chi source."""
        return self.chi0 / self.kappa

    def __post_init__(self) -> None:
        positive = (self.chi0, self.kappa, self.lambda_h, self.wave_speed)
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("bare LFM parameters must be positive and finite")
        if (
            not np.isfinite(self.background_norm_sq)
            or self.background_norm_sq < 0.0
        ):
            raise ValueError("background_norm_sq must be finite and nonnegative")
        stencil_links(self.gov01_stencil)
        stencil_links(self.gov02_stencil)


@dataclass(frozen=True)
class BareHamiltonRates:
    """Hamiltonian vector field for the bare LFM coordinate registers."""

    wave: np.ndarray
    wave_momentum: np.ndarray
    chi: np.ndarray
    chi_momentum: np.ndarray


def _validated_registers(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wave_components = _as_components(wave, "wave")
    momentum_components = _as_components(wave_momentum, "wave_momentum")
    chi_source = np.asarray(chi)
    chi_p_source = np.asarray(chi_momentum)
    chi_dtype = (
        np.longdouble
        if chi_source.dtype == np.dtype(np.longdouble)
        else np.float64
    )
    chi_p_dtype = (
        np.longdouble
        if chi_p_source.dtype == np.dtype(np.longdouble)
        else np.float64
    )
    chi_array = np.asarray(chi, dtype=chi_dtype)
    chi_momentum_array = np.asarray(chi_momentum, dtype=chi_p_dtype)
    if wave_components.shape != momentum_components.shape:
        raise ValueError("wave and wave_momentum shapes must match")
    if chi_array.ndim != 3 or chi_momentum_array.ndim != 3:
        raise ValueError("chi and chi_momentum must have shape (N,N,N)")
    if chi_array.shape != chi_momentum_array.shape:
        raise ValueError("chi and chi_momentum shapes must match")
    if wave_components.shape[1:] != chi_array.shape:
        raise ValueError("wave and chi spatial shapes must match")
    return (
        wave_components,
        momentum_components,
        chi_array,
        chi_momentum_array,
    )


def _gradient_site_density(
    values: np.ndarray,
    *,
    coefficient: float,
    stencil: str,
    components: bool,
) -> np.ndarray:
    spatial_shape = values.shape[1:] if components else values.shape
    density = np.zeros(
        spatial_shape,
        dtype=np.result_type(values.dtype, np.float64),
    )
    shift = _shift_components if components else _shift_scalar
    for offset, weight in stencil_links(stencil):
        difference = shift(values, offset) - values
        squared = (
            np.sum(difference**2, axis=0)
            if components
            else difference**2
        )
        density += 0.25 * coefficient * weight * squared
    return density


def bare_site_energy(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters = BareLFMParameters(),
) -> np.ndarray:
    """Return exact endpoint-split site energy for the bare Hamiltonian."""
    wave_values, wave_p, chi_values, chi_p = _validated_registers(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
    )
    norm_sq = np.sum(wave_values**2, axis=0)
    onsite = (
        0.5 * np.sum(wave_p**2, axis=0)
        + chi_p**2 / (2.0 * parameters.chi_inertia)
        + 0.5
        * chi_values**2
        * (norm_sq - parameters.background_norm_sq)
        + parameters.chi_inertia
        * parameters.lambda_h
        * (chi_values**2 - parameters.chi0**2) ** 2
    )
    wave_gradient = _gradient_site_density(
        wave_values,
        coefficient=parameters.wave_speed**2,
        stencil=parameters.gov01_stencil,
        components=True,
    )
    chi_gradient = _gradient_site_density(
        chi_values,
        coefficient=parameters.chi_inertia * parameters.wave_speed**2,
        stencil=parameters.gov02_stencil,
        components=False,
    )
    return onsite + wave_gradient + chi_gradient


def bare_hamilton_rates(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters = BareLFMParameters(),
) -> BareHamiltonRates:
    """Return Hamilton's equations for the conservative bare system."""
    wave_values, wave_p, chi_values, chi_p = _validated_registers(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
    )
    norm_sq = np.sum(wave_values**2, axis=0)
    wave_rate = wave_p
    wave_momentum_rate = (
        parameters.wave_speed**2
        * _laplacian_components(wave_values, parameters.gov01_stencil)
        - chi_values[np.newaxis, ...] ** 2 * wave_values
    )
    chi_rate = chi_p / parameters.chi_inertia
    chi_momentum_rate = (
        parameters.chi_inertia
        * parameters.wave_speed**2
        * _laplacian_scalar(chi_values, parameters.gov02_stencil)
        - chi_values * (norm_sq - parameters.background_norm_sq)
        - 4.0
        * parameters.chi_inertia
        * parameters.lambda_h
        * chi_values
        * (chi_values**2 - parameters.chi0**2)
    )
    return BareHamiltonRates(
        wave=wave_rate,
        wave_momentum=wave_momentum_rate,
        chi=chi_rate,
        chi_momentum=chi_momentum_rate,
    )


def _gradient_site_rate(
    values: np.ndarray,
    rates: np.ndarray,
    *,
    coefficient: float,
    stencil: str,
    components: bool,
) -> np.ndarray:
    spatial_shape = values.shape[1:] if components else values.shape
    result = np.zeros(
        spatial_shape,
        dtype=np.result_type(values.dtype, rates.dtype, np.float64),
    )
    shift = _shift_components if components else _shift_scalar
    for offset, weight in stencil_links(stencil):
        difference = shift(values, offset) - values
        rate_difference = shift(rates, offset) - rates
        product = (
            np.sum(difference * rate_difference, axis=0)
            if components
            else difference * rate_difference
        )
        result += 0.5 * coefficient * weight * product
    return result


def bare_site_energy_rate(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters = BareLFMParameters(),
) -> np.ndarray:
    """Differentiate endpoint-split site energy along Hamilton's equations."""
    wave_values, wave_p, chi_values, chi_p = _validated_registers(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
    )
    rates = bare_hamilton_rates(
        wave_values,
        wave_p,
        chi_values,
        chi_p,
        parameters,
    )
    norm_sq = np.sum(wave_values**2, axis=0)
    onsite_rate = (
        np.sum(wave_p * rates.wave_momentum, axis=0)
        + (chi_p / parameters.chi_inertia) * rates.chi_momentum
        + chi_values
        * rates.chi
        * (norm_sq - parameters.background_norm_sq)
        + chi_values**2 * np.sum(wave_values * rates.wave, axis=0)
        + 4.0
        * parameters.chi_inertia
        * parameters.lambda_h
        * chi_values
        * (chi_values**2 - parameters.chi0**2)
        * rates.chi
    )
    wave_gradient_rate = _gradient_site_rate(
        wave_values,
        rates.wave,
        coefficient=parameters.wave_speed**2,
        stencil=parameters.gov01_stencil,
        components=True,
    )
    chi_gradient_rate = _gradient_site_rate(
        chi_values,
        rates.chi,
        coefficient=parameters.chi_inertia * parameters.wave_speed**2,
        stencil=parameters.gov02_stencil,
        components=False,
    )
    return onsite_rate + wave_gradient_rate + chi_gradient_rate


def oriented_energy_currents(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters = BareLFMParameters(),
) -> LinkCurrentMap:
    """Return total outgoing energy current on every declared oriented link."""
    wave_values, wave_p, chi_values, chi_p = _validated_registers(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
    )
    currents: LinkCurrentMap = {}
    wave_coefficient = -0.5 * parameters.wave_speed**2
    for offset, weight in stencil_links(parameters.gov01_stencil):
        difference = _shift_components(wave_values, offset) - wave_values
        endpoint_rate_sum = (
            _shift_components(wave_p, offset) + wave_p
        )
        currents[offset] = (
            wave_coefficient
            * weight
            * np.sum(difference * endpoint_rate_sum, axis=0)
        )

    chi_rate = chi_p / parameters.chi_inertia
    chi_coefficient = (
        -0.5 * parameters.chi_inertia * parameters.wave_speed**2
    )
    for offset, weight in stencil_links(parameters.gov02_stencil):
        difference = _shift_scalar(chi_values, offset) - chi_values
        endpoint_rate_sum = _shift_scalar(chi_rate, offset) + chi_rate
        contribution = (
            chi_coefficient * weight * difference * endpoint_rate_sum
        )
        if offset in currents:
            currents[offset] = currents[offset] + contribution
        else:
            currents[offset] = contribution
    return currents


def energy_current_divergence(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters = BareLFMParameters(),
) -> np.ndarray:
    """Return the sum of all outgoing oriented currents at each cube."""
    currents = oriented_energy_currents(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
        parameters,
    )
    dtype = np.result_type(
        *(current.dtype for current in currents.values()),
        np.float64,
    )
    result = np.zeros(np.asarray(chi).shape, dtype=dtype)
    for current in currents.values():
        result += current
    return result


def bare_energy_continuity_residual(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters = BareLFMParameters(),
) -> np.ndarray:
    """Return the exact lattice residual dh/dt + sum_j J(i->j)."""
    return bare_site_energy_rate(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
        parameters,
    ) + energy_current_divergence(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
        parameters,
    )
