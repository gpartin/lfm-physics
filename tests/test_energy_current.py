"""Tests for exact bare-LFM lattice energy continuity."""

from __future__ import annotations

import numpy as np
import pytest

from lfm.analysis.energy_current import (
    BareLFMParameters,
    BareLFMState,
    bare_energy_continuity_residual,
    bare_hamilton_rates,
    bare_site_energy,
    bare_site_energy_rate,
    bare_total_energy,
    energy_current_divergence,
    oriented_energy_currents,
    step_bare_lfm,
    wave_component_site_energy,
)
from lfm.core.stencils import laplacian_19pt, laplacian_27pt


def _state(components: int, size: int = 6) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260724 + components)
    shape = (components, size, size, size)
    wave = 0.02 * rng.normal(size=shape)
    wave_momentum = 0.03 * rng.normal(size=shape)
    chi = 19.0 + 0.001 * rng.normal(size=shape[1:])
    chi_momentum = 0.02 * rng.normal(size=shape[1:])
    return wave, wave_momentum, chi, chi_momentum


@pytest.mark.parametrize(
    ("gov01_stencil", "gov02_stencil"),
    (("19", "19"), ("27", "19"), ("19", "27"), ("27", "27")),
)
@pytest.mark.parametrize("components", (1, 2, 6))
def test_exact_site_continuity(
    gov01_stencil: str,
    gov02_stencil: str,
    components: int,
) -> None:
    state = _state(components)
    parameters = BareLFMParameters(
        gov01_stencil=gov01_stencil,
        gov02_stencil=gov02_stencil,
    )
    residual = bare_energy_continuity_residual(*state, parameters)
    energy_rate = bare_site_energy_rate(*state, parameters)
    divergence = energy_current_divergence(*state, parameters)
    scale = max(
        float(np.max(np.abs(energy_rate))),
        float(np.max(np.abs(divergence))),
        1.0,
    )
    assert float(np.max(np.abs(residual))) / scale < 1.0e-12
    assert (
        abs(float(np.sum(energy_rate)))
        / max(
            float(np.sum(np.abs(energy_rate))),
            1.0,
        )
        < 1.0e-12
    )


def test_oriented_currents_are_antisymmetric() -> None:
    state = _state(6)
    parameters = BareLFMParameters(
        gov01_stencil="27",
        gov02_stencil="19",
    )
    currents = oriented_energy_currents(*state, parameters)
    for offset, current in currents.items():
        reverse = tuple(-value for value in offset)
        aligned_reverse = np.roll(
            currents[reverse],
            shift=offset,
            axis=(0, 1, 2),
        )
        scale = max(float(np.max(np.abs(current))), 1.0)
        assert float(np.max(np.abs(current + aligned_reverse))) / scale < 1.0e-12


def test_site_energy_matches_laplacian_quadratic_form() -> None:
    wave, wave_p, chi, chi_p = _state(6)
    for gov01_stencil, gov02_stencil in (
        ("19", "19"),
        ("27", "19"),
        ("19", "27"),
        ("27", "27"),
    ):
        parameters = BareLFMParameters(
            gov01_stencil=gov01_stencil,
            gov02_stencil=gov02_stencil,
        )
        lap01 = laplacian_19pt if gov01_stencil == "19" else laplacian_27pt
        lap02 = laplacian_19pt if gov02_stencil == "19" else laplacian_27pt
        norm_sq = np.sum(wave**2, axis=0)
        onsite = (
            0.5 * np.sum(wave_p**2)
            + np.sum(chi_p**2) / (2.0 * parameters.chi_inertia)
            + 0.5 * np.sum(chi**2 * norm_sq)
            + parameters.chi_inertia
            * parameters.lambda_h
            * np.sum((chi**2 - parameters.chi0**2) ** 2)
        )
        wave_gradient = (
            -0.5
            * parameters.wave_speed**2
            * sum(np.sum(component * lap01(component)) for component in wave)
        )
        chi_displacement = chi - parameters.chi0
        chi_gradient = (
            -0.5
            * parameters.chi_inertia
            * parameters.wave_speed**2
            * np.sum(chi_displacement * lap02(chi_displacement))
        )
        direct = float(onsite + wave_gradient + chi_gradient)
        site_sum = float(
            np.sum(
                bare_site_energy(
                    wave,
                    wave_p,
                    chi,
                    chi_p,
                    parameters,
                )
            )
        )
        assert abs(site_sum - direct) / max(abs(direct), 1.0) < 1.0e-12


def test_analytic_site_rate_matches_centered_difference() -> None:
    state = _state(2)
    parameters = BareLFMParameters(
        background_norm_sq=0.01,
        gov01_stencil="19",
        gov02_stencil="27",
    )
    rates = bare_hamilton_rates(*state, parameters)
    # A larger displacement is used for this float64 unit smoke test.
    # The decision gate separately retains 1e-7 with Decimal arithmetic.
    epsilon = 1.0e-4
    plus = tuple(
        value + epsilon * rate
        for value, rate in zip(
            state,
            (
                rates.wave,
                rates.wave_momentum,
                rates.chi,
                rates.chi_momentum,
            ),
            strict=True,
        )
    )
    minus = tuple(
        value - epsilon * rate
        for value, rate in zip(
            state,
            (
                rates.wave,
                rates.wave_momentum,
                rates.chi,
                rates.chi_momentum,
            ),
            strict=True,
        )
    )
    numerical = (bare_site_energy(*plus, parameters) - bare_site_energy(*minus, parameters)) / (
        2.0 * epsilon
    )
    analytic = bare_site_energy_rate(*state, parameters)
    scale = max(float(np.max(np.abs(analytic))), 1.0)
    assert float(np.max(np.abs(numerical - analytic))) / scale < 2.0e-7


def test_internal_basis_rotation_invariance() -> None:
    wave, wave_p, chi, chi_p = _state(6)
    parameters = BareLFMParameters()
    rng = np.random.default_rng(1197)
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    rotated_wave = np.einsum("ab,bxyz->axyz", rotation, wave)
    rotated_p = np.einsum("ab,bxyz->axyz", rotation, wave_p)
    original = (
        bare_site_energy(wave, wave_p, chi, chi_p, parameters),
        bare_site_energy_rate(wave, wave_p, chi, chi_p, parameters),
        energy_current_divergence(wave, wave_p, chi, chi_p, parameters),
    )
    rotated = (
        bare_site_energy(rotated_wave, rotated_p, chi, chi_p, parameters),
        bare_site_energy_rate(
            rotated_wave,
            rotated_p,
            chi,
            chi_p,
            parameters,
        ),
        energy_current_divergence(
            rotated_wave,
            rotated_p,
            chi,
            chi_p,
            parameters,
        ),
    )
    for before, after in zip(original, rotated, strict=True):
        scale = max(float(np.max(np.abs(before))), 1.0)
        assert float(np.max(np.abs(after - before))) / scale < 1.0e-12


@pytest.mark.parametrize("chi_potential", ("quartic", "flat_octic"))
@pytest.mark.parametrize("spacing", (1.0, 0.125))
def test_continuity_with_physical_spacing_and_potential(
    chi_potential: str,
    spacing: float,
) -> None:
    state = _state(6)
    parameters = BareLFMParameters(
        spacing=spacing,
        chi_potential=chi_potential,
        gov01_stencil="27",
        gov02_stencil="19",
    )
    residual = bare_energy_continuity_residual(*state, parameters)
    rate = bare_site_energy_rate(*state, parameters)
    scale = max(float(np.max(np.abs(rate))), 1.0)
    assert float(np.max(np.abs(residual))) / scale < 3.0e-12


@pytest.mark.parametrize("chi_potential", ("quartic", "flat_octic"))
def test_velocity_verlet_preserves_bare_energy_to_second_order(
    chi_potential: str,
) -> None:
    raw = _state(6, size=5)
    parameters = BareLFMParameters(
        spacing=0.2,
        chi_potential=chi_potential,
        gov01_stencil="19",
        gov02_stencil="27",
    )
    drifts = []
    final_state = None
    for dt, steps in ((0.001, 40), (0.0005, 80)):
        state = BareLFMState(*(value.copy() for value in raw))
        initial = bare_total_energy(state, parameters)
        for _ in range(steps):
            state = step_bare_lfm(state, dt, parameters)
        final = bare_total_energy(state, parameters)
        drifts.append(abs(final - initial) / max(abs(initial), 1.0))
        final_state = state
    assert drifts[0] < 1.0e-4
    assert drifts[1] < 0.4 * drifts[0]
    assert final_state is not None
    component_energy = wave_component_site_energy(final_state, 4, parameters)
    assert component_energy.shape == raw[2].shape
    assert float(np.min(component_energy)) >= 0.0
