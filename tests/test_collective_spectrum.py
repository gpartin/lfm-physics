"""Tests for bare-LFM collective-mode diagnostics."""

from __future__ import annotations

import numpy as np

from lfm.analysis.collective_spectrum import (
    berry_action_derivative_audit,
    collective_mode_eigenpairs,
    collective_qep_matrices,
    discrete_stiffness_19,
    gov02_background_residual,
    mode_polarization,
    principal_symbol_audit,
    rotating_background,
    vacuum_spectrum_audit,
    zero_chi_branch_audit,
)
from lfm.constants import CHI0, KAPPA


def test_principal_symbol_has_no_vacuum_vector_or_tensor_mode() -> None:
    audit = principal_symbol_audit()
    assert audit["real_field_count"] == 7
    assert audit["background_dependent"] is False
    assert audit["trivial_vacuum_gapless_spin_1_count"] == 0
    assert audit["trivial_vacuum_gapless_spin_2_count"] == 0


def test_flat_octic_makes_only_the_scalar_chi_mode_gapless() -> None:
    quartic = vacuum_spectrum_audit("canonical_quartic")
    octic = vacuum_spectrum_audit("flat_octic")
    assert quartic["gapless_spin_0_count"] == 0
    assert octic["gapless_spin_0_count"] == 1
    assert octic["gapless_spin_1_count"] == 0
    assert octic["gapless_spin_2_count"] == 0


def test_zero_chi_branch_has_massless_scalars_but_no_linear_gravity_coupling() -> None:
    critical = zero_chi_branch_audit(0.0, "flat_octic")["stability_threshold_density"]
    below = zero_chi_branch_audit(0.5 * critical, "flat_octic")
    at = zero_chi_branch_audit(critical, "flat_octic")
    above = zero_chi_branch_audit(1.5 * critical, "flat_octic")
    assert below["stability"] == "TACHYONIC"
    assert at["stability"] == "CRITICAL_CHI_GAPLESS_NONLINEARLY_RESTORED"
    assert above["stability"] == "LINEARLY_STABLE_CHI_GAPPED"
    assert at["gov01_massless_real_scalar_count"] == 6
    assert at["bare_gauss_constraint"] is False
    assert at["linear_matter_to_chi_source_coefficient"] == 0.0
    assert at["linear_chi_to_matter_response_coefficient"] == 0.0


def test_background_equilibrium_residuals() -> None:
    q = np.diag([0.7, 0.7, 0.7])
    for model in ("canonical_quartic", "flat_octic"):
        background = rotating_background(100.0, q, model=model, spacing=0.1)
        assert abs(gov02_background_residual(background)) < 1.0e-10


def test_q_zero_even_sideband_is_exact_discrete_stiffness() -> None:
    background = rotating_background(10.0, model="canonical_quartic", spacing=0.2)
    wave_vector = np.array([0.3, 0.2, 0.1])
    _, k_matrix = collective_qep_matrices(background, wave_vector)
    expected = discrete_stiffness_19(wave_vector, spacing=0.2)
    assert np.allclose(np.diag(k_matrix)[:6], expected)


def test_relative_phase_branch_is_quadratic_at_small_k() -> None:
    background = rotating_background(1.0, model="canonical_quartic", spacing=0.05)
    frequencies = []
    for wave_number in (0.02, 0.04, 0.08):
        values, _ = collective_mode_eigenpairs(background, np.array([wave_number, 0.0, 0.0]))
        expected = (
            np.sqrt(
                background.carrier_frequencies[0] ** 2
                + discrete_stiffness_19(np.array([wave_number, 0.0, 0.0]), spacing=0.05)
            )
            - background.carrier_frequencies[0]
        )
        stable_positive = [
            value.real for value in values if value.real > 1.0e-10 and abs(value.imag) < 1.0e-8
        ]
        observed = min(stable_positive, key=lambda value: abs(value - expected))
        assert np.isclose(observed, expected, rtol=1.0e-7, atol=1.0e-12)
        frequencies.append(observed)
    slope = np.polyfit(np.log([0.02, 0.04, 0.08]), np.log(frequencies), 1)[0]
    assert 1.9 < slope < 2.1


def test_phase_displacement_strain_has_zero_tt_projection() -> None:
    state = np.zeros(7, dtype=np.complex128)
    state[3:6] = np.array([1.0 + 0.2j, -0.3j, 0.7])
    result = mode_polarization(state, np.array([0.4, 0.2, -0.1]))
    assert result["phase_strain_tt_fraction"] < 1.0e-28


def test_bare_action_does_not_contain_composite_maxwell_term() -> None:
    audit = berry_action_derivative_audit()
    assert audit["bare_action_contains_independent_f_squared"] is False
    assert audit["inhomogeneous_maxwell_equation_is_bare_euler_lagrange_equation"] is False


def test_common_condensate_instability_matches_small_k_derivation() -> None:
    background = rotating_background(100.0, model="canonical_quartic", spacing=0.05)
    attraction = (
        4.0 * (KAPPA / CHI0) * background.chi**2 * background.total_density / background.chi_mass_sq
    )
    predicted_growth_per_k = np.sqrt(attraction) / (2.0 * background.carrier_frequencies[0])
    wave_number = 0.002
    values, _ = collective_mode_eigenpairs(background, np.array([wave_number, 0.0, 0.0]))
    observed_growth_per_k = max(abs(value.imag) for value in values) / wave_number
    assert np.isclose(observed_growth_per_k, predicted_growth_per_k, rtol=2.0e-4)


def test_instability_growth_converges_with_spacing() -> None:
    wave_number = 0.01
    growth_rates = []
    for spacing in (0.1, 0.05, 0.025):
        background = rotating_background(100.0, model="flat_octic", spacing=spacing)
        values, _ = collective_mode_eigenpairs(background, np.array([wave_number, 0.0, 0.0]))
        growth_rates.append(max(abs(value.imag) for value in values))
    assert np.ptp(growth_rates) / np.mean(growth_rates) < 1.0e-6
