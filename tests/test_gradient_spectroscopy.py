from __future__ import annotations

import numpy as np

from lfm.analysis.gradient_spectroscopy import (
    cross_validated_mode,
    expected_massless_ir_ratio,
    extract_low_momentum_modes,
    native_gradient_operators,
    operator_completeness_audit,
)
from lfm.constants import CHI0
from lfm.experiment.euclidean_r2 import EuclideanR2Config, EuclideanR2Sampler


def test_u3_basis_is_complete() -> None:
    result = operator_completeness_audit()
    assert result["pass"]


def test_zero_field_has_zero_gradient_operators() -> None:
    psi = np.zeros((4, 4, 4, 4, 3), dtype=np.complex128)
    chi = np.full((4, 4, 4, 4), CHI0)
    operators = native_gradient_operators(psi, chi)
    assert operators.shape == (4, 4, 4, 4, 19, 4)
    assert np.max(np.abs(operators)) == 0.0


def test_mode_shapes_and_expected_ratio() -> None:
    rng = np.random.default_rng(2)
    operators = rng.normal(size=(4, 4, 4, 4, 19, 4))
    modes = extract_low_momentum_modes(operators)
    assert modes["transverse_k1"].shape == (6, 19)
    assert modes["longitudinal_k1"].shape == (3, 19)
    assert np.isclose(expected_massless_ir_ratio(4), 2.0)


def test_euclidean_sampler_changes_state_locally() -> None:
    sampler = EuclideanR2Sampler(
        EuclideanR2Config(linear_size=4, model="canonical_quartic", seed=3)
    )
    result = sampler.sweep()
    assert result["psi_total"] == 4**4
    assert result["phi_total"] == 4**4
    assert np.any(np.abs(sampler.psi) > 0.0)


def _complex_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    variance: float,
) -> np.ndarray:
    return np.sqrt(0.5 * variance) * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def test_cross_validated_spectroscopy_recovers_synthetic_massless_channel() -> None:
    rng = np.random.default_rng(12)
    samples = 1200
    channels = 19
    size = 8
    expected = expected_massless_ir_ratio(size)
    modes = {
        "transverse_k1": _complex_noise(rng, (samples, 6, channels), 1.0),
        "transverse_k2": _complex_noise(rng, (samples, 6, channels), 1.0),
        "longitudinal_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "polarization_0_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "polarization_1_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "cone_p1_k1": _complex_noise(rng, (samples, 6, channels), 1.0),
        "temporal_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "temporal_k2": _complex_noise(rng, (samples, 3, channels), 1.0),
    }
    modes["transverse_k1"][..., 0] = _complex_noise(rng, (samples, 6), expected)
    modes["transverse_k2"][..., 0] = _complex_noise(rng, (samples, 6), 1.0)
    modes["longitudinal_k1"][..., 0] = _complex_noise(rng, (samples, 3), 0.01 * expected)
    modes["polarization_0_k1"][..., 0] = _complex_noise(rng, (samples, 3), expected)
    modes["polarization_1_k1"][..., 0] = _complex_noise(rng, (samples, 3), expected)
    modes["cone_p1_k1"][..., 0] = _complex_noise(rng, (samples, 6), 0.5 * expected)
    result = cross_validated_mode(modes, "raw_u3", size)
    assert max(abs(value / expected - 1.0) for value in result.heldout_ir_ratios) < 0.12
    assert max(abs(value / 0.5 - 1.0) for value in result.heldout_cone_ratios) < 0.12
    assert max(result.heldout_longitudinal_ratios) < 0.03


def test_cross_validated_spectroscopy_rejects_white_vector_noise() -> None:
    rng = np.random.default_rng(21)
    samples = 1200
    channels = 19
    modes = {
        "transverse_k1": _complex_noise(rng, (samples, 6, channels), 1.0),
        "transverse_k2": _complex_noise(rng, (samples, 6, channels), 1.0),
        "longitudinal_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "polarization_0_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "polarization_1_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "cone_p1_k1": _complex_noise(rng, (samples, 6, channels), 1.0),
        "temporal_k1": _complex_noise(rng, (samples, 3, channels), 1.0),
        "temporal_k2": _complex_noise(rng, (samples, 3, channels), 1.0),
    }
    result = cross_validated_mode(modes, "raw_u3", 8)
    assert max(result.heldout_ir_ratios) < 1.25
