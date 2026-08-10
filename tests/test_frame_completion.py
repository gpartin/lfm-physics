"""Tests for the unpromoted spacetime cube-frame candidate algebra."""

import numpy as np
import pytest

from lfm.analysis.frame_completion import (
    FRAME_SHAPE_COUNT,
    analytic_rest_energy_response,
    frame_projectors,
    frame_static_operator,
    frame_static_response,
    minimized_source_cross_energy,
    source_projection_weights,
    zero_momentum_frame_spectrum,
)


def test_frame_projectors_are_orthogonal_and_complete() -> None:
    scale, shape = frame_projectors()
    identity = np.eye(10)
    assert np.allclose(scale @ scale, scale)
    assert np.allclose(shape @ shape, shape)
    assert np.allclose(scale @ shape, 0.0)
    assert np.allclose(scale + shape, identity)
    assert np.isclose(np.trace(scale), 1.0)
    assert np.isclose(np.trace(shape), FRAME_SHAPE_COUNT)


def test_rest_energy_has_fixed_scale_and_shape_weights() -> None:
    weights = source_projection_weights()
    assert weights["scale"] == pytest.approx(0.25)
    assert weights["shape"] == pytest.approx(0.75)


def test_zero_momentum_spectrum_has_nine_shape_zeros() -> None:
    spectrum = zero_momentum_frame_spectrum(radial_mass_sq=12.5)
    assert np.count_nonzero(np.abs(spectrum) <= 1.0e-12) == 9
    assert spectrum[-1] == pytest.approx(12.5)


@pytest.mark.parametrize("normalization", [1.0, 19.0, 63.0, 1197.0])
def test_matrix_and_analytic_responses_agree(normalization: float) -> None:
    measured = frame_static_response(
        0.17,
        radial_mass_sq=372.0,
        normalization=normalization,
    )
    expected = analytic_rest_energy_response(
        0.17,
        radial_mass_sq=372.0,
        normalization=normalization,
    )
    assert measured == pytest.approx(expected, rel=1.0e-13)


def test_positive_operator_and_attractive_cross_energy() -> None:
    operator = frame_static_operator(
        0.2,
        radial_mass_sq=372.0,
        normalization=1197.0,
    )
    assert np.min(np.linalg.eigvalsh(operator)) > 0.0
    assert (
        minimized_source_cross_energy(
            0.2,
            2.0,
            radial_mass_sq=372.0,
            normalization=1197.0,
        )
        < 0.0
    )
