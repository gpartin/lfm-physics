"""Tests for the unpromoted R3 link-frame action prototype."""

import numpy as np
import pytest

from lfm.foundations.r3_link_frame import (
    R3LinkFrameParameters,
    R3ProductLink,
    chiral_frame_curvature,
    frame_shape_acceleration,
    internal_covariant_difference,
    product_plaquette_energy,
    r3_action_declaration,
    r3_action_fingerprint,
    transform_internal_matter,
    transform_product_link,
)


def _rotation(size: int, left: int, right: int, angle: float) -> np.ndarray:
    matrix = np.eye(size)
    matrix[left, left] = np.cos(angle)
    matrix[right, right] = np.cos(angle)
    matrix[left, right] = -np.sin(angle)
    matrix[right, left] = np.sin(angle)
    return matrix


def _su3_rotation(left: int, right: int, angle: float) -> np.ndarray:
    return _rotation(3, left, right, angle).astype(np.complex128)


def test_internal_covariant_difference_is_locally_covariant() -> None:
    link = R3ProductLink(
        frame=_rotation(4, 0, 1, 0.13),
        phase=np.exp(0.21j),
        color=_su3_rotation(0, 1, 0.17),
    )
    matter_i = np.asarray([1.0 + 0.2j, -0.3j, 0.5])
    matter_j = np.asarray([0.2, 0.7 - 0.1j, -0.4j])
    phase_i = np.exp(0.31j)
    phase_j = np.exp(-0.27j)
    color_i = _su3_rotation(1, 2, 0.23)
    color_j = _su3_rotation(0, 2, -0.19)
    frame_i = _rotation(4, 1, 2, 0.11)
    frame_j = _rotation(4, 2, 3, -0.09)
    difference = internal_covariant_difference(matter_i, matter_j, link)
    transformed_link = transform_product_link(
        link,
        frame_i=frame_i,
        frame_j=frame_j,
        phase_i=phase_i,
        phase_j=phase_j,
        color_i=color_i,
        color_j=color_j,
    )
    transformed = internal_covariant_difference(
        transform_internal_matter(
            matter_i,
            phase=phase_i,
            color=color_i,
        ),
        transform_internal_matter(
            matter_j,
            phase=phase_j,
            color=color_j,
        ),
        transformed_link,
    )
    expected = transform_internal_matter(
        difference,
        phase=phase_i,
        color=color_i,
    )
    assert np.allclose(transformed, expected, atol=1.0e-12)


def test_reverse_link_is_exact_inverse() -> None:
    link = R3ProductLink(
        frame=_rotation(4, 0, 3, 0.4),
        phase=np.exp(0.5j),
        color=_su3_rotation(1, 2, -0.3),
    )
    identity = link.compose(link.reverse())
    assert np.allclose(identity.frame, np.eye(4), atol=1.0e-12)
    assert identity.phase == pytest.approx(1.0 + 0.0j)
    assert np.allclose(identity.color, np.eye(3), atol=1.0e-12)


def test_product_plaquette_energy_is_positive_and_zero_at_identity() -> None:
    parameters = R3LinkFrameParameters()
    zero = product_plaquette_energy(R3ProductLink.identity(), parameters)
    assert zero["total"] == pytest.approx(0.0, abs=1.0e-14)
    holonomy = R3ProductLink(
        frame=_rotation(4, 0, 1, 0.14),
        phase=np.exp(0.18j),
        color=_su3_rotation(0, 2, 0.16),
    )
    energy = product_plaquette_energy(holonomy, parameters)
    assert energy["frame_even"] >= 0.0
    assert energy["frame_chiral"] > 0.0
    assert energy["phase"] > 0.0
    assert energy["color"] > 0.0
    assert energy["total"] > 0.0


def test_chiral_frame_weights_remain_positive() -> None:
    holonomy = _rotation(4, 0, 1, 0.2) @ _rotation(4, 2, 3, 0.1)
    plus, minus, omega = chiral_frame_curvature(holonomy)
    assert plus.shape == minus.shape == (3,)
    assert np.allclose(omega, -omega.T)
    for epsilon in (-0.9, 0.0, 0.1, 0.9):
        energy = product_plaquette_energy(
            R3ProductLink(
                frame=holonomy,
                phase=1.0,
                color=np.eye(3),
            ),
            R3LinkFrameParameters(epsilon_w=epsilon),
        )
        assert energy["frame_chiral"] >= 0.0


def test_frame_source_preserves_traceless_shape_sector() -> None:
    shape = np.zeros((2, 2, 2, 10))
    laplacian = np.zeros_like(shape)
    density = np.ones((2, 2, 2))
    acceleration = frame_shape_acceleration(laplacian, density, shape)
    trace = np.sum(acceleration[..., :4], axis=-1)
    assert np.max(np.abs(trace)) <= 1.0e-14
    assert np.all(acceleration[..., 0] < 0.0)
    assert np.all(acceleration[..., 1:4] > 0.0)


def test_action_declaration_is_stable_and_retains_radial_chi() -> None:
    declaration = r3_action_declaration()
    assert declaration["canonical_status"] == ("UNPROMOTED_FOUNDATIONAL_CANDIDATE")
    assert "full_mexican_hat" in declaration["retained_sectors"]
    assert len(r3_action_fingerprint()) == 64
    assert r3_action_fingerprint() == r3_action_fingerprint()
