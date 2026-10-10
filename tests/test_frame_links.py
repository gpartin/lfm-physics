"""Tests for local frame-comparison provenance algebra."""

import numpy as np
import pytest

from lfm.analysis.frame_links import (
    linked_frame_difference,
    loop_holonomy,
    loop_mismatch_energy,
    reconstructed_frame_link,
)


def _frames() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_0 = np.eye(4)
    frame_1 = np.diag([1.1, 0.9, 1.2, 0.8])
    frame_2 = np.asarray(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.1, 0.1, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.1, 0.0, 0.0, 1.2],
        ]
    )
    return frame_0, frame_1, frame_2


def test_reconstructed_frame_links_are_flat() -> None:
    frame_0, frame_1, frame_2 = _frames()
    link_01 = reconstructed_frame_link(frame_0, frame_1)
    link_12 = reconstructed_frame_link(frame_1, frame_2)
    link_20 = reconstructed_frame_link(frame_2, frame_0)
    holonomy = loop_holonomy(link_01, link_12, link_20)
    assert np.allclose(holonomy, np.eye(4), atol=1.0e-12)
    assert loop_mismatch_energy(holonomy) == pytest.approx(
        0.0,
        abs=1.0e-24,
    )


def test_linked_difference_transports_site_values() -> None:
    frame_0, frame_1, _ = _frames()
    global_value = np.asarray([1.0, 2.0, 3.0, 4.0])
    value_0 = np.linalg.solve(frame_0, global_value)
    value_1 = np.linalg.solve(frame_1, global_value)
    link_01 = reconstructed_frame_link(frame_0, frame_1)
    assert np.allclose(
        linked_frame_difference(value_0, value_1, link_01),
        0.0,
        atol=1.0e-12,
    )


def test_independent_link_can_have_positive_loop_mismatch() -> None:
    frame_0, frame_1, frame_2 = _frames()
    link_01 = reconstructed_frame_link(frame_0, frame_1)
    link_12 = reconstructed_frame_link(frame_1, frame_2)
    link_20 = reconstructed_frame_link(frame_2, frame_0)
    independent = link_01.copy()
    independent[0, 1] += 0.05
    holonomy = loop_holonomy(independent, link_12, link_20)
    assert loop_mismatch_energy(holonomy) > 0.0
