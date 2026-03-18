"""Tests for lfm.core.stencils — Laplacian operators."""

import numpy as np
import pytest

from lfm.core.stencils import laplacian_7pt, laplacian_19pt


class TestLaplacian19pt:
    def test_constant_field_zero(self):
        """Laplacian of a constant = 0."""
        field = np.ones((16, 16, 16))
        lap = laplacian_19pt(field)
        np.testing.assert_allclose(lap, 0.0, atol=1e-12)

    def test_linear_field_zero(self):
        """Laplacian of a linear function = 0 on periodic grid.

        A linear function on a periodic grid wraps around, but the
        stencil at each point sees a constant gradient → ∇²=0.
        """
        N = 16
        x = np.arange(N, dtype=float)
        field = np.broadcast_to(x[:, None, None], (N, N, N)).copy()
        lap = laplacian_19pt(field)
        # Not exactly 0 due to periodic discontinuity, but interior points are 0
        interior = lap[2:-2, 2:-2, 2:-2]
        np.testing.assert_allclose(interior, 0.0, atol=1e-12)

    def test_quadratic_field(self):
        """Laplacian of x² = 2 (constant)."""
        N = 32
        x = np.arange(N, dtype=float) - N / 2
        field = np.broadcast_to(x[:, None, None] ** 2, (N, N, N)).copy()
        lap = laplacian_19pt(field)
        # Interior should be ≈ 2.0 (away from periodic boundaries)
        interior = lap[4:-4, 4:-4, 4:-4]
        np.testing.assert_allclose(interior, 2.0, atol=0.1)

    def test_symmetry(self):
        """Symmetric input → symmetric output."""
        N = 16
        field = np.zeros((N, N, N))
        field[N // 2, N // 2, N // 2] = 1.0
        lap = laplacian_19pt(field)
        # Should have cubic symmetry
        c = N // 2
        assert lap[c + 1, c, c] == lap[c - 1, c, c]
        assert lap[c, c + 1, c] == lap[c, c - 1, c]
        assert lap[c, c, c + 1] == lap[c, c, c - 1]
        # Face and edge should differ (different weights)
        assert lap[c + 1, c, c] != lap[c + 1, c + 1, c]

    def test_output_shape(self):
        field = np.random.default_rng(42).standard_normal((20, 20, 20))
        lap = laplacian_19pt(field)
        assert lap.shape == field.shape


class TestLaplacian7pt:
    def test_constant_field_zero(self):
        field = np.ones((16, 16, 16))
        lap = laplacian_7pt(field)
        np.testing.assert_allclose(lap, 0.0, atol=1e-12)

    def test_output_shape(self):
        field = np.random.default_rng(42).standard_normal((20, 20, 20))
        assert laplacian_7pt(field).shape == field.shape
