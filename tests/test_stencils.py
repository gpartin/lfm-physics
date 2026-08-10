"""Tests for lfm.core.stencils — Laplacian operators."""

import numpy as np

from lfm.core.stencils import (
    eigenvalue_19pt,
    eigenvalue_27pt,
    gradient_19pt,
    laplacian_7pt,
    laplacian_19pt,
    laplacian_27pt,
)


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

    def test_eigenvalue_matches_single_fourier_mode(self):
        N = 16
        coords = np.arange(N, dtype=np.float64)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        kx_i, ky_i, kz_i = 2, 1, 0
        kx = 2.0 * np.pi * kx_i / N
        ky = 2.0 * np.pi * ky_i / N
        kz = 2.0 * np.pi * kz_i / N
        field = np.cos(kx * X + ky * Y + kz * Z)
        lap = laplacian_19pt(field)
        lam = float(eigenvalue_19pt(np.array(kx), np.array(ky), np.array(kz)))
        np.testing.assert_allclose(lap, lam * field, atol=1e-12)


class TestGradient19pt:
    def test_constant_field_zero(self):
        gradient = gradient_19pt(np.ones((12, 12, 12)))
        for component in gradient:
            np.testing.assert_allclose(component, 0.0, atol=1e-12)

    def test_axis_fourier_mode(self):
        size = 24
        k = 2.0 * np.pi / size
        x = np.arange(size, dtype=np.float64)
        field = np.broadcast_to(
            np.sin(k * x)[:, None, None],
            (size, size, size),
        ).copy()
        gx, gy, gz = gradient_19pt(field)
        expected = np.broadcast_to(
            (np.sin(k) * np.cos(k * x))[:, None, None],
            field.shape,
        )
        np.testing.assert_allclose(gx, expected, atol=1e-12)
        np.testing.assert_allclose(gy, 0.0, atol=1e-12)
        np.testing.assert_allclose(gz, 0.0, atol=1e-12)


class TestLaplacian27pt:
    def test_constant_field_zero(self):
        field = np.ones((16, 16, 16))
        lap = laplacian_27pt(field)
        np.testing.assert_allclose(lap, 0.0, atol=1e-12)

    def test_eigenvalue_matches_single_fourier_mode(self):
        size = 16
        coords = np.arange(size, dtype=np.float64)
        x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
        kx_i, ky_i, kz_i = 2, 1, 3
        kx = 2.0 * np.pi * kx_i / size
        ky = 2.0 * np.pi * ky_i / size
        kz = 2.0 * np.pi * kz_i / size
        field = np.cos(kx * x + ky * y + kz * z)
        lap = laplacian_27pt(field)
        eigenvalue = float(
            eigenvalue_27pt(np.array(kx), np.array(ky), np.array(kz))
        )
        np.testing.assert_allclose(lap, eigenvalue * field, atol=1e-12)


class TestLaplacian7pt:
    def test_constant_field_zero(self):
        field = np.ones((16, 16, 16))
        lap = laplacian_7pt(field)
        np.testing.assert_allclose(lap, 0.0, atol=1e-12)

    def test_output_shape(self):
        field = np.random.default_rng(42).standard_normal((20, 20, 20))
        assert laplacian_7pt(field).shape == field.shape
