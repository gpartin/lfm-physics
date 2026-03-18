"""Tests for lfm.fields subpackage."""

import numpy as np
import pytest

from lfm.constants import CHI0, KAPPA
from lfm.fields import (
    equilibrate_chi,
    equilibrate_from_fields,
    gaussian_soliton,
    grid_positions,
    place_solitons,
    poisson_solve_fft,
    seed_noise,
    sparse_positions,
    tetrahedral_positions,
    uniform_chi,
    wave_kick,
)


# ──── soliton.py ────


class TestGaussianSoliton:
    def test_shape_and_dtype(self):
        pr, pi = gaussian_soliton(16, (8, 8, 8), 5.0, 2.0)
        assert pr.shape == (16, 16, 16)
        assert pr.dtype == np.float32

    def test_peak_at_center(self):
        pr, pi = gaussian_soliton(16, (8, 8, 8), 5.0, 2.0, phase=0.0)
        assert np.isclose(pr[8, 8, 8], 5.0, rtol=1e-5)
        assert np.isclose(pi[8, 8, 8], 0.0, atol=1e-6)

    def test_phase_pi_gives_negative(self):
        pr, pi = gaussian_soliton(16, (8, 8, 8), 5.0, 2.0, phase=np.pi)
        assert np.isclose(pr[8, 8, 8], -5.0, rtol=1e-5)
        assert abs(pi[8, 8, 8]) < 1e-5

    def test_decays_away_from_center(self):
        pr, _ = gaussian_soliton(32, (16, 16, 16), 5.0, 2.0)
        assert pr[16, 16, 16] > pr[16, 16, 20]
        assert pr[16, 16, 20] > pr[16, 16, 24]

    def test_symmetry(self):
        pr, _ = gaussian_soliton(32, (16, 16, 16), 5.0, 3.0)
        assert np.isclose(pr[16, 16, 13], pr[16, 16, 19], rtol=1e-5)
        assert np.isclose(pr[13, 16, 16], pr[16, 19, 16], rtol=1e-5)


class TestPlaceSolitons:
    def test_shape_and_dtype(self):
        positions = [(4, 4, 4), (12, 12, 12)]
        pr, pi = place_solitons(16, positions, 5.0, 2.0)
        assert pr.shape == (3, 16, 16, 16)
        assert pr.dtype == np.float32

    def test_round_robin_color(self):
        positions = [(4, 4, 4), (12, 12, 12), (4, 12, 4)]
        pr, pi = place_solitons(16, positions, 5.0, 2.0)
        # First soliton → color 0, second → color 1, third → color 2
        assert pr[0, 4, 4, 4] > 0
        assert pr[1, 12, 12, 12] > 0
        assert pr[2, 4, 12, 4] > 0

    def test_explicit_colors(self):
        positions = [(4, 4, 4), (12, 12, 12)]
        pr, _ = place_solitons(16, positions, 5.0, 2.0, colors=[2, 2])
        assert pr[2, 4, 4, 4] > 0
        assert pr[2, 12, 12, 12] > 0
        assert np.allclose(pr[0], 0)
        assert np.allclose(pr[1], 0)

    def test_mixed_phases(self):
        positions = [(4, 4, 4), (12, 12, 12)]
        pr, pi = place_solitons(
            16, positions, 5.0, 2.0, phases=[0.0, np.pi], colors=[0, 0]
        )
        # Opposite-charge solitons on same color
        assert pr[0, 4, 4, 4] > 0
        assert pr[0, 12, 12, 12] < 0


class TestWaveKick:
    def test_preserves_norm(self):
        N = 8
        pr = np.random.default_rng(42).standard_normal((N, N, N)).astype(np.float32)
        pi = np.zeros_like(pr)
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        pr_prev, pi_prev = wave_kick(pr, pi, chi, 0.02)
        norm_before = np.sum(pr**2 + pi**2)
        norm_after = np.sum(pr_prev**2 + pi_prev**2)
        assert np.isclose(norm_before, norm_after, rtol=1e-4)

    def test_activates_imaginary(self):
        N = 8
        pr = np.ones((N, N, N), dtype=np.float32)
        pi = np.zeros((N, N, N), dtype=np.float32)
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        _, pi_prev = wave_kick(pr, pi, chi, 0.02)
        # Should have non-zero imaginary after kick
        assert np.any(np.abs(pi_prev) > 0.01)

    def test_works_with_4d(self):
        # Multi-color fields
        pr = np.ones((3, 8, 8, 8), dtype=np.float32)
        pi = np.zeros((3, 8, 8, 8), dtype=np.float32)
        chi = np.full((8, 8, 8), CHI0, dtype=np.float32)
        pr_prev, pi_prev = wave_kick(pr, pi, chi, 0.02)
        assert pr_prev.shape == (3, 8, 8, 8)


# ──── equilibrium.py ────


class TestPoissonSolve:
    def test_zero_source_gives_zero(self):
        source = np.zeros((16, 16, 16), dtype=np.float32)
        phi = poisson_solve_fft(source, 16)
        assert np.allclose(phi, 0, atol=1e-6)

    def test_zero_mean_solution(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((16, 16, 16)).astype(np.float32)
        phi = poisson_solve_fft(source, 16)
        assert abs(np.mean(phi)) < 1e-5

    def test_roundtrip_spectral(self):
        """Test ∇²(solution) = source via spectral Laplacian (exact)."""
        N = 32
        rng = np.random.default_rng(99)
        source = rng.standard_normal((N, N, N)).astype(np.float32)
        source -= np.mean(source)  # zero-mean (solvable)
        phi = poisson_solve_fft(source, N)

        # Spectral Laplacian: multiply by -K² in Fourier space
        kx = np.fft.fftfreq(N) * 2 * np.pi
        ky = np.fft.fftfreq(N) * 2 * np.pi
        kz = np.fft.rfftfreq(N) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K2 = KX**2 + KY**2 + KZ**2
        lap_hat = -K2 * np.fft.rfftn(phi)
        laplacian = np.fft.irfftn(lap_hat, s=(N, N, N))
        assert np.allclose(laplacian, source, atol=1e-3)


class TestEquilibrateChi:
    def test_zero_energy_gives_chi0(self):
        psi_sq = np.zeros((16, 16, 16), dtype=np.float32)
        chi = equilibrate_chi(psi_sq)
        assert np.allclose(chi, CHI0, atol=1e-5)

    def test_energy_creates_well(self):
        N = 32
        psi_sq = np.zeros((N, N, N), dtype=np.float32)
        # Central blob of energy
        c = N // 2
        psi_sq[c - 2 : c + 3, c - 2 : c + 3, c - 2 : c + 3] = 100.0
        chi = equilibrate_chi(psi_sq)
        # Chi at center should be BELOW background (well)
        assert chi[c, c, c] < CHI0

    def test_boundary_mask(self):
        N = 16
        psi_sq = np.ones((N, N, N), dtype=np.float32)
        mask = np.zeros((N, N, N), dtype=bool)
        mask[0, :, :] = True
        mask[-1, :, :] = True
        chi = equilibrate_chi(psi_sq, boundary_mask=mask)
        assert np.allclose(chi[0], CHI0)
        assert np.allclose(chi[-1], CHI0)


class TestEquilibrateFromFields:
    def test_real_field(self):
        N = 16
        pr, _ = gaussian_soliton(N, (8, 8, 8), 5.0, 2.0)
        chi = equilibrate_from_fields(pr, psi_i=None)
        assert chi.shape == (N, N, N)
        assert chi[8, 8, 8] < CHI0

    def test_complex_field(self):
        N = 16
        pr, pi = gaussian_soliton(N, (8, 8, 8), 5.0, 2.0, phase=0.5)
        chi = equilibrate_from_fields(pr, pi)
        assert chi[8, 8, 8] < CHI0

    def test_color_field(self):
        positions = [(8, 8, 8)]
        pr, pi = place_solitons(16, positions, 5.0, 2.0)
        chi = equilibrate_from_fields(pr, pi)
        assert chi.shape == (16, 16, 16)
        assert chi[8, 8, 8] < CHI0


# ──── random.py ────


class TestSeedNoise:
    def test_shape_single(self):
        pr, pi = seed_noise(16, amplitude=1e-6, n_colors=1, rng=42)
        assert pr.shape == (16, 16, 16)
        assert pr.dtype == np.float32

    def test_shape_color(self):
        pr, pi = seed_noise(16, amplitude=1e-6, n_colors=3, rng=42)
        assert pr.shape == (3, 16, 16, 16)

    def test_amplitude_order(self):
        pr, pi = seed_noise(16, amplitude=1e-6, rng=42)
        assert np.std(pr) < 1e-4  # should be ~1e-6

    def test_reproducible(self):
        a, _ = seed_noise(16, rng=42)
        b, _ = seed_noise(16, rng=42)
        assert np.array_equal(a, b)


class TestUniformChi:
    def test_value_and_shape(self):
        chi = uniform_chi(16)
        assert chi.shape == (16, 16, 16)
        assert np.allclose(chi, 19.0)

    def test_custom_chi0(self):
        chi = uniform_chi(8, chi0=10.0)
        assert np.allclose(chi, 10.0)


# ──── arrangements.py ────


class TestTetrahedralPositions:
    def test_four_positions(self):
        pos = tetrahedral_positions(64)
        assert pos.shape == (4, 3)

    def test_inside_grid(self):
        pos = tetrahedral_positions(64)
        assert np.all(pos >= 0)
        assert np.all(pos < 64)

    def test_equal_distances(self):
        """Vertices of a regular tetrahedron are all equidistant."""
        pos = tetrahedral_positions(64)
        dists = []
        for i in range(4):
            for j in range(i + 1, 4):
                dists.append(np.linalg.norm(pos[i] - pos[j]))
        # All 6 distances should be equal
        assert np.allclose(dists, dists[0], rtol=1e-10)


class TestSparsePositions:
    def test_shape(self):
        pos = sparse_positions(64, 10, seed=42)
        assert pos.shape == (10, 3)

    def test_inside_sphere(self):
        N = 64
        pos = sparse_positions(N, 10, boundary_fraction=0.15, sigma=3.0, seed=42)
        center = N / 2.0
        radii = np.linalg.norm(pos - center, axis=1)
        r_inner = N / 2.0 - 0.15 * N
        r_max = r_inner - 9.0
        assert np.all(radii <= r_max + 0.01)

    def test_minimum_separation(self):
        pos = sparse_positions(64, 5, sigma=3.0, seed=42)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dist = np.linalg.norm(pos[i] - pos[j])
                assert dist >= 6.0 - 0.01  # 2*sigma

    def test_too_many_raises(self):
        with pytest.raises(ValueError):
            sparse_positions(16, 100, sigma=3.0, seed=42)


class TestGridPositions:
    def test_shape(self):
        pos = grid_positions(32, 3)
        assert pos.shape == (27, 3)  # 3³

    def test_inside_grid(self):
        pos = grid_positions(32, 3)
        assert np.all(pos > 0)
        assert np.all(pos < 32)


# ──── Integration: top-level import ────


class TestTopLevelImport:
    def test_fields_importable_from_lfm(self):
        import lfm

        assert hasattr(lfm, "gaussian_soliton")
        assert hasattr(lfm, "equilibrate_chi")
        assert hasattr(lfm, "seed_noise")
        assert hasattr(lfm, "tetrahedral_positions")
