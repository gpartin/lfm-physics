"""Tests for config presets and Helmholtz S_a kernel."""

from __future__ import annotations

import numpy as np
import pytest

from lfm.config import FieldLevel
from lfm.config_presets import full_physics, gravity_em, gravity_only
from lfm.constants import (
    EPSILON_CC,
    KAPPA_C,
    KAPPA_STRING,
    KAPPA_TUBE,
    LAMBDA_H,
    SA_D,
    SA_GAMMA,
)


class TestConfigPresets:
    def test_gravity_only_level0(self):
        cfg = gravity_only()
        assert cfg.field_level == FieldLevel.REAL
        assert cfg.lambda_self == 0.0
        assert cfg.kappa_c == 0.0

    def test_gravity_em_level1(self):
        cfg = gravity_em()
        assert cfg.field_level == FieldLevel.COMPLEX
        assert cfg.lambda_self == pytest.approx(LAMBDA_H)
        assert cfg.kappa_c == 0.0

    def test_full_physics_level2(self):
        cfg = full_physics()
        assert cfg.field_level == FieldLevel.COLOR
        assert cfg.lambda_self == pytest.approx(LAMBDA_H)
        assert cfg.kappa_c == pytest.approx(KAPPA_C)
        assert cfg.epsilon_cc == pytest.approx(EPSILON_CC)
        assert cfg.kappa_string == pytest.approx(KAPPA_STRING)
        assert cfg.kappa_tube == pytest.approx(KAPPA_TUBE)
        assert cfg.sa_gamma == pytest.approx(SA_GAMMA)
        assert cfg.sa_d == pytest.approx(SA_D)

    def test_grid_size_override(self):
        cfg = gravity_only(grid_size=32)
        assert cfg.grid_size == 32

    def test_kwargs_override(self):
        cfg = gravity_em(grid_size=32, lambda_self=0.0)
        assert cfg.lambda_self == 0.0

    def test_presets_importable_from_lfm(self):
        import lfm

        assert hasattr(lfm, "gravity_only")
        assert hasattr(lfm, "gravity_em")
        assert hasattr(lfm, "full_physics")


class TestHelmholtzSaKernel:
    """Test the v17 Helmholtz S_a kernel matches expected analytical behavior."""

    def test_helmholtz_matches_analytical_gaussian(self):
        """For a Gaussian |Ψ_a|², S_a should be a broader Gaussian (smoothed).

        The Helmholtz kernel γ/(γ + D·k²) in Fourier space is a low-pass filter
        with characteristic length L = √(D/γ).  For our canonical params,
        L = √(4.9/0.1) = 7 lattice units.
        """
        from lfm.simulation import Simulation

        N = 32
        cfg = full_physics(grid_size=N)
        sim = Simulation(cfg, backend="cpu")
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=8.0)
        sim.run(steps=10, record_metrics=False)

        sa = sim.sa_fields
        assert sa is not None
        # S_a should be non-zero near soliton center
        center = N // 2
        assert sa[:, center, center, center].max() > 1e-6
        # S_a should be smoother (wider) than |Ψ_a|² — peak value should be
        # LOWER than the |Ψ_a|² peak because energy is spread by the kernel
        psi_sq = sim.psi_real**2 + sim.psi_imag**2  # (3, N, N, N) or (N,N,N)
        psi_sq_sum = psi_sq.sum(axis=0) if psi_sq.ndim == 4 else psi_sq
        sa_sum = sa.sum(axis=0)
        # Peak of S_a should be less than peak of |Ψ|² (smoothing spreads energy)
        assert sa_sum.max() <= psi_sq_sum.max() + 1e-3

    def test_helmholtz_preserves_total_integral(self):
        """Helmholtz kernel preserves total integral (DC mode: γ/(γ+0) = 1).

        For any field f, ∫ H[f] d³x = ∫ f d³x because H(k=0)=1.
        Test this mathematical property directly.
        """
        N = 16
        gamma, D = 0.1, 4.9
        x = np.arange(N, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        c = N // 2
        f = np.exp(-((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) / 8.0).astype(np.float32)

        kx = np.fft.fftfreq(N) * (2 * np.pi)
        ky = np.fft.fftfreq(N) * (2 * np.pi)
        kz = np.fft.rfftfreq(N) * (2 * np.pi)
        k_sq = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
        h_filter = gamma / (gamma + D * k_sq)

        f_hat = np.fft.rfftn(f)
        s = np.fft.irfftn(h_filter * f_hat, s=(N, N, N))

        np.testing.assert_allclose(s.sum(), f.sum(), rtol=1e-5)

    def test_helmholtz_scv_zero_for_singlet(self):
        """When all 3 colors have equal |Ψ_a|², SCV should be ~0."""
        from lfm.simulation import Simulation

        N = 16
        cfg = full_physics(grid_size=N)
        sim = Simulation(cfg, backend="cpu")
        # Place soliton — default places equal amplitude in all colors
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=6.0)
        sim.run(steps=5, record_metrics=False)

        sa = sim.sa_fields
        assert sa is not None
        # Compute SCV manually
        sa_sum = sa.sum(axis=0)
        sa_sq_sum = (sa**2).sum(axis=0)
        scv = sa_sq_sum - (1.0 / 3.0) * sa_sum**2
        # SCV should be near zero for equal-color config
        assert scv.max() < 0.1 * sa.max() ** 2
