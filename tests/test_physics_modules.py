"""Tests for new physics modules: metric, phase, angular_momentum, boosted, sweep_2d."""

import numpy as np
import pytest

from lfm.analysis.angular_momentum import (
    angular_momentum_density,
    precession_rate,
    total_angular_momentum,
)
from lfm.analysis.metric import (
    effective_metric_00,
    gravitational_potential,
    metric_perturbation,
    metric_refractive_index,
    op05_spherical_chi_deflection,
    schwarzschild_chi,
    schwarzschild_radius_si,
    time_dilation_factor,
)
from lfm.analysis.phase import (
    charge_density,
    coulomb_interaction_energy,
    noether_spatial_current,
    phase_coherence,
    phase_current_energy_density,
    phase_field,
    positive_noether_current,
)
from lfm.config import SimulationConfig
from lfm.constants import SOLAR_MASS_KG, SOLAR_RADIUS_M
from lfm.fields.boosted import boosted_soliton
from lfm.sweep import sweep_2d

# ── Metric ───────────────────────────────────────────────────────────


class TestMetric:
    def test_vacuum_g00(self):
        chi = np.full((8, 8, 8), 19.0)
        g00 = effective_metric_00(chi)
        np.testing.assert_allclose(g00, -1.0)

    def test_well_g00(self):
        chi = np.full((8, 8, 8), 19.0 / 2)
        g00 = effective_metric_00(chi)
        np.testing.assert_allclose(g00, -0.25)

    def test_metric_perturbation_vacuum(self):
        chi = np.full((8, 8, 8), 19.0)
        h00 = metric_perturbation(chi)
        np.testing.assert_allclose(h00, 0.0, atol=1e-14)

    def test_metric_perturbation_well(self):
        chi = np.full((8, 8, 8), 18.0)
        h00 = metric_perturbation(chi)
        assert np.all(h00 > 0)

    def test_time_dilation_vacuum(self):
        chi = np.full((8, 8, 8), 19.0)
        td = time_dilation_factor(chi)
        np.testing.assert_allclose(td, 1.0)

    def test_time_dilation_well(self):
        chi = np.full((8, 8, 8), 9.5)
        td = time_dilation_factor(chi)
        np.testing.assert_allclose(td, 0.5)

    def test_gravitational_potential_vacuum(self):
        chi = np.full((8, 8, 8), 19.0)
        phi = gravitational_potential(chi)
        np.testing.assert_allclose(phi, 0.0, atol=1e-14)

    def test_gravitational_potential_negative_in_well(self):
        chi = np.full((8, 8, 8), 15.0)
        phi = gravitational_potential(chi)
        assert np.all(phi > 0)  # h00/2 > 0 when chi < chi0

    def test_schwarzschild_chi_boundary(self):
        chi = schwarzschild_chi(32, (16, 16, 16), r_s=3.0)
        assert chi.shape == (32, 32, 32)
        # Far from center → approximately chi0 (corner distance ~27.7)
        assert chi[0, 0, 0] > 17.5
        # At center → 0 (inside horizon)
        assert chi[16, 16, 16] == pytest.approx(0.0, abs=0.1)

    def test_metric_refractive_index_increases_in_chi_well(self):
        chi = np.array([19.0, 18.99], dtype=np.float64)
        n_eff = metric_refractive_index(chi)
        assert n_eff[0] == pytest.approx(1.0)
        assert n_eff[1] > 1.0

    def test_op05_solar_limb_deflection_recovers_arcsecond_scale(self):
        result = op05_spherical_chi_deflection(
            SOLAR_MASS_KG,
            SOLAR_RADIUS_M,
            x_extent_multiplier=500.0,
            sample_count=20001,
        )
        assert result["schwarzschild_radius_m"] == pytest.approx(
            schwarzschild_radius_si(SOLAR_MASS_KG),
            rel=1e-14,
        )
        assert result["recovered_angle_arcsec"] == pytest.approx(1.751243281, rel=1e-3)
        assert abs(result["comparator_relative_error"]) < 1e-4

    def test_op05_deflection_is_linear_in_weak_lens_mass(self):
        half = op05_spherical_chi_deflection(
            0.5 * SOLAR_MASS_KG,
            SOLAR_RADIUS_M,
            x_extent_multiplier=250.0,
            sample_count=5001,
        )
        full = op05_spherical_chi_deflection(
            SOLAR_MASS_KG,
            SOLAR_RADIUS_M,
            x_extent_multiplier=250.0,
            sample_count=5001,
        )
        ratio = half["recovered_angle_arcsec"] / full["recovered_angle_arcsec"]
        assert ratio == pytest.approx(0.5, rel=1e-5)


# ── Phase ────────────────────────────────────────────────────────────


class TestPhase:
    def test_phase_of_real(self):
        """Pure real positive field → phase = 0."""
        psi_r = np.ones((8, 8, 8))
        psi_i = np.zeros((8, 8, 8))
        theta = phase_field(psi_r, psi_i)
        np.testing.assert_allclose(theta, 0.0)

    def test_phase_of_negative_real(self):
        """Pure real negative → phase = ±π."""
        psi_r = -np.ones((8, 8, 8))
        psi_i = np.zeros((8, 8, 8))
        theta = phase_field(psi_r, psi_i)
        np.testing.assert_allclose(np.abs(theta), np.pi)

    def test_charge_density_static(self):
        """Static field → zero charge density."""
        psi_r = np.ones((8, 8, 8))
        psi_i = np.zeros((8, 8, 8))
        rho = charge_density(psi_r, psi_i)
        np.testing.assert_allclose(rho, 0.0)

    def test_charge_density_with_prev(self):
        psi_r = np.ones((8, 8, 8))
        psi_i = np.zeros((8, 8, 8))
        # Imaginary part was zero, now 0.1 → dpsi_i/dt = 0.1/0.02 = 5
        rho = charge_density(
            psi_r, 0.1 * np.ones((8, 8, 8)), dt=0.02, psi_r_prev=psi_r, psi_i_prev=psi_i
        )
        # ρ = psi_r * dpsi_i_dt = 1 * 5 = 5
        np.testing.assert_allclose(rho, 5.0)

    def test_coherence_uniform(self):
        """All same phase → coherence = 1."""
        psi_r = np.ones((8, 8, 8)) * 5.0
        psi_i = np.zeros((8, 8, 8))
        assert phase_coherence(psi_r, psi_i) == pytest.approx(1.0)

    def test_coherence_random(self):
        """Random phases → coherence near 0."""
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, (16, 16, 16))
        psi_r = np.cos(theta)
        psi_i = np.sin(theta)
        assert phase_coherence(psi_r, psi_i) < 0.15

    def test_coherence_zero_field(self):
        psi_r = np.zeros((8, 8, 8))
        psi_i = np.zeros((8, 8, 8))
        assert phase_coherence(psi_r, psi_i) == 0.0

    def test_coulomb_same_phase(self):
        """Same phase → positive interaction energy (repulsion)."""
        psi = np.ones((8, 8, 8))
        zero = np.zeros((8, 8, 8))
        e_int = coulomb_interaction_energy(psi, zero, psi, zero)
        assert e_int > 0

    def test_coulomb_opposite_phase(self):
        """Opposite phase → negative interaction energy (attraction)."""
        psi = np.ones((8, 8, 8))
        zero = np.zeros((8, 8, 8))
        e_int = coulomb_interaction_energy(psi, zero, -psi, zero)
        assert e_int < 0

    def test_noether_spatial_current_plane_wave(self):
        N = 16
        k = 2.0 * np.pi / N
        x = np.arange(N, dtype=np.float32)
        phase = k * x[:, None, None]
        psi_r = np.broadcast_to(np.cos(phase), (N, N, N)).astype(np.float32)
        psi_i = np.broadcast_to(np.sin(phase), (N, N, N)).astype(np.float32)
        jx = noether_spatial_current(psi_r, psi_i, axis=0)
        np.testing.assert_allclose(jx, np.sin(k), rtol=1e-5, atol=1e-6)

    def test_noether_spatial_current_preserves_float64(self):
        N = 16
        k = 2.0 * np.pi / N
        x = np.arange(N, dtype=np.float64)
        phase = k * x[:, None, None]
        psi_r = np.broadcast_to(np.cos(phase), (N, N, N)).astype(np.float64)
        psi_i = np.broadcast_to(np.sin(phase), (N, N, N)).astype(np.float64)
        jx = positive_noether_current(psi_r, psi_i, axis=0)
        assert jx.dtype == np.float64
        np.testing.assert_allclose(jx, np.sin(k), rtol=1e-12, atol=1e-12)

    def test_positive_noether_current_clips_reverse_wave(self):
        N = 16
        k = 2.0 * np.pi / N
        x = np.arange(N, dtype=np.float32)
        phase = -k * x[:, None, None]
        psi_r = np.broadcast_to(np.cos(phase), (N, N, N)).astype(np.float32)
        psi_i = np.broadcast_to(np.sin(phase), (N, N, N)).astype(np.float32)
        jx = positive_noether_current(psi_r, psi_i, axis=0)
        np.testing.assert_allclose(jx, 0.0, atol=1e-6)

    def test_phase_current_energy_static_uniform_zero(self):
        psi_r = np.ones((8, 8, 8), dtype=np.float64)
        psi_i = np.zeros((8, 8, 8), dtype=np.float64)
        rho = phase_current_energy_density(psi_r, psi_i, psi_r, psi_i, dt=0.02)
        assert rho.dtype == np.float64
        np.testing.assert_allclose(rho, 0.0, atol=1e-14)

    def test_phase_current_energy_global_phase_invariant(self):
        N = 16
        dt = 0.02
        k = 2.0 * np.pi / N
        omega = 0.35
        x = np.arange(N, dtype=np.float64)
        phase = k * x[:, None, None]
        psi_r = np.broadcast_to(np.cos(phase), (N, N, N)).astype(np.float64)
        psi_i = np.broadcast_to(np.sin(phase), (N, N, N)).astype(np.float64)
        prev_phase = phase - omega * dt
        psi_r_prev = np.broadcast_to(np.cos(prev_phase), (N, N, N)).astype(np.float64)
        psi_i_prev = np.broadcast_to(np.sin(prev_phase), (N, N, N)).astype(np.float64)

        rho = phase_current_energy_density(psi_r, psi_i, psi_r_prev, psi_i_prev, dt=dt)

        phi0 = 1.234
        psi_r_rot = np.cos(phi0) * psi_r - np.sin(phi0) * psi_i
        psi_i_rot = np.sin(phi0) * psi_r + np.cos(phi0) * psi_i
        psi_r_prev_rot = np.cos(phi0) * psi_r_prev - np.sin(phi0) * psi_i_prev
        psi_i_prev_rot = np.sin(phi0) * psi_r_prev + np.cos(phi0) * psi_i_prev
        rho_rot = phase_current_energy_density(
            psi_r_rot,
            psi_i_rot,
            psi_r_prev_rot,
            psi_i_prev_rot,
            dt=dt,
        )
        np.testing.assert_allclose(rho_rot, rho, rtol=1e-12, atol=1e-12)

    def test_phase_current_energy_sees_constant_amplitude_plane_wave(self):
        N = 32
        dt = 0.01
        amp = 0.4
        k = 2.0 * np.pi / N
        omega = k
        x = np.arange(N, dtype=np.float64)
        phase = k * x[:, None, None]
        prev_phase = phase - omega * dt
        psi_r = np.broadcast_to(amp * np.cos(phase), (N, N, N)).astype(np.float64)
        psi_i = np.broadcast_to(amp * np.sin(phase), (N, N, N)).astype(np.float64)
        psi_r_prev = np.broadcast_to(amp * np.cos(prev_phase), (N, N, N)).astype(np.float64)
        psi_i_prev = np.broadcast_to(amp * np.sin(prev_phase), (N, N, N)).astype(np.float64)

        amp_sq = psi_r * psi_r + psi_i * psi_i
        rho = phase_current_energy_density(psi_r, psi_i, psi_r_prev, psi_i_prev, dt=dt)

        np.testing.assert_allclose(amp_sq, amp * amp, rtol=1e-14, atol=1e-14)
        assert float(np.mean(rho)) > 0.0
        assert float(np.std(rho)) < 1e-12

    def test_phase_current_energy_scales_with_carrier(self):
        N = 64
        dt = 0.005
        amp = 0.25
        x = np.arange(N, dtype=np.float64)

        def mean_rho(mode: int) -> float:
            k = 2.0 * np.pi * mode / N
            phase = k * x[:, None, None]
            prev_phase = phase - k * dt
            psi_r = np.broadcast_to(amp * np.cos(phase), (N, N, N)).astype(np.float64)
            psi_i = np.broadcast_to(amp * np.sin(phase), (N, N, N)).astype(np.float64)
            psi_r_prev = np.broadcast_to(amp * np.cos(prev_phase), (N, N, N)).astype(np.float64)
            psi_i_prev = np.broadcast_to(amp * np.sin(prev_phase), (N, N, N)).astype(np.float64)
            return float(
                np.mean(
                    phase_current_energy_density(
                        psi_r,
                        psi_i,
                        psi_r_prev,
                        psi_i_prev,
                        dt=dt,
                    )
                )
            )

        rho_1 = mean_rho(1)
        rho_2 = mean_rho(2)
        assert rho_2 / rho_1 == pytest.approx(4.0, rel=0.08)


# ── Angular Momentum ─────────────────────────────────────────────────


class TestAngularMomentum:
    def test_static_field_zero_L(self):
        """No time variation → zero angular momentum."""
        psi_r = np.ones((8, 8, 8), dtype=np.float64) * 0.5
        psi_i = np.zeros((8, 8, 8), dtype=np.float64)
        Lx, Ly, Lz = total_angular_momentum(psi_r, psi_i, psi_r, psi_i, dt=0.02)
        assert abs(Lx) < 1e-10
        assert abs(Ly) < 1e-10
        assert abs(Lz) < 1e-10

    def test_density_shape(self):
        psi = np.ones((8, 8, 8), dtype=np.float64)
        z = np.zeros_like(psi)
        Lx, Ly, Lz = angular_momentum_density(psi, z, psi, z, dt=0.02)
        assert Lx.shape == (8, 8, 8)

    def test_precession_rate_insufficient_data(self):
        assert precession_rate([], 1.0) == 0.0
        assert precession_rate([(1, 0, 0)], 1.0) == 0.0

    def test_precession_rate_constant(self):
        """Constant L → zero precession."""
        L_history = [(1.0, 0.0, 0.0)] * 10
        omega = precession_rate(L_history, 1.0)
        assert abs(omega) < 1e-10

    def test_precession_rate_rotating(self):
        """L rotating at known rate."""
        omega0 = 0.1  # rad per unit time
        L_history = [(np.cos(omega0 * t), np.sin(omega0 * t), 0.0) for t in range(20)]
        omega = precession_rate(L_history, 1.0)
        assert omega == pytest.approx(omega0, rel=0.01)


# ── Boosted Soliton ─────────────────────────────────────────────────


class TestBoostedSoliton:
    def test_static_soliton(self):
        """Zero velocity → e_dot = 0, same as gaussian_soliton."""
        psi_r, psi_i, e_dot = boosted_soliton(16, (8, 8, 8), amplitude=3.0, sigma=2.5)
        assert psi_r.shape == (16, 16, 16)
        np.testing.assert_allclose(e_dot, 0.0)
        assert psi_r.max() == pytest.approx(3.0, rel=0.01)

    def test_boosted_real_has_edot(self):
        """Real soliton with velocity → non-zero e_dot."""
        psi_r, psi_i, e_dot = boosted_soliton(
            16,
            (8, 8, 8),
            amplitude=3.0,
            sigma=2.5,
            velocity=(0.1, 0.0, 0.0),
        )
        assert np.max(np.abs(e_dot)) > 0

    def test_boosted_complex_phase_gradient(self):
        """Complex soliton with velocity → phase varies along boost axis."""
        psi_r, psi_i, e_dot = boosted_soliton(
            16,
            (8, 8, 8),
            amplitude=3.0,
            sigma=2.5,
            phase=0.5,
            velocity=(0.05, 0.0, 0.0),
        )
        # Phase should vary along x (axis 0, where velocity is)
        theta = np.arctan2(psi_i[:, 8, 8], psi_r[:, 8, 8])
        # Should not be constant where amplitude is non-negligible
        amp = np.sqrt(psi_r[:, 8, 8] ** 2 + psi_i[:, 8, 8] ** 2)
        mask = amp > 0.01 * amp.max()
        assert np.std(theta[mask]) > 0.01

    def test_amplitude_preserved(self):
        """Boost should preserve peak amplitude."""
        pr_s, _, _ = boosted_soliton(16, (8, 8, 8), 3.0, 2.5)
        pr_b, pi_b, _ = boosted_soliton(
            16, (8, 8, 8), 3.0, 2.5, phase=0.5, velocity=(0.02, 0.0, 0.0)
        )
        # |Ψ| should have same peak
        mod_static = pr_s.max()
        mod_boosted = np.sqrt(pr_b**2 + pi_b**2).max()
        assert mod_boosted == pytest.approx(mod_static, rel=0.01)


# ── Sweep 2D ────────────────────────────────────────────────────────


class TestSweep2D:
    def test_basic_2d_sweep(self):
        cfg = SimulationConfig(grid_size=8, e_amplitude=1.0)
        results = sweep_2d(
            cfg,
            param1="kappa",
            values1=[1 / 63, 1 / 31],
            param2="lambda_self",
            values2=[0.0, 0.1],
            steps=5,
            metric_names=["chi_min"],
            equilibrate=False,
        )
        assert len(results) == 4  # 2 × 2
        for row in results:
            assert "kappa" in row
            assert "lambda_self" in row
            assert "chi_min" in row

    def test_2d_sweep_with_soliton(self):
        cfg = SimulationConfig(grid_size=8, e_amplitude=1.0)
        results = sweep_2d(
            cfg,
            param1="kappa",
            values1=[1 / 63],
            param2="chi0",
            values2=[19.0],
            steps=5,
            soliton={"amplitude": 2.0},
            equilibrate=False,
        )
        assert len(results) == 1
        assert results[0]["chi_min"] < 19.0  # Soliton creates a well
