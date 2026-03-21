"""Tests to boost coverage on units, observables, energy, and tracker modules."""

import numpy as np
import pytest

from lfm.analysis.energy import (
    continuity_residual,
    energy_conservation_drift,
    fluid_fields,
    total_energy,
)
from lfm.analysis.observables import (
    confinement_proxy,
    find_peaks,
    fit_power_law,
    measure_force,
    measure_separation,
    momentum_density,
    radial_profile,
    weak_parity_asymmetry,
)
from lfm.units import CosmicScale, PlanckScale

# ── Units: CosmicScale ──────────────────────────────────────────────


class TestCosmicScale:
    def test_cell_to_mpc(self):
        s = CosmicScale(box_mpc=100.0, grid_size=256)
        assert s.cell_to_mpc() == pytest.approx(100.0 / 256)

    def test_step_to_gyr_present(self):
        s = CosmicScale()
        assert s.step_to_gyr(541_000) == pytest.approx(13.8, abs=0.01)

    def test_gyr_to_step_roundtrip(self):
        s = CosmicScale()
        step = s.gyr_to_step(13.8)
        assert step == pytest.approx(541_000, rel=0.01)

    def test_format_cosmic_time_gyr(self):
        s = CosmicScale()
        text = s.format_cosmic_time(541_000)
        assert "Gyr" in text

    def test_format_cosmic_time_myr(self):
        s = CosmicScale()
        text = s.format_cosmic_time(10)  # tiny → Myr
        assert "Myr" in text


# ── Units: PlanckScale ──────────────────────────────────────────────


class TestPlanckScale:
    def test_default_observable_universe(self):
        ps = PlanckScale(grid_size=256)
        assert ps.cells_per_planck > 1e50
        assert not ps.is_planck_resolution
        assert ps.box_size_mpc > 1000

    def test_at_planck_resolution(self):
        ps = PlanckScale.at_planck_resolution(grid_size=256)
        assert ps.is_planck_resolution
        assert ps.cells_per_planck == pytest.approx(1.0)
        assert ps.cell_size_m == pytest.approx(1.616e-35, rel=0.01)

    def test_step_to_gyr_roundtrip(self):
        ps = PlanckScale(grid_size=256)
        gyr = ps.step_to_gyr(6401)
        step = ps.gyr_to_step(gyr)
        assert step == pytest.approx(6401, abs=1)

    def test_step_to_planck_ticks(self):
        ps = PlanckScale.at_planck_resolution()
        # 1 step = 0.02 Planck ticks (dt=0.02, 1 cell = 1 Planck)
        assert ps.step_to_planck_ticks(100) == pytest.approx(2.0)

    def test_planck_ticks_to_step(self):
        ps = PlanckScale.at_planck_resolution()
        assert ps.planck_ticks_to_step(2.0) == 100

    def test_step_to_seconds(self):
        ps = PlanckScale.at_planck_resolution()
        sec = ps.step_to_seconds(1)
        assert sec > 0
        assert sec < 1e-40  # Planck-resolution steps are extremely tiny

    def test_str_planck_resolution(self):
        ps = PlanckScale.at_planck_resolution()
        text = str(ps)
        assert "Planck-resolution" in text

    def test_str_default(self):
        ps = PlanckScale()
        text = str(ps)
        assert "Mpc" in text

    def test_gyr_per_step(self):
        ps = PlanckScale()
        assert ps.gyr_per_step > 0

    def test_planck_ticks_per_step(self):
        ps = PlanckScale()
        assert ps.planck_ticks_per_step > 0


# ── Observables ─────────────────────────────────────────────────────


class TestRadialProfile:
    def test_uniform_field(self):
        field = np.ones((16, 16, 16))
        result = radial_profile(field)
        np.testing.assert_allclose(result["profile"], 1.0, atol=1e-10)

    def test_center_peak(self):
        field = np.zeros((16, 16, 16))
        field[8, 8, 8] = 100.0
        result = radial_profile(field, center=(8, 8, 8))
        assert result["profile"][0] == pytest.approx(100.0)
        assert result["profile"][3] == pytest.approx(0.0)

    def test_custom_center(self):
        field = np.ones((16, 16, 16))
        result = radial_profile(field, center=(4, 4, 4), max_radius=3.0)
        assert len(result["r"]) == 4  # 0, 1, 2, 3


class TestFindPeaks:
    def test_two_peaks(self):
        field = np.zeros((16, 16, 16))
        field[4, 4, 4] = 10.0
        field[12, 12, 12] = 8.0
        peaks = find_peaks(field, n=2, min_separation=3)
        assert len(peaks) == 2
        assert peaks[0] == (4, 4, 4)  # Brightest first

    def test_empty_field(self):
        field = np.zeros((8, 8, 8))
        peaks = find_peaks(field, n=2)
        assert peaks == []


class TestMeasureSeparation:
    def test_known_distance(self):
        field = np.zeros((16, 16, 16))
        field[4, 8, 8] = 10.0
        field[12, 8, 8] = 8.0
        dist = measure_separation(field, min_peak_separation=3)
        assert dist == pytest.approx(8.0)

    def test_single_peak(self):
        field = np.zeros((16, 16, 16))
        field[8, 8, 8] = 10.0
        dist = measure_separation(field)
        assert dist == 0.0


class TestMeasureForce:
    def test_uniform_chi_zero_force(self):
        chi = np.full((16, 16, 16), 19.0)
        force = measure_force(chi, chi, (8, 8, 8))
        np.testing.assert_allclose(force, 0.0, atol=1e-10)

    def test_gradient_chi(self):
        chi = np.full((16, 16, 16), 19.0)
        # Create gradient along x
        for i in range(16):
            chi[i, :, :] = 19.0 - 0.1 * i
        force = measure_force(chi, chi, (8, 8, 8))
        assert force[0] > 0  # Force toward lower chi (positive x)


class TestConfinementProxy:
    def test_uniform_zero(self):
        chi = np.full((16, 16, 16), 19.0)
        result = confinement_proxy(chi, (4, 8, 8), (12, 8, 8))
        assert result["mean_depression"] == pytest.approx(0.0)
        assert result["distance"] == pytest.approx(8.0)

    def test_with_well(self):
        chi = np.full((16, 16, 16), 19.0)
        chi[8, 8, 8] = 15.0
        result = confinement_proxy(chi, (4, 8, 8), (12, 8, 8))
        assert result["line_integral"] > 0


class TestWeakParityAsymmetry:
    def test_symmetric_field(self):
        chi = np.full((16, 16, 16), 19.0)
        result = weak_parity_asymmetry(chi, axis=0)
        assert result["asymmetry"] == pytest.approx(0.0, abs=1e-10)

    def test_invalid_axis(self):
        chi = np.full((16, 16, 16), 19.0)
        with pytest.raises(ValueError):
            weak_parity_asymmetry(chi, axis=3)

    def test_2d_raises(self):
        chi = np.full((16, 16), 19.0)
        with pytest.raises(ValueError):
            weak_parity_asymmetry(chi)


class TestFitPowerLaw:
    def test_perfect_inverse(self):
        r = np.arange(2, 20, dtype=float)
        profile = 1.0 / r
        exp, r_sq = fit_power_law(r, profile, r_min=2.0)
        assert exp == pytest.approx(-1.0, abs=0.01)
        assert r_sq > 0.99

    def test_too_few_points(self):
        exp, r_sq = fit_power_law(np.array([1.0]), np.array([1.0]))
        assert np.isnan(exp)
        assert r_sq == 0.0


class TestMomentumDensity:
    def test_real_field_zero_j(self):
        psi_r = np.ones((8, 8, 8), dtype=np.float32)
        psi_i = np.zeros((8, 8, 8), dtype=np.float32)
        result = momentum_density(psi_r, psi_i)
        np.testing.assert_allclose(result["j_total"], 0.0, atol=1e-6)

    def test_4d_multicolor(self):
        psi_r = np.ones((3, 8, 8, 8), dtype=np.float32)
        psi_i = np.zeros((3, 8, 8, 8), dtype=np.float32)
        result = momentum_density(psi_r, psi_i)
        assert result["j_x"].shape == (8, 8, 8)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            momentum_density(np.ones((8, 8, 8)), np.ones((4, 4, 4)))


# ── Energy ──────────────────────────────────────────────────────────


class TestTotalEnergy:
    def test_positive(self):
        N = 8
        psi = np.random.default_rng(0).standard_normal((N, N, N)).astype(np.float32) * 0.1
        psi_prev = psi.copy()
        chi = np.full((N, N, N), 19.0, dtype=np.float32)
        e = total_energy(psi, psi_prev, chi, dt=0.02)
        assert e > 0


class TestEnergyConservationDrift:
    def test_no_drift(self):
        assert energy_conservation_drift(100.0, 100.0) == 0.0

    def test_known_drift(self):
        assert energy_conservation_drift(100.0, 101.0) == pytest.approx(1.0)

    def test_zero_initial(self):
        assert energy_conservation_drift(0.0, 0.0) == 0.0


class TestFluidFields:
    def test_basic(self):
        N = 8
        rng = np.random.default_rng(42)
        psi_r = rng.standard_normal((N, N, N)).astype(np.float32) * 0.1
        psi_r_prev = psi_r + rng.standard_normal((N, N, N)).astype(np.float32) * 0.001
        chi = np.full((N, N, N), 19.0, dtype=np.float32)
        result = fluid_fields(psi_r, psi_r_prev, chi, dt=0.02)
        assert result["epsilon_mean"] > 0
        assert "v_rms" in result
        assert result["vx"].shape == (N, N, N)

    def test_with_complex(self):
        N = 8
        rng = np.random.default_rng(42)
        psi_r = rng.standard_normal((N, N, N)).astype(np.float32) * 0.1
        psi_i = rng.standard_normal((N, N, N)).astype(np.float32) * 0.1
        chi = np.full((N, N, N), 19.0, dtype=np.float32)
        result = fluid_fields(psi_r, psi_r, chi, dt=0.02, psi_i=psi_i, psi_i_prev=psi_i)
        assert result["epsilon_mean"] > 0


class TestContinuityResidual:
    def test_static_nan(self):
        N = 8
        eps = np.ones((N, N, N))
        gx = gy = gz = np.zeros((N, N, N))
        res = continuity_residual(eps, eps, gx, gy, gz, dt=0.02)
        # Static → deps_dt = 0 → normalization = 0 → nan
        assert np.isnan(res)

    def test_finite(self):
        N = 8
        rng = np.random.default_rng(0)
        eps0 = rng.random((N, N, N)) + 1.0
        eps1 = eps0 + rng.standard_normal((N, N, N)) * 0.01
        gx = rng.standard_normal((N, N, N)) * 0.1
        gy = rng.standard_normal((N, N, N)) * 0.1
        gz = rng.standard_normal((N, N, N)) * 0.1
        res = continuity_residual(eps0, eps1, gx, gy, gz, dt=0.02)
        assert np.isfinite(res)
        assert res > 0
