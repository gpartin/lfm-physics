"""Tests for lfm.analysis subpackage."""

import numpy as np
import pytest

from lfm.constants import CHI0, WELL_THRESHOLD, VOID_THRESHOLD
from lfm.analysis import (
    chi_statistics,
    compute_metrics,
    count_clusters,
    energy_components,
    energy_conservation_drift,
    interior_mask,
    total_energy,
    void_fraction,
    well_fraction,
)


N = 16  # small grid for fast tests


def _make_fields(n=N, amplitude=1.0, with_imag=False):
    """Create simple test fields: Gaussian blob in uniform chi."""
    rng = np.random.default_rng(42)
    x = np.arange(n, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    center = n / 2.0
    r2 = (X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2
    psi_r = (amplitude * np.exp(-r2 / 8.0)).astype(np.float32)
    # Previous step: slightly different for finite time derivative
    psi_r_prev = (psi_r * 0.99).astype(np.float32)
    chi = np.full((n, n, n), CHI0, dtype=np.float32)
    psi_i = None
    psi_i_prev = None
    if with_imag:
        psi_i = (amplitude * 0.5 * np.exp(-r2 / 8.0)).astype(np.float32)
        psi_i_prev = (psi_i * 0.99).astype(np.float32)
    return psi_r, psi_r_prev, chi, psi_i, psi_i_prev


# ──── energy.py ────


class TestEnergyComponents:
    def test_returns_three_arrays(self):
        pr, prp, chi, _, _ = _make_fields()
        T, G, V = energy_components(pr, prp, chi, dt=0.02)
        assert T.shape == (N, N, N)
        assert G.shape == (N, N, N)
        assert V.shape == (N, N, N)

    def test_all_non_negative(self):
        pr, prp, chi, _, _ = _make_fields()
        T, G, V = energy_components(pr, prp, chi, dt=0.02)
        assert np.all(T >= -1e-10)
        assert np.all(G >= -1e-10)
        assert np.all(V >= -1e-10)

    def test_potential_grows_with_chi(self):
        pr, prp, chi_low, _, _ = _make_fields()
        chi_high = chi_low * 2
        _, _, V_low = energy_components(pr, prp, chi_low, dt=0.02)
        _, _, V_high = energy_components(pr, prp, chi_high, dt=0.02)
        assert np.sum(V_high) > np.sum(V_low)

    def test_kinetic_zero_for_static(self):
        pr, _, chi, _, _ = _make_fields()
        T, _, _ = energy_components(pr, pr, chi, dt=0.02)  # same prev
        assert np.allclose(T, 0.0)

    def test_complex_field(self):
        pr, prp, chi, pi, pip = _make_fields(with_imag=True)
        T, G, V = energy_components(pr, prp, chi, dt=0.02, psi_i=pi, psi_i_prev=pip)
        # Complex field should have more energy than real-only
        T_real, G_real, V_real = energy_components(pr, prp, chi, dt=0.02)
        assert np.sum(T) >= np.sum(T_real) - 1e-10
        assert np.sum(V) > np.sum(V_real)

    def test_color_field(self):
        n = 12
        pr = np.random.default_rng(1).standard_normal((3, n, n, n)).astype(np.float32) * 0.1
        prp = pr * 0.99
        chi = np.full((n, n, n), CHI0, dtype=np.float32)
        T, G, V = energy_components(pr, prp, chi, dt=0.02)
        assert T.shape == (n, n, n)


class TestTotalEnergy:
    def test_scalar_result(self):
        pr, prp, chi, _, _ = _make_fields()
        E = total_energy(pr, prp, chi, dt=0.02)
        assert isinstance(E, float)
        assert E > 0

    def test_matches_sum_of_components(self):
        pr, prp, chi, _, _ = _make_fields()
        E = total_energy(pr, prp, chi, dt=0.02)
        T, G, V = energy_components(pr, prp, chi, dt=0.02)
        assert np.isclose(E, np.sum(T + G + V), rtol=1e-10)


class TestEnergyConservationDrift:
    def test_zero_drift(self):
        assert energy_conservation_drift(100.0, 100.0) == 0.0

    def test_one_percent(self):
        assert np.isclose(energy_conservation_drift(100.0, 101.0), 1.0)

    def test_zero_initial(self):
        assert energy_conservation_drift(0.0, 5.0) == 0.0

    def test_negative_drift(self):
        assert np.isclose(energy_conservation_drift(200.0, 196.0), 2.0)


# ──── structure.py ────


class TestChiStatistics:
    def test_keys(self):
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        stats = chi_statistics(chi)
        assert set(stats.keys()) == {"min", "max", "mean", "std"}

    def test_uniform_chi(self):
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        stats = chi_statistics(chi)
        assert np.isclose(stats["min"], CHI0)
        assert np.isclose(stats["max"], CHI0)
        assert np.isclose(stats["mean"], CHI0)
        assert np.isclose(stats["std"], 0.0, atol=1e-6)

    def test_with_mask(self):
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        chi[0, 0, 0] = 5.0  # outlier at corner
        mask = interior_mask(N, boundary_fraction=0.3)
        stats_masked = chi_statistics(chi, mask)
        stats_full = chi_statistics(chi)
        # The masked version should NOT see the corner outlier
        assert stats_full["min"] < stats_masked["min"]


class TestWellFraction:
    def test_all_vacuum(self):
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        assert well_fraction(chi) == 0.0

    def test_all_well(self):
        chi = np.full((N, N, N), 10.0, dtype=np.float32)
        assert well_fraction(chi) == 1.0

    def test_half(self):
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        chi[: N // 2] = 10.0
        frac = well_fraction(chi)
        assert np.isclose(frac, 0.5)


class TestVoidFraction:
    def test_all_vacuum(self):
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        assert void_fraction(chi) == 1.0

    def test_all_well(self):
        chi = np.full((N, N, N), 10.0, dtype=np.float32)
        assert void_fraction(chi) == 0.0


class TestCountClusters:
    def test_single_blob(self):
        field = np.zeros((N, N, N), dtype=np.float32)
        field[6:10, 6:10, 6:10] = 10.0  # one blob
        n = count_clusters(field, threshold_percentile=50)
        assert n == 1

    def test_two_separated_blobs(self):
        field = np.zeros((N, N, N), dtype=np.float32)
        field[1:3, 1:3, 1:3] = 10.0
        field[12:14, 12:14, 12:14] = 10.0
        n = count_clusters(field, threshold_percentile=50)
        assert n == 2

    def test_uniform_field(self):
        field = np.ones((N, N, N), dtype=np.float32)
        n = count_clusters(field, threshold_percentile=50)
        # Uniform field: all values equal → none strictly above percentile → 0
        assert n == 0


class TestInteriorMask:
    def test_shape(self):
        mask = interior_mask(N, 0.3)
        assert mask.shape == (N, N, N)
        assert mask.dtype == bool

    def test_center_is_interior(self):
        mask = interior_mask(N, 0.3)
        c = N // 2
        assert mask[c, c, c]

    def test_corner_is_boundary(self):
        mask = interior_mask(N, 0.3)
        assert not mask[0, 0, 0]

    def test_larger_boundary_means_smaller_interior(self):
        m1 = interior_mask(N, 0.2)
        m2 = interior_mask(N, 0.5)
        assert np.sum(m1) > np.sum(m2)


# ──── metrics.py ────


class TestComputeMetrics:
    def test_returns_dict(self):
        pr, prp, chi, _, _ = _make_fields()
        m = compute_metrics(pr, prp, chi, dt=0.02)
        assert isinstance(m, dict)

    def test_expected_keys(self):
        pr, prp, chi, _, _ = _make_fields()
        m = compute_metrics(pr, prp, chi, dt=0.02)
        expected = {
            "energy_kinetic",
            "energy_gradient",
            "energy_potential",
            "energy_total",
            "chi_min",
            "chi_max",
            "chi_mean",
            "chi_std",
            "well_fraction",
            "void_fraction",
            "n_clusters",
            "psi_sq_total",
        }
        assert set(m.keys()) == expected

    def test_energy_total_consistent(self):
        pr, prp, chi, _, _ = _make_fields()
        m = compute_metrics(pr, prp, chi, dt=0.02)
        expected = m["energy_kinetic"] + m["energy_gradient"] + m["energy_potential"]
        assert np.isclose(m["energy_total"], expected, rtol=1e-10)

    def test_complex_field(self):
        pr, prp, chi, pi, pip = _make_fields(with_imag=True)
        m = compute_metrics(pr, prp, chi, dt=0.02, psi_i=pi, psi_i_prev=pip)
        assert m["energy_total"] > 0
        assert m["psi_sq_total"] > 0

    def test_with_interior_mask(self):
        pr, prp, chi, _, _ = _make_fields()
        mask = interior_mask(N)
        m = compute_metrics(pr, prp, chi, dt=0.02, interior_mask=mask)
        assert m["chi_min"] >= 0


# ──── Top-level import check ────


def test_top_level_imports():
    """Verify analysis functions are available from lfm directly."""
    import lfm

    assert hasattr(lfm, "energy_components")
    assert hasattr(lfm, "total_energy")
    assert hasattr(lfm, "energy_conservation_drift")
    assert hasattr(lfm, "chi_statistics")
    assert hasattr(lfm, "well_fraction")
    assert hasattr(lfm, "void_fraction")
    assert hasattr(lfm, "count_clusters")
    assert hasattr(lfm, "interior_mask")
    assert hasattr(lfm, "compute_metrics")
