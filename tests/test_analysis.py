"""Tests for lfm.analysis subpackage."""

import numpy as np
import pytest

from lfm.analysis import (
    chi_statistics,
    compute_metrics,
    confinement_proxy,
    count_clusters,
    energy_components,
    energy_conservation_drift,
    interior_mask,
    momentum_density,
    total_energy,
    void_fraction,
    weak_parity_asymmetry,
    well_fraction,
)
from lfm.constants import CHI0

N = 16  # small grid for fast tests


def _make_fields(n=N, amplitude=1.0, with_imag=False):
    """Create simple test fields: Gaussian blob in uniform chi."""
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
        assert {"min", "max", "mean", "std"} <= set(stats.keys())

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
    assert hasattr(lfm, "momentum_density")
    assert hasattr(lfm, "weak_parity_asymmetry")
    assert hasattr(lfm, "confinement_proxy")


class TestWeakStrongHelpers:
    def test_momentum_density_zero_for_constant_phase(self):
        n = 10
        pr = np.ones((n, n, n), dtype=np.float32)
        pi = np.zeros((n, n, n), dtype=np.float32)
        j = momentum_density(pr, pi)
        assert np.allclose(j["j_x"], 0.0)
        assert np.allclose(j["j_y"], 0.0)
        assert np.allclose(j["j_z"], 0.0)
        assert np.allclose(j["j_total"], 0.0)

    def test_momentum_density_color_shape(self):
        n = 8
        pr = np.random.default_rng(0).standard_normal((3, n, n, n)).astype(np.float32)
        pi = np.random.default_rng(1).standard_normal((3, n, n, n)).astype(np.float32)
        j = momentum_density(pr, pi)
        assert j["j_total"].shape == (n, n, n)

    def test_weak_parity_asymmetry_balanced(self):
        n = 12
        chi = np.full((n, n, n), CHI0, dtype=np.float32)
        # symmetric dip around center
        chi[5, 6, 6] = CHI0 - 1.0
        chi[7, 6, 6] = CHI0 - 1.0
        a = weak_parity_asymmetry(chi, axis=0)
        assert abs(a["asymmetry"]) < 1e-6

    def test_weak_parity_asymmetry_signed(self):
        n = 12
        chi = np.full((n, n, n), CHI0, dtype=np.float32)
        chi[8, 6, 6] = CHI0 - 2.0  # +x side dip
        chi[4, 6, 6] = CHI0 - 0.5  # -x side dip
        a = weak_parity_asymmetry(chi, axis=0)
        assert a["asymmetry"] > 0

    def test_confinement_proxy_monotonic_with_distance(self):
        n = 20
        chi = np.full((n, n, n), CHI0, dtype=np.float32)
        # Build a low-chi tube along x at y=z=10
        chi[2:18, 10, 10] = CHI0 - 2.0
        short = confinement_proxy(chi, (6, 10, 10), (10, 10, 10), samples=32)
        long = confinement_proxy(chi, (6, 10, 10), (16, 10, 10), samples=64)
        assert long["line_integral"] > short["line_integral"]


# ---------------------------------------------------------------------------
# P026: rotation_curve_fit
# ---------------------------------------------------------------------------


class TestRotationCurveFit:
    def test_returns_required_keys(self):
        import numpy as np

        from lfm.analysis.observables import rotation_curve_fit

        row = {"r_kpc": np.linspace(1, 10, 8), "v_obs_kms": np.full(8, 100.0)}
        r = np.linspace(1, 20, 15)
        v = np.ones(15) * 50.0
        result = rotation_curve_fit(row, r, v)
        for k in ("tau_best", "chi2", "tau_grid", "chi2_grid", "r_kpc", "v_obs", "v_sim_best"):
            assert k in result, f"missing key {k!r}"

    def test_best_chi2_is_minimum(self):
        import numpy as np

        from lfm.analysis.observables import rotation_curve_fit

        row = {"r_kpc": np.linspace(1, 10, 8), "v_obs_kms": np.ones(8) * 80.0}
        r = np.linspace(1, 20, 15)
        v = np.ones(15)
        result = rotation_curve_fit(row, r, v, n_tau=10)
        assert result["chi2"] == pytest.approx(float(result["chi2_grid"].min()), rel=1e-4)

    def test_raises_on_zero_radii(self):
        import numpy as np

        from lfm.analysis.observables import rotation_curve_fit

        row = {"r_kpc": np.zeros(5), "v_obs_kms": np.ones(5)}
        with pytest.raises(ValueError):
            rotation_curve_fit(row, np.ones(5), np.ones(5))


# ---------------------------------------------------------------------------
# P027: collider_event_display
# ---------------------------------------------------------------------------


class TestColliderEventDisplay:
    def test_returns_string(self):
        from lfm.analysis.tracker import collider_event_display

        result = collider_event_display({"events": []})
        assert isinstance(result, str)

    def test_empty_events(self):
        from lfm.analysis.tracker import collider_event_display

        result = collider_event_display({"events": []})
        assert "no events" in result

    def test_event_in_output(self):
        from lfm.analysis.tracker import collider_event_display

        events = [
            {"time_step": 100, "type": "MERGE", "particle_a": 0, "particle_b": 1, "r_min": 0.5}
        ]
        result = collider_event_display({"events": events, "n_particles": 2, "total_steps": 500})
        assert "MERGE" in result
        assert "p0<->p1" in result

    def test_score_shown(self):
        from lfm.analysis.tracker import collider_event_display

        result = collider_event_display({"events": [], "score": 3.14})
        assert "3.14" in result

    def test_width_respected(self):
        from lfm.analysis.tracker import collider_event_display

        result = collider_event_display({"events": []}, width=50)
        for line in result.splitlines():
            assert len(line) <= 52  # allow slight overflow for box chars


# ---------------------------------------------------------------------------
# P029: sparc_load + list_sparc_galaxies
# ---------------------------------------------------------------------------


class TestSparcLoad:
    def test_no_arg_returns_five_galaxies(self):
        from lfm.analysis.sparc import sparc_load

        data = sparc_load()
        assert len(data) == 5

    def test_known_names(self):
        from lfm.analysis.sparc import list_sparc_galaxies

        names = list_sparc_galaxies()
        assert set(names) == {"NGC6503", "NGC3198", "DDO154", "IC2574", "UGC2885"}

    def test_single_name_lookup(self):
        from lfm.analysis.sparc import sparc_load

        data = sparc_load("NGC6503")
        assert "NGC6503" in data
        row = data["NGC6503"]
        assert "r_kpc" in row and "v_obs_kms" in row and "v_err_kms" in row

    def test_data_shapes_consistent(self):
        from lfm.analysis.sparc import sparc_load

        for name, row in sparc_load().items():
            assert len(row["r_kpc"]) == len(row["v_obs_kms"]) == len(row["v_err_kms"]), (
                f"{name}: inconsistent array lengths"
            )

    def test_missing_dir_raises(self, tmp_path):
        from lfm.analysis.sparc import sparc_load

        with pytest.raises(FileNotFoundError):
            sparc_load(tmp_path / "nonexistent")

    def test_empty_dir_returns_empty(self, tmp_path):
        from lfm.analysis.sparc import sparc_load

        result = sparc_load(tmp_path)
        assert result == {}

    def test_list_sparc_sorted(self):
        from lfm.analysis.sparc import list_sparc_galaxies

        names = list_sparc_galaxies()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# P030: disk_positions + initialize_disk b_cells
# ---------------------------------------------------------------------------


class TestBCells:
    def test_disk_positions_default_b_cells_zero(self):
        import numpy as np

        from lfm.fields.arrangements import disk_positions

        pos = disk_positions(64, 10, seed=0)
        center = 32.0
        assert np.allclose(pos[:, 2], center)  # plane_axis=2 default

    def test_b_cells_shifts_plane_axis(self):
        import numpy as np

        from lfm.fields.arrangements import disk_positions

        shift = 5.0
        pos = disk_positions(64, 10, seed=0, b_cells=shift)
        center = 32.0
        assert np.allclose(pos[:, 2], center + shift)

    def test_b_cells_negative(self):
        import numpy as np

        from lfm.fields.arrangements import disk_positions

        pos = disk_positions(64, 10, seed=0, b_cells=-3.0)
        assert np.allclose(pos[:, 2], 32.0 - 3.0)

    def test_b_cells_plane_axis_0(self):
        import numpy as np

        from lfm.fields.arrangements import disk_positions

        pos = disk_positions(64, 10, plane_axis=0, seed=0, b_cells=4.0)
        assert np.allclose(pos[:, 0], 32.0 + 4.0)
