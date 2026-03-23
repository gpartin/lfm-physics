"""Tests for v1.1.0 features.

Covers:
* Simulation.run_with_snapshots()
* lfm.viz.project_field / plot_projection
* lfm.viz.spacetime_diagram
* lfm.viz.animate_slice / animate_three_slices
* lfm.analysis.cosmology  (correlation_function, matter_power_spectrum,
                           halo_mass_function, void_statistics)
* lfm.analysis.grav_waves (gravitational_wave_strain, gw_quadrupole, gw_power)
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import CHI0
from lfm.simulation import Simulation

N = 16  # tiny grid for fast tests
DT = 0.02


def _small_config(**kwargs):
    defaults = dict(
        grid_size=N,
        field_level=FieldLevel.REAL,
        boundary_type=BoundaryType.FROZEN,
        report_interval=100,
        dt=DT,
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def _make_chi_field(n=N, val=CHI0, dip_center=True):
    """Uniform χ field with optional central dip."""
    chi = np.full((n, n, n), val, dtype=np.float32)
    if dip_center:
        c = n // 2
        chi[c - 1 : c + 2, c - 1 : c + 2, c - 1 : c + 2] = val - 4.0
    return chi


def _make_energy_density(n=N, amplitude=1.0):
    x = np.arange(n, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - n / 2) ** 2 + (Y - n / 2) ** 2 + (Z - n / 2) ** 2
    return (amplitude * np.exp(-r2 / 8.0)).astype(np.float32)


def _make_snapshots(n_snaps=5, with_energy=False, n=N):
    """Light-weight snapshot list (no simulation needed)."""
    snaps = []
    chi = _make_chi_field(n=n)
    ed = _make_energy_density(n=n)
    for i in range(n_snaps):
        sn: dict = {"step": i * 10, "chi": chi.copy()}
        if with_energy:
            sn["energy_density"] = (ed * (1.0 + 0.1 * i)).copy()
        snaps.append(sn)
    return snaps


# ═══════════════════════════════════════════════════════════════════════
# Simulation.run_with_snapshots
# ═══════════════════════════════════════════════════════════════════════


class TestRunWithSnapshots:
    def _sim(self):
        sim = Simulation(_small_config())
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0)
        return sim

    def test_returns_correct_count_exact_multiple(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(200, snapshot_every=100)
        assert len(snaps) == 2

    def test_returns_correct_count_with_remainder(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(250, snapshot_every=100)
        # 2 full blocks + 1 remainder = 3
        assert len(snaps) == 3

    def test_default_field_is_chi(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(100, snapshot_every=100)
        assert "chi" in snaps[0]
        assert "psi_real" not in snaps[0]

    def test_requested_fields_present(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(
            100, snapshot_every=100, fields=["chi", "psi_real", "energy_density"]
        )
        snap = snaps[-1]
        assert "chi" in snap
        assert "psi_real" in snap
        assert "energy_density" in snap

    def test_step_key_increases(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(400, snapshot_every=100)
        steps = [s["step"] for s in snaps]
        assert steps == sorted(steps)
        assert steps[-1] == 400

    def test_chi_shape_correct(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(100, snapshot_every=100)
        assert snaps[0]["chi"].shape == (N, N, N)

    def test_snap_is_copy(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(200, snapshot_every=200)
        old_val = snaps[0]["chi"][0, 0, 0]
        # Mutating the live field should not change the snapshot
        sim.chi[0, 0, 0] = 999.0
        assert snaps[0]["chi"][0, 0, 0] == pytest.approx(old_val)

    def test_psi_imag_absent_for_real_field(self):
        sim = self._sim()
        snaps = sim.run_with_snapshots(
            100, snapshot_every=100, fields=["psi_imag"]
        )
        # Real simulation → psi_imag is None → key absent
        assert "psi_imag" not in snaps[0]

    def test_psi_imag_present_for_complex_field(self):
        sim = Simulation(_small_config(field_level=FieldLevel.COMPLEX))
        sim.place_soliton((N // 2, N // 2, N // 2), amplitude=3.0)
        snaps = sim.run_with_snapshots(
            100, snapshot_every=100, fields=["psi_imag"]
        )
        assert "psi_imag" in snaps[0]


# ═══════════════════════════════════════════════════════════════════════
# lfm.viz.projection
# ═══════════════════════════════════════════════════════════════════════


class TestProjectField:
    def test_sum_reduces_axis(self):
        from lfm.viz.projection import project_field

        field = np.ones((N, N, N), dtype=np.float32)
        proj = project_field(field, axis=2, method="sum")
        assert proj.shape == (N, N)
        assert np.allclose(proj, N)  # sum of N ones

    def test_mean_reduces_axis(self):
        from lfm.viz.projection import project_field

        field = np.ones((N, N, N), dtype=np.float32) * 7.0
        proj = project_field(field, axis=0, method="mean")
        assert proj.shape == (N, N)
        assert np.allclose(proj, 7.0)

    def test_max_returns_maximum(self):
        from lfm.viz.projection import project_field

        field = np.zeros((N, N, N), dtype=np.float32)
        field[N // 2, :, :] = 5.0
        proj = project_field(field, axis=0, method="max")
        assert np.allclose(proj, 5.0)

    def test_invalid_method_raises(self):
        from lfm.viz.projection import project_field

        with pytest.raises(ValueError, match="Unknown method"):
            project_field(np.ones((N, N, N)), method="bad")

    def test_all_axes(self):
        from lfm.viz.projection import project_field

        field = np.random.rand(N, N, N).astype(np.float32)
        for ax in (0, 1, 2):
            proj = project_field(field, axis=ax)
            assert proj.shape == (N, N)


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
class TestPlotProjection:
    def test_returns_figure_axes(self):
        from lfm.viz.projection import plot_projection

        field = _make_chi_field()
        fig, ax = plot_projection(field, axis=2, log=False)
        assert fig is not None
        assert ax is not None

    def test_log_scale(self):
        from lfm.viz.projection import plot_projection

        field = _make_chi_field()
        fig, ax = plot_projection(field, axis=2, log=True)
        assert fig is not None


# ═══════════════════════════════════════════════════════════════════════
# lfm.viz.spacetime
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
class TestSpacetimeDiagram:
    def test_returns_figure_axes(self):
        from lfm.viz.spacetime import spacetime_diagram

        snaps = _make_snapshots(8)
        fig, ax = spacetime_diagram(snaps, field="chi")
        assert fig is not None
        assert ax is not None

    def test_empty_snapshots_raises(self):
        from lfm.viz.spacetime import spacetime_diagram

        with pytest.raises(ValueError, match="empty"):
            spacetime_diagram([], field="chi")

    def test_missing_field_raises(self):
        from lfm.viz.spacetime import spacetime_diagram

        snaps = _make_snapshots(4)
        with pytest.raises(KeyError):
            spacetime_diagram(snaps, field="nonexistent_field")


# ═══════════════════════════════════════════════════════════════════════
# lfm.viz.animation
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
class TestAnimateSlice:
    def test_animate_slice_returns_animation(self):
        from lfm.viz.animation import animate_slice

        snaps = _make_snapshots(5)
        anim = animate_slice(snaps, field="chi")
        # FuncAnimation object
        assert anim is not None

    def test_animate_three_slices_returns_animation(self):
        from lfm.viz.animation import animate_three_slices

        snaps = _make_snapshots(5)
        anim = animate_three_slices(snaps, field="chi")
        assert anim is not None


# ═══════════════════════════════════════════════════════════════════════
# lfm.analysis.grav_waves
# ═══════════════════════════════════════════════════════════════════════


class TestGravitationalWaveStrain:
    def test_vacuum_gives_zero(self):
        from lfm.analysis.grav_waves import gravitational_wave_strain

        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        h = gravitational_wave_strain(chi)
        assert np.allclose(h, 0.0)

    def test_below_vacuum_negative(self):
        from lfm.analysis.grav_waves import gravitational_wave_strain

        chi = np.full((N, N, N), CHI0 - 2.0, dtype=np.float32)
        h = gravitational_wave_strain(chi)
        assert np.all(h < 0.0)

    def test_shape_preserved(self):
        from lfm.analysis.grav_waves import gravitational_wave_strain

        chi = _make_chi_field()
        h = gravitational_wave_strain(chi)
        assert h.shape == (N, N, N)

    def test_scale_correct(self):
        from lfm.analysis.grav_waves import gravitational_wave_strain

        chi = np.full((N, N, N), CHI0 * 2.0, dtype=np.float32)
        h = gravitational_wave_strain(chi)
        assert np.allclose(h, 1.0)


class TestGWQuadrupole:
    def test_returns_3x3(self):
        from lfm.analysis.grav_waves import gw_quadrupole

        ed = _make_energy_density()
        I = gw_quadrupole(ed)
        assert I.shape == (3, 3)

    def test_is_symmetric(self):
        from lfm.analysis.grav_waves import gw_quadrupole

        ed = _make_energy_density()
        I = gw_quadrupole(ed)
        assert np.allclose(I, I.T)

    def test_trace_is_zero_by_construction(self):
        """Reduced quadrupole has trace ∝ Σ(x_i² - r²/3) = r² - r² = 0."""
        from lfm.analysis.grav_waves import gw_quadrupole

        ed = _make_energy_density()
        I = gw_quadrupole(ed)
        assert abs(np.trace(I)) < 1e-6 * abs(I).max() + 1e-10

    def test_explicit_center(self):
        from lfm.analysis.grav_waves import gw_quadrupole

        ed = _make_energy_density()
        c = float(N // 2)
        I = gw_quadrupole(ed, center=(c, c, c))
        assert I.shape == (3, 3)

    def test_spherical_source_nearly_zero(self):
        """Perfectly spherical source has I_ij ≈ 0 (no quadrupole moment)."""
        from lfm.analysis.grav_waves import gw_quadrupole

        x = np.arange(N, dtype=np.float64)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        c = N / 2.0
        r2 = (X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2
        ed = np.exp(-r2 / 4.0).astype(np.float32)
        I = gw_quadrupole(ed, center=(c, c, c))
        # For spherically symmetric source the off-diagonal → 0, diagonal << mass
        assert abs(I[0, 1]) < 1e-6
        assert abs(I[0, 2]) < 1e-6


class TestGWPower:
    def test_raises_on_too_few_snapshots(self):
        from lfm.analysis.grav_waves import gw_power

        snaps = _make_snapshots(2, with_energy=True)
        with pytest.raises(ValueError, match="3"):
            gw_power(snaps, field="energy_density")

    def test_raises_missing_field(self):
        from lfm.analysis.grav_waves import gw_power

        snaps = _make_snapshots(5)  # only "chi"
        with pytest.raises(KeyError):
            gw_power(snaps, field="energy_density")

    def test_returns_dict_keys(self):
        from lfm.analysis.grav_waves import gw_power

        snaps = _make_snapshots(6, with_energy=True)
        result = gw_power(snaps, field="energy_density", dt=0.02)
        assert "t" in result
        assert "luminosity" in result
        assert "I_tensor" in result

    def test_luminosity_non_negative(self):
        from lfm.analysis.grav_waves import gw_power

        snaps = _make_snapshots(8, with_energy=True)
        result = gw_power(snaps, field="energy_density", dt=0.02)
        assert np.all(result["luminosity"] >= 0.0)

    def test_I_tensor_shape(self):
        from lfm.analysis.grav_waves import gw_power

        snaps = _make_snapshots(8, with_energy=True)
        result = gw_power(snaps, field="energy_density", dt=0.02)
        n_frames = len(snaps)
        assert result["I_tensor"].shape == (n_frames, 3, 3)

    def test_t_array_length(self):
        from lfm.analysis.grav_waves import gw_power

        snaps = _make_snapshots(7, with_energy=True)
        result = gw_power(snaps, field="energy_density", dt=0.5)
        # luminosity is returned for every frame
        assert len(result["luminosity"]) == len(snaps)
        assert len(result["t"]) == len(snaps)


# ═══════════════════════════════════════════════════════════════════════
# lfm.analysis.cosmology
# ═══════════════════════════════════════════════════════════════════════


class TestCorrelationFunction:
    def test_returns_dict_keys(self):
        from lfm.analysis.cosmology import correlation_function

        field = _make_chi_field()
        result = correlation_function(field)
        assert "r" in result
        assert "xi" in result
        assert "n_pairs" in result

    def test_r_monotonically_increasing(self):
        from lfm.analysis.cosmology import correlation_function

        field = _make_chi_field()
        result = correlation_function(field)
        r = result["r"]
        assert np.all(np.diff(r) > 0)

    def test_xi_at_zero_lag_positive(self):
        from lfm.analysis.cosmology import correlation_function

        field = _make_chi_field()
        result = correlation_function(field)
        # ξ(r→0) for a structured field should be positive
        assert result["xi"][0] > 0

    def test_uniform_field_near_zero(self):
        from lfm.analysis.cosmology import correlation_function

        field = np.full((N, N, N), CHI0, dtype=np.float32)
        result = correlation_function(field)
        # Uniform → δ = 0 everywhere → ξ ≈ 0
        assert np.allclose(result["xi"], 0.0, atol=1e-9)


class TestMatterPowerSpectrum:
    def test_returns_dict_keys(self):
        from lfm.analysis.cosmology import matter_power_spectrum

        ed = _make_energy_density()
        result = matter_power_spectrum(ed)
        assert "k" in result
        assert "pk" in result
        assert "pk_raw" in result
        assert "n_modes" in result

    def test_k_positive(self):
        from lfm.analysis.cosmology import matter_power_spectrum

        ed = _make_energy_density()
        r = matter_power_spectrum(ed)
        assert np.all(r["k"] > 0)

    def test_pk_non_negative(self):
        from lfm.analysis.cosmology import matter_power_spectrum

        ed = _make_energy_density()
        r = matter_power_spectrum(ed)
        assert np.all(r["pk"] >= 0)

    def test_n_modes_positive(self):
        from lfm.analysis.cosmology import matter_power_spectrum

        ed = _make_energy_density()
        r = matter_power_spectrum(ed)
        # Some bins may be empty on a small grid; total must be positive
        assert r["n_modes"].sum() > 0


class TestHaloMassFunction:
    def test_returns_dict_keys(self):
        from lfm.analysis.cosmology import halo_mass_function

        chi = _make_chi_field(dip_center=True)
        ed = _make_energy_density()
        result = halo_mass_function(chi, ed)
        assert "m_bins" in result
        assert "dn_dlnm" in result
        assert "n_halos" in result
        assert "masses" in result

    def test_finds_at_least_one_halo(self):
        from lfm.analysis.cosmology import halo_mass_function

        # Create a clear deep dip to ensure a halo is detected
        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        c = N // 2
        chi[c - 2 : c + 3, c - 2 : c + 3, c - 2 : c + 3] = 13.0  # deep well
        ed = _make_energy_density()
        result = halo_mass_function(chi, ed, chi_threshold=14.0)
        assert result["n_halos"] >= 1

    def test_no_halos_above_threshold(self):
        from lfm.analysis.cosmology import halo_mass_function

        chi = np.full((N, N, N), CHI0, dtype=np.float32)  # uniform, all ≥ chi0
        ed = _make_energy_density()
        # threshold below CHI0 → no halos
        result = halo_mass_function(chi, ed, chi_threshold=CHI0 - 20.0)
        assert result["n_halos"] == 0


class TestVoidStatistics:
    def test_returns_dict_keys(self):
        from lfm.analysis.cosmology import void_statistics

        chi = _make_chi_field()
        result = void_statistics(chi)
        assert "r_bins" in result
        assert "dn_dr" in result
        assert "n_voids" in result
        assert "sizes" in result

    def test_no_voids_when_all_below_min(self):
        from lfm.analysis.cosmology import void_statistics

        chi = np.full((N, N, N), 10.0, dtype=np.float32)  # all below any chi_min
        result = void_statistics(chi, chi_min=100.0)
        assert result["n_voids"] == 0

    def test_finds_void_region(self):
        from lfm.analysis.cosmology import void_statistics

        chi = np.full((N, N, N), CHI0, dtype=np.float32)
        # A large void-like region at chi ~= chi0
        result = void_statistics(chi, chi_min=CHI0 - 0.1)
        assert result["n_voids"] >= 1
