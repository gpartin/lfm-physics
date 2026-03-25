"""Tests for v1.2.0 features.

Covers:
* lfm.experiment.Slit       — dataclass defaults and fields
* lfm.experiment.Barrier    — mask construction, apply, attenuate, measure, callback
* lfm.experiment.DetectorScreen — record, pattern, click_pattern, line_profile, reset
* Simulation.place_barrier  — convenience factory
* Simulation.add_detector   — convenience factory
* lfm.viz.volume_render_available  — probe function
* lfm.viz.plot_interference_pattern — 2-D heatmap + profile
* lfm.viz.render_3d_volume(backend="matplotlib")
* lfm.viz.animate_double_slit  — FuncAnimation
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import CHI0
from lfm.experiment.barrier import Barrier, Slit
from lfm.experiment.detector import DetectorScreen
from lfm.simulation import Simulation

N = 24  # small grid for fast tests
DT = 0.02
BARRIER_POS = N // 2        # z = 12
SLIT_CENTER_A = N // 2 - 4  # y =  8
SLIT_CENTER_B = N // 2 + 4  # y = 16


def _cfg(**kwargs) -> SimulationConfig:
    defaults = dict(
        grid_size=N,
        field_level=FieldLevel.REAL,
        boundary_type=BoundaryType.FROZEN,
        report_interval=50,
        dt=DT,
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def _sim(**kwargs) -> Simulation:
    s = Simulation(_cfg(**kwargs))
    s.place_soliton((N // 2, N // 2, N // 8), amplitude=3.0)
    return s


def _sim_complex(**kwargs) -> Simulation:
    s = Simulation(_cfg(field_level=FieldLevel.COMPLEX, **kwargs))
    s.place_soliton((N // 2, N // 2, N // 8), amplitude=3.0)
    return s


def _two_slits() -> list[Slit]:
    return [
        Slit(center=SLIT_CENTER_A, width=3),
        Slit(center=SLIT_CENTER_B, width=3),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Slit dataclass
# ═══════════════════════════════════════════════════════════════════════


class TestSlitDataclass:
    def test_center_is_required(self):
        with pytest.raises(TypeError):
            Slit()  # missing required 'center'

    def test_defaults(self):
        s = Slit(center=10)
        assert s.width == 4
        assert s.detector is False
        assert s.detector_strength == 1.0

    def test_custom_values(self):
        s = Slit(center=5, width=6, detector=True, detector_strength=0.7)
        assert s.center == 5
        assert s.width == 6
        assert s.detector is True
        assert s.detector_strength == pytest.approx(0.7)


# ═══════════════════════════════════════════════════════════════════════
# Barrier — mask construction
# ═══════════════════════════════════════════════════════════════════════


class TestBarrierMasks:
    def _barrier(self, sim=None, **kwargs) -> Barrier:
        sim = sim or _sim()
        defaults = dict(axis=2, position=BARRIER_POS, slits=_two_slits())
        defaults.update(kwargs)
        return Barrier(sim, **defaults)

    def test_barrier_mask_is_bool_3d(self):
        b = self._barrier()
        assert b.mask.dtype == bool
        assert b.mask.shape == (N, N, N)

    def test_barrier_mask_sum_nonzero(self):
        b = self._barrier()
        assert b.mask.sum() > 0

    def test_slit_masks_count(self):
        b = self._barrier()
        assert len(b.slit_masks) == 2

    def test_slit_mask_is_bool_3d(self):
        b = self._barrier()
        for sm in b.slit_masks:
            assert sm.dtype == bool
            assert sm.shape == (N, N, N)

    def test_barrier_and_slits_disjoint(self):
        """Slit openings must not overlap with solid barrier cells."""
        b = self._barrier()
        for sm in b.slit_masks:
            assert not (b.mask & sm).any()

    def test_single_slit_produces_one_mask(self):
        b = self._barrier(slits=[Slit(center=N // 2, width=3)])
        assert len(b.slit_masks) == 1

    def test_slits_property_matches_input(self):
        inp = _two_slits()
        b = self._barrier(slits=inp)
        assert b.slits == inp


# ═══════════════════════════════════════════════════════════════════════
# Barrier — apply
# ═══════════════════════════════════════════════════════════════════════


class TestBarrierApply:
    def test_barrier_chi_is_high(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
        chi = sim.chi
        # All solid barrier cells should have chi > CHI0
        assert (chi[b.mask] > CHI0).all()

    def test_slit_chi_is_chi0(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
        chi = sim.chi
        for sm in b.slit_masks:
            assert np.allclose(chi[sm], CHI0, atol=1.0)

    def test_absorb_zeros_psi_in_barrier(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits(), absorb=True)
        psi = sim.psi_real
        # Barrier cells should have zero psi
        assert np.allclose(psi[b.mask], 0.0)

    def test_no_absorb_leaves_psi(self):
        sim = _sim()
        # Put some energy in the barrier zone first
        pr = sim.psi_real.copy()
        pr[BARRIER_POS, :, :] += 1.0
        sim.set_psi_real(pr)
        _before_sum = sim.psi_real[BARRIER_POS, :, :].sum()

        _b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits(), absorb=False)
        # With absorb=False, psi should NOT have been zeroed (may differ
        # only due to slit relaxation, but we check solid barrier cells)
        assert sim.psi_real[BARRIER_POS, :, :].sum() != pytest.approx(0.0, abs=1e-6)

    def test_apply_restores_high_chi_after_manual_reset(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
        # Manually destroy the barrier
        chi = sim.chi.copy()
        chi[b.mask] = CHI0
        sim.set_chi(chi)
        assert np.allclose(sim.chi[b.mask], CHI0)  # barrier gone
        # apply() should restore it
        b.apply()
        assert (sim.chi[b.mask] > CHI0).all()


# ═══════════════════════════════════════════════════════════════════════
# Barrier — which-path detection
# ═══════════════════════════════════════════════════════════════════════


class TestBarrierWhichPath:
    def test_attenuate_is_noop(self):
        """attenuate_slits is a no-op; which-path detection uses χ in apply()."""
        sim = _sim()
        slits = [
            Slit(center=SLIT_CENTER_A, width=4, detector=True, detector_strength=0.5),
            Slit(center=SLIT_CENTER_B, width=4),
        ]
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=slits, absorb=False)
        pr = sim.psi_real.copy()
        pr[b.slit_masks[0]] = 5.0
        sim.set_psi_real(pr)
        before = sim.psi_real[b.slit_masks[0]].mean()
        b.attenuate_slits()
        after = sim.psi_real[b.slit_masks[0]].mean()
        assert after == pytest.approx(before, rel=1e-5)

    def test_non_detector_slit_unaffected(self):
        sim = _sim()
        slits = [
            Slit(center=SLIT_CENTER_A, width=4, detector=True, detector_strength=1.0),
            Slit(center=SLIT_CENTER_B, width=4, detector=False),
        ]
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=slits, absorb=False)
        pr = sim.psi_real.copy()
        pr[b.slit_masks[1]] = 3.0
        sim.set_psi_real(pr)
        before = sim.psi_real[b.slit_masks[1]].mean()
        b.attenuate_slits()
        after = sim.psi_real[b.slit_masks[1]].mean()
        assert after == pytest.approx(before, rel=1e-5)

    def test_full_strength_leaves_psi_unchanged(self):
        """Even at full strength, attenuate_slits is a no-op (χ-based)."""
        sim = _sim()
        slits = [Slit(center=SLIT_CENTER_A, width=4, detector=True, detector_strength=1.0)]
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=slits, absorb=False)
        pr = sim.psi_real.copy()
        pr[b.slit_masks[0]] = 4.0
        sim.set_psi_real(pr)
        b.attenuate_slits()
        assert np.allclose(sim.psi_real[b.slit_masks[0]], 4.0, atol=1e-6)

    def test_attenuate_complex_field_unchanged(self):
        """Complex psi_imag also unchanged by no-op attenuate_slits."""
        sim = _sim_complex()
        slits = [Slit(center=SLIT_CENTER_A, width=4, detector=True, detector_strength=0.5)]
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=slits, absorb=False)
        pi = sim.psi_imag.copy()
        pi[b.slit_masks[0]] = 6.0
        sim.set_psi_imag(pi)
        before = sim.psi_imag[b.slit_masks[0]].mean()
        b.attenuate_slits()
        after = sim.psi_imag[b.slit_masks[0]].mean()
        assert after == pytest.approx(before, rel=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Barrier — measure_slits
# ═══════════════════════════════════════════════════════════════════════


class TestBarrierMeasure:
    def test_keys_match_slit_count(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
        result = b.measure_slits()
        assert set(result.keys()) == {"slit_0", "slit_1"}

    def test_values_are_floats(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
        for v in b.measure_slits().values():
            assert isinstance(v, float)

    def test_single_slit_single_key(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=[Slit(center=N // 2, width=3)])
        assert list(b.measure_slits().keys()) == ["slit_0"]


# ═══════════════════════════════════════════════════════════════════════
# Barrier — step_callback
# ═══════════════════════════════════════════════════════════════════════


class TestBarrierCallback:
    def test_callback_signature(self):
        """step_callback accepts (sim, step) — no exception."""
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
        b.step_callback(sim, 0)  # should not raise

    def test_callback_preserves_high_chi(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
        # Destroy barrier
        chi = sim.chi.copy()
        chi[b.mask] = CHI0
        sim.set_chi(chi)
        # Callback should restore it
        b.step_callback(sim, 1)
        assert (sim.chi[b.mask] > CHI0).all()

    def test_callback_calls_attenuate_if_detector(self):
        """attenuate_slits is a no-op; psi unchanged by callback."""
        sim = _sim()
        slits = [Slit(center=SLIT_CENTER_A, width=4, detector=True, detector_strength=0.5)]
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=slits, absorb=False)
        pr = sim.psi_real.copy()
        pr[b.slit_masks[0]] = 5.0
        sim.set_psi_real(pr)
        b.step_callback(sim, 0)
        assert sim.psi_real[b.slit_masks[0]].mean() == pytest.approx(5.0, rel=1e-5)

    def test_callback_no_attenuate_without_detector(self):
        sim = _sim()
        b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits(), absorb=False)
        pr = sim.psi_real.copy()
        pr[b.slit_masks[0]] = 5.0
        sim.set_psi_real(pr)
        b.step_callback(sim, 0)
        # Slit cells (not barrier) — attenuate not called so value unchanged
        assert sim.psi_real[b.slit_masks[0]].mean() == pytest.approx(5.0, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# Barrier — repr
# ═══════════════════════════════════════════════════════════════════════


def test_barrier_repr():
    sim = _sim()
    b = Barrier(sim, axis=2, position=BARRIER_POS, slits=_two_slits())
    r = repr(b)
    assert "Barrier" in r


# ═══════════════════════════════════════════════════════════════════════
# DetectorScreen — recording
# ═══════════════════════════════════════════════════════════════════════


class TestDetectorRecord:
    def test_default_position(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        assert s.position == int(N * 0.80)

    def test_custom_position(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2, position=18)
        assert s.position == 18

    def test_n_frames_starts_at_zero(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        assert s.n_frames == 0

    def test_record_increments_n_frames(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        s.record()
        assert s.n_frames == 1
        s.record()
        assert s.n_frames == 2

    def test_step_callback_equivalent_to_record(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        s.step_callback(sim, 0)
        assert s.n_frames == 1


# ═══════════════════════════════════════════════════════════════════════
# DetectorScreen — pattern
# ═══════════════════════════════════════════════════════════════════════


class TestDetectorPattern:
    def test_pattern_shape_before_recording(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        assert s.pattern.shape == (N, N)
        assert np.allclose(s.pattern, 0.0)

    def test_pattern_shape_after_recording(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        for _ in range(3):
            s.record()
        assert s.pattern.shape == (N, N)

    def test_pattern_is_sum_of_frames(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        for _ in range(4):
            s.record()
        # Must equal sum of all individual snapshots
        assert np.allclose(s.pattern, s.snapshots.sum(axis=0))

    def test_snapshots_shape(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        for _ in range(5):
            s.record()
        assert s.snapshots.shape == (5, N, N)

    def test_snapshots_empty_before_recording(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        assert s.snapshots.shape == (0, N, N)

    def test_reset_clears_frames(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        for _ in range(3):
            s.record()
        assert s.n_frames == 3
        s.reset()
        assert s.n_frames == 0
        assert s.pattern.shape == (N, N)
        assert np.allclose(s.pattern, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# DetectorScreen — click_pattern & line_profile
# ═══════════════════════════════════════════════════════════════════════


class TestDetectorDerived:
    def test_click_pattern_shape(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        s.record()
        cp = s.click_pattern(n_particles=100, seed=42)
        assert cp.shape == (N, N)

    def test_click_pattern_sum(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        s.record()
        n = 200
        cp = s.click_pattern(n_particles=n, seed=0)
        assert cp.sum() == n

    def test_click_pattern_zero_before_recording(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        cp = s.click_pattern(n_particles=50)
        assert cp.sum() == 0

    def test_line_profile_shape(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        s.record()
        lp = s.line_profile()
        assert lp.shape == (N,)

    def test_line_profile_axis1(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        s.record()
        lp = s.line_profile(perp_axis=1)
        assert lp.shape == (N,)

    def test_mean_pattern_shape(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2)
        for _ in range(3):
            s.record()
        mp = s.mean_pattern()
        assert mp.shape == (N, N)


# ═══════════════════════════════════════════════════════════════════════
# DetectorScreen — field selection
# ═══════════════════════════════════════════════════════════════════════


class TestDetectorFields:
    def test_records_psi_real_field(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2, field="psi_real")
        s.record()
        assert s.n_frames == 1

    def test_records_chi_field(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2, field="chi")
        s.record()
        assert s.n_frames == 1

    def test_unknown_field_raises(self):
        sim = _sim()
        s = DetectorScreen(sim, axis=2, field="bad_field")
        with pytest.raises(ValueError, match="Unknown field"):
            s.record()

    def test_psi_imag_returns_zeros_for_real_sim(self):
        """psi_imag field on a REAL Simulation returns zeros (not an error)."""
        sim = _sim()
        s = DetectorScreen(sim, axis=2, field="psi_imag")
        s.record()
        # Returns zeros since psi_imag is None for REAL simulations
        assert np.allclose(s.pattern, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# DetectorScreen — repr
# ═══════════════════════════════════════════════════════════════════════


def test_detector_repr():
    sim = _sim()
    s = DetectorScreen(sim, axis=2, position=20)
    r = repr(s)
    assert "DetectorScreen" in r
    assert "20" in r


# ═══════════════════════════════════════════════════════════════════════
# Simulation convenience methods
# ═══════════════════════════════════════════════════════════════════════


class TestSimulationConvenienceMethods:
    def test_place_barrier_returns_barrier(self):
        sim = _sim()
        b = sim.place_barrier(axis=2, position=BARRIER_POS, slits=_two_slits())
        assert isinstance(b, Barrier)

    def test_place_barrier_default_slits(self):
        sim = _sim()
        b = sim.place_barrier(axis=2, position=BARRIER_POS)
        assert len(b.slit_masks) == 2

    def test_add_detector_returns_detector(self):
        sim = _sim()
        s = sim.add_detector(axis=2)
        assert isinstance(s, DetectorScreen)

    def test_add_detector_custom_position(self):
        sim = _sim()
        s = sim.add_detector(axis=2, position=19)
        assert s.position == 19

    def test_place_barrier_absorb_false(self):
        sim = _sim()
        b = sim.place_barrier(axis=2, position=BARRIER_POS, absorb=False)
        assert isinstance(b, Barrier)

    def test_barrier_and_detector_together(self):
        """Both can be created on the same simulation."""
        sim = _sim()
        b = sim.place_barrier(axis=2, position=BARRIER_POS)
        s = sim.add_detector(axis=2, position=N - 4)
        assert isinstance(b, Barrier)
        assert isinstance(s, DetectorScreen)


# ═══════════════════════════════════════════════════════════════════════
# lfm.viz.quantum — volume_render_available
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
class TestVizQuantum:
    def test_volume_render_available_is_bool(self):
        from lfm.viz.quantum import volume_render_available

        result = volume_render_available()
        assert isinstance(result, bool)

    def test_plot_interference_pattern_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lfm.viz.quantum import plot_interference_pattern

        pattern = np.random.rand(N, N).astype(np.float32)
        fig = plot_interference_pattern(pattern)
        assert fig is not None
        plt.close("all")

    def test_plot_interference_pattern_no_profile(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lfm.viz.quantum import plot_interference_pattern

        pattern = np.random.rand(N, N).astype(np.float32)
        fig_no = plot_interference_pattern(pattern, show_profile=False)
        fig_yes = plot_interference_pattern(pattern, show_profile=True)
        assert fig_no is not None
        # With profile we always have more axes than without
        assert len(fig_yes.axes) > len(fig_no.axes)
        plt.close("all")

    def test_plot_interference_pattern_bad_shape_raises(self):
        from lfm.viz.quantum import plot_interference_pattern

        with pytest.raises(ValueError):
            plot_interference_pattern(np.ones((N, N, N)))

    def test_render_3d_volume_matplotlib_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lfm.viz.quantum import render_3d_volume

        field = np.random.rand(8, 8, 8).astype(np.float32)
        fig = render_3d_volume(field, backend="matplotlib")
        assert fig is not None
        plt.close("all")

    def test_render_3d_volume_bad_backend_raises(self):
        from lfm.viz.quantum import render_3d_volume

        with pytest.raises(ValueError, match="Unknown backend"):
            render_3d_volume(np.ones((8, 8, 8)), backend="bad_backend")

    def test_render_3d_volume_bad_shape_raises(self):
        from lfm.viz.quantum import render_3d_volume

        with pytest.raises(ValueError, match="3-D"):
            render_3d_volume(np.ones((8, 8)), backend="matplotlib")

    def test_animate_double_slit_returns_func_animation(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from lfm.viz.quantum import animate_double_slit

        n_snaps = 5
        snapshots = [
            {"step": i * 10, "energy_density": np.random.rand(N, N, N).astype(np.float32)}
            for i in range(n_snaps)
        ]
        anim = animate_double_slit(
            snapshots,
            barrier_axis=2,
            barrier_position=N // 2,
            detector_position=N - 4,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_animate_double_slit_empty_raises(self):
        from lfm.viz.quantum import animate_double_slit

        with pytest.raises(ValueError, match="empty"):
            animate_double_slit([])

    def test_animate_double_slit_missing_field_raises(self):
        from lfm.viz.quantum import animate_double_slit

        snapshots = [{"step": 0, "chi": np.ones((N, N, N), dtype=np.float32)}]
        with pytest.raises(ValueError, match="energy_density"):
            animate_double_slit(snapshots, field="energy_density")


# ═══════════════════════════════════════════════════════════════════════
# Public-API exports
# ═══════════════════════════════════════════════════════════════════════


def test_top_level_imports():
    """Barrier, Slit, DetectorScreen are importable from the top-level lfm package."""
    import lfm

    assert hasattr(lfm, "Barrier")
    assert hasattr(lfm, "Slit")
    assert hasattr(lfm, "DetectorScreen")


def test_version_is_v12():
    import lfm

    parts = lfm.__version__.split(".")
    assert int(parts[0]) >= 1
    assert int(parts[1]) >= 2


# ═══════════════════════════════════════════════════════════════════════
# animate_double_slit_3d
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
class TestAnimateDoubleSlit3D:
    def _snapshots(self, n_snaps: int = 5) -> list[dict]:
        return [
            {
                "step": i * 10,
                "energy_density": np.random.rand(N, N, N).astype(np.float32),
            }
            for i in range(n_snaps)
        ]

    def test_returns_func_animation(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from lfm.viz.quantum import animate_double_slit_3d

        anim = animate_double_slit_3d(
            self._snapshots(),
            barrier_axis=2,
            barrier_position=N // 2,
            detector_position=N - 4,
            source_position=N // 4,
            slit_centers=[N // 2 - 3, N // 2 + 3],
            max_frames=5,
            max_points=50,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_empty_snapshots_raises(self):
        from lfm.viz.quantum import animate_double_slit_3d

        with pytest.raises(ValueError, match="empty"):
            animate_double_slit_3d([])

    def test_missing_field_raises(self):
        from lfm.viz.quantum import animate_double_slit_3d

        snapshots = [{"step": 0, "chi": np.ones((N, N, N), dtype=np.float32)}]
        with pytest.raises(ValueError, match="energy_density"):
            animate_double_slit_3d(snapshots, field="energy_density")

    def test_2d_field_raises(self):
        from lfm.viz.quantum import animate_double_slit_3d

        snapshots = [
            {"step": 0, "energy_density": np.ones((N, N), dtype=np.float32)}
        ]
        with pytest.raises(ValueError, match="3-D"):
            animate_double_slit_3d(snapshots)

    def test_subsamples_when_exceeding_max_frames(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from lfm.viz.quantum import animate_double_slit_3d

        anim = animate_double_slit_3d(
            self._snapshots(20),
            max_frames=5,
            max_points=20,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════
# animate_3d_slices
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
class TestAnimate3DSlices:
    def _snapshots(self, n_snaps: int = 5) -> list[dict]:
        return [
            {
                "step": i * 10,
                "energy_density": np.random.rand(N, N, N).astype(np.float32),
            }
            for i in range(n_snaps)
        ]

    def test_returns_func_animation(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from lfm.viz.quantum import animate_3d_slices

        anim = animate_3d_slices(self._snapshots())
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_empty_snapshots_raises(self):
        from lfm.viz.quantum import animate_3d_slices

        with pytest.raises(ValueError, match="empty"):
            animate_3d_slices([])

    def test_missing_field_raises(self):
        from lfm.viz.quantum import animate_3d_slices

        snapshots = [{"step": 0, "chi": np.ones((N, N, N), dtype=np.float32)}]
        with pytest.raises(ValueError, match="energy_density"):
            animate_3d_slices(snapshots, field="energy_density")

    def test_2d_field_raises(self):
        from lfm.viz.quantum import animate_3d_slices

        snapshots = [
            {"step": 0, "energy_density": np.ones((N, N), dtype=np.float32)}
        ]
        with pytest.raises(ValueError, match="3-D"):
            animate_3d_slices(snapshots)


# ═══════════════════════════════════════════════════════════════════════
# save_snapshots / load_snapshots roundtrip
# ═══════════════════════════════════════════════════════════════════════


class TestSnapshotIO:
    def _snapshots(self, n: int = 3, grid: int = 8) -> list[dict]:
        return [
            {
                "step": i * 100,
                "energy_density": np.random.rand(grid, grid, grid).astype(
                    np.float32
                ),
                "chi": np.random.rand(grid, grid, grid).astype(np.float32),
            }
            for i in range(n)
        ]

    def test_roundtrip_preserves_count(self, tmp_path):
        from lfm.io import save_snapshots, load_snapshots

        snaps = self._snapshots(5)
        path = save_snapshots(snaps, tmp_path / "test.npz")
        loaded = load_snapshots(path)
        assert len(loaded) == 5

    def test_roundtrip_preserves_steps(self, tmp_path):
        from lfm.io import save_snapshots, load_snapshots

        snaps = self._snapshots(3)
        path = save_snapshots(snaps, tmp_path / "test.npz")
        loaded = load_snapshots(path)
        for orig, loaded_s in zip(snaps, loaded):
            assert loaded_s["step"] == orig["step"]

    def test_roundtrip_preserves_fields(self, tmp_path):
        from lfm.io import save_snapshots, load_snapshots

        snaps = self._snapshots(2)
        path = save_snapshots(snaps, tmp_path / "test.npz")
        loaded = load_snapshots(path)
        for orig, loaded_s in zip(snaps, loaded):
            np.testing.assert_array_almost_equal(
                loaded_s["energy_density"], orig["energy_density"], decimal=5
            )
            np.testing.assert_array_almost_equal(
                loaded_s["chi"], orig["chi"], decimal=5
            )

    def test_roundtrip_preserves_field_keys(self, tmp_path):
        from lfm.io import save_snapshots, load_snapshots

        snaps = self._snapshots(2)
        path = save_snapshots(snaps, tmp_path / "test.npz")
        loaded = load_snapshots(path)
        assert "energy_density" in loaded[0]
        assert "chi" in loaded[0]
        assert "step" in loaded[0]

    def test_adds_npz_suffix(self, tmp_path):
        from lfm.io import save_snapshots

        snaps = self._snapshots(1)
        path = save_snapshots(snaps, tmp_path / "no_suffix")
        assert path.suffix == ".npz"

    def test_empty_raises(self):
        from lfm.io import save_snapshots

        with pytest.raises(ValueError, match="empty"):
            save_snapshots([], "dummy.npz")

    def test_compressed_smaller_than_raw(self, tmp_path):
        from lfm.io import save_snapshots

        snaps = self._snapshots(10, grid=16)
        p_comp = save_snapshots(snaps, tmp_path / "comp.npz", compress=True)
        p_raw = save_snapshots(snaps, tmp_path / "raw.npz", compress=False)
        assert p_comp.stat().st_size < p_raw.stat().st_size
