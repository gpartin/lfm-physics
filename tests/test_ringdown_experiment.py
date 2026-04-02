"""Smoke tests for ringdown experiment APIs."""

from pathlib import Path

from lfm.experiment import next5_falsification_projection_v2, qnm_mode_projection_check


def test_qnm_mode_projection_check_smoke(tmp_path: Path):
    result = qnm_mode_projection_check(
        N=24,
        amplitude=6.0,
        sigma=4.0,
        ring_steps=160,
        record_every=4,
        perturb_profile="uniform",
        perturb_frac=0.05,
        probe_offsets=(3, 4, 5),
        capture_movie=False,
    )
    written = result.save("ringdown_smoke", directory=tmp_path, save_movie=False)
    assert "summary_json" in written
    assert written["summary_json"].exists()
    assert "comparison" in result.summary
    assert len(result.summary["probe_fits"]) == 3


def test_next5_falsification_projection_v2_smoke(tmp_path: Path):
    result = next5_falsification_projection_v2(
        f1_grid=20,
        f2_grids=(16, 20, 24),
        f3_grid=20,
        f4_grid=20,
        base_amp=6.0,
        base_sigma=4.0,
        f1_ring_steps=120,
        f2_ring_steps=100,
        f3_ring_steps=120,
        f4_ring_steps=140,
        merger_pre_steps=60,
        record_every=4,
    )
    written = result.save("next5_smoke", directory=tmp_path)
    assert written["summary_json"].exists()
    assert set(result.summary["verdicts"]) == {
        "F1_mode_ratio_projection",
        "F2_resolution_convergence_projection",
        "F3_basis_invariance_projection",
        "F4_merger_comoving_projection",
        "F5_external_consistency_projection",
    }
