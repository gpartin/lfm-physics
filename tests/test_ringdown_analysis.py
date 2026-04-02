"""Tests for ringdown analysis utilities."""

import numpy as np

from lfm.analysis import (
    fit_ringdown_series,
    project_field_onto_modes,
    relative_spread,
    split_frequency_bands,
    target_band_summary,
)


def test_relative_spread_basic():
    vals = [10.0, 12.0, 11.0]
    s = relative_spread(vals)
    assert 0.17 < s < 0.19


def test_split_frequency_bands():
    out = split_frequency_bands([0.2, 0.8, 1.2, 2.0], min_target=1.0)
    assert out["slow"] == [0.2, 0.8]
    assert out["target"] == [1.2, 2.0]


def test_fit_ringdown_series_valid():
    dt = 0.02
    t = np.arange(0.0, 20.0, dt)
    omega = 9.0
    gamma = 0.15
    y = np.exp(-gamma * t) * np.cos(omega * t)
    fit = fit_ringdown_series(t, y, start_frac=0.1, min_peaks=8)
    assert fit["valid"]
    assert abs(float(fit["omega"]) - omega) < 0.5
    assert abs(float(fit["gamma"]) - gamma) < 0.1


def test_fit_ringdown_series_invalid_for_short_signal():
    t = np.linspace(0.0, 1.0, 20)
    y = np.cos(2.0 * np.pi * t)
    fit = fit_ringdown_series(t, y)
    assert not fit["valid"]


def test_project_field_onto_modes_identifies_dominant_mode():
    n = 24
    ii, jj, kk = np.indices((n, n, n), dtype=np.float64)
    # Pure cosine on k=(1,0,0)
    field = np.cos(2.0 * np.pi * ii / n)
    coeffs = project_field_onto_modes(field, [(1, 0, 0), (0, 1, 0)], subtract_mean=True)
    a = abs(coeffs["(1,0,0)"])
    b = abs(coeffs["(0,1,0)"])
    assert a > 0.45  # cosine projection magnitude is ~0.5 on matching complex mode
    assert b < 1e-3


def test_project_field_onto_modes_center_shift_changes_phase_not_magnitude():
    n = 24
    ii, jj, kk = np.indices((n, n, n), dtype=np.float64)
    field = np.cos(2.0 * np.pi * ii / n)
    c1 = project_field_onto_modes(field, [(1, 0, 0)], subtract_mean=True)["(1,0,0)"]
    c2 = project_field_onto_modes(
        field,
        [(1, 0, 0)],
        subtract_mean=True,
        center_shift=(3, 0, 0),
    )["(1,0,0)"]
    assert abs(abs(c1) - abs(c2)) < 1e-6


def test_target_band_summary_uses_only_valid_rows():
    rows = [
        {"omega": 19.3, "valid": True},
        {"omega": 19.4, "valid": True},
        {"omega": 0.2, "valid": True},
        {"omega": 99.0, "valid": False},
    ]
    s = target_band_summary(rows, min_target=1.0)
    assert s["valid_mode_count"] == 3
    assert len(s["target_band"]) == 2
    assert len(s["slow_band"]) == 1
    assert 19.3 < float(s["target_center"]) < 19.4
