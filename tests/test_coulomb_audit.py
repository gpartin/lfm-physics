"""Reject proxy, wrong-power, nonreciprocal and unresolved Coulomb claims."""

import numpy as np
import pytest

from experiments.quantitative.exp_03_coulomb_law import run
from experiments.quantitative.exp_14_coulomb_dynamics import (
    adjudicate,
    fit_acceleration,
    scaling_gate,
)


def synthetic(power=-2):
    return [
        dict(
            q_left=q1,
            q_right=q2,
            separation=d,
            a_left=-q1 * q2 * d**power,
            a_right=q1 * q2 * d**power,
            uncertainty=1e-8,
        )
        for q1, q2 in ((1, 1), (-1, -1), (1, -1), (-1, 1))
        for d in (10, 14, 18, 22)
    ]


def test_historical_control_is_not_force_validation(capsys):
    result = run()
    assert result["passed"]
    assert result["evolution_steps"] == 0
    assert result["coulomb_status"] == result["maxwell_status"] == "NOT_ESTABLISHED"
    assert "initial interference control only" in capsys.readouterr().out


def test_quadratic_acceleration_with_nonzero_initial_velocity():
    t = np.linspace(0, 2, 20)
    assert fit_acceleration(t, 3 + 4 * t - 0.7 * t * t)["acceleration"] == pytest.approx(-1.4)


def test_invalid_positions_rejected():
    with pytest.raises(ValueError):
        fit_acceleration(np.arange(6), [0, 1, 2, np.nan, 4, 5])


def test_phase_labels_cannot_certify_charge():
    assert adjudicate(synthetic())["coulomb_status"] == "NOT_ESTABLISHED"
    result = adjudicate(synthetic(), signed_sources=True, converged=True, stable_objects=True)
    assert result["screen_status"] == "PASS"
    assert result["coulomb_status"] == "NOT_ESTABLISHED"
    assert result["maxwell_status"] == "NOT_TESTED"


@pytest.mark.parametrize("power", [-1, -3, 0])
def test_wrong_force_power_rejected(power):
    assert not scaling_gate([10, 14, 18, 22], [d**power for d in (10, 14, 18, 22)])["passed"]


def test_broken_reciprocity_rejected():
    rows = synthetic()
    rows[0]["a_right"] *= 2
    assert not adjudicate(rows)["gates"]["equal_mass_reciprocity"]


def test_missing_sign_sector_rejected():
    assert not adjudicate(synthetic()[:12])["gates"]["inverse_square"]


def test_unresolved_signal_rejected():
    rows = synthetic()
    for row in rows:
        row["uncertainty"] = 1
    assert not adjudicate(rows)["gates"]["resolved_signal"]


def test_nonfinite_rejected():
    rows = synthetic()
    rows[0]["a_left"] = np.nan
    assert adjudicate(rows)["coulomb_status"] == "NOT_ESTABLISHED"


def test_sector_dependent_coupling_rejected():
    rows = synthetic()
    for row in rows:
        if row["q_left"] != row["q_right"]:
            row["a_left"] *= 2
            row["a_right"] *= 2
    assert not adjudicate(rows)["gates"]["common_unit_charge_coupling"]


def test_overlap_matches_gaussian_control():
    from experiments.quantitative.exp_03_coulomb_law import AMP, SIGMA

    result = run()
    for row in result["measurements"]:
        analytic = 4 * AMP**2 * np.pi**1.5 * SIGMA**3 * np.exp(-(row["sep"] ** 2) / (4 * SIGMA**2))
        assert row["delta"] == pytest.approx(analytic, rel=1e-5)
