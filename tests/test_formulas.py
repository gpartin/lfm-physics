"""Tests for lfm.formulas subpackage."""

import math

import numpy as np

from lfm.formulas import (
    alpha_em,
    alpha_strong,
    angular_momentum_mass,
    down_quark_mass_ratio,
    higgs_w_mass_ratio,
    lepton_mass_ratio,
    mass_table,
    muon_g2_anomaly,
    n_efoldings,
    n_generations,
    n_gluons,
    omega_lambda,
    omega_matter,
    predict_all,
    proton_mass_ratio,
    proton_mass_ratio_3gen,
    sin2_theta_w,
    up_quark_mass_ratio,
    w_boson_mass_ratio,
    z_recombination,
    z_w_mass_ratio,
)

CHI0 = 19.0


# ──── predictions.py ────


class TestCouplings:
    def test_alpha_em(self):
        a = alpha_em()
        assert np.isclose(a, 11 / (480 * math.pi))
        assert np.isclose(1 / a, 137.088, rtol=0.001)

    def test_alpha_strong(self):
        assert np.isclose(alpha_strong(), 2 / 17)

    def test_sin2_theta_w(self):
        assert np.isclose(sin2_theta_w(), 3 / 8)


class TestCosmology:
    def test_omega_lambda(self):
        assert np.isclose(omega_lambda(), 13 / 19)

    def test_omega_matter(self):
        assert np.isclose(omega_matter(), 6 / 19)

    def test_omega_sum_to_one(self):
        assert np.isclose(omega_lambda() + omega_matter(), 1.0)

    def test_n_efoldings(self):
        assert n_efoldings() == 60.0

    def test_z_recombination(self):
        assert z_recombination() == 1089.0

    def test_n_generations(self):
        assert n_generations() == 3

    def test_n_gluons(self):
        assert n_gluons() == 8


class TestMuonG2:
    def test_order_of_magnitude(self):
        g2 = muon_g2_anomaly()
        assert 1e-10 < g2 < 1e-8


class TestPredictAll:
    def test_returns_dict(self):
        catalog = predict_all()
        assert isinstance(catalog, dict)
        assert len(catalog) >= 29

    def test_each_entry_has_fields(self):
        catalog = predict_all()
        for name, entry in catalog.items():
            assert "predicted" in entry, f"{name} missing 'predicted'"
            assert "measured" in entry, f"{name} missing 'measured'"
            assert "error_pct" in entry, f"{name} missing 'error_pct'"
            assert "formula" in entry, f"{name} missing 'formula'"

    def test_most_predictions_within_5_percent(self):
        catalog = predict_all()
        within_5 = sum(1 for e in catalog.values() if e["error_pct"] < 5.0)
        total = len(catalog)
        ratio = within_5 / total
        assert ratio > 0.8, f"Only {within_5}/{total} predictions within 5%"

    def test_exact_predictions(self):
        catalog = predict_all()
        exact = ["sin2_theta_w", "N_efoldings", "N_generations", "N_gluons"]
        for name in exact:
            assert np.isclose(
                catalog[name]["error_pct"], 0.0, atol=0.5
            ), f"{name} should be exact, got {catalog[name]['error_pct']:.2f}%"


# ──── masses.py ────


class TestAngularMomentumMass:
    def test_l_zero(self):
        assert angular_momentum_mass(0) == 0.0

    def test_l_14(self):
        assert angular_momentum_mass(14) == 14 * 15  # 210

    def test_l_42(self):
        assert angular_momentum_mass(42) == 42 * 43  # 1806


class TestLeptonMasses:
    def test_electron(self):
        assert lepton_mass_ratio(1) == 1.0

    def test_muon(self):
        ratio = lepton_mass_ratio(2)
        assert ratio == 210.0
        # Compare to measured: 206.768 → ~1.6% error
        assert abs(ratio - 206.768) / 206.768 * 100 < 2.0

    def test_tau(self):
        ratio = lepton_mass_ratio(3)
        assert ratio == 3540.0  # 59 × 60
        assert abs(ratio - 3477.0) / 3477.0 * 100 < 2.0


class TestQuarkMasses:
    def test_up_gen1(self):
        assert up_quark_mass_ratio(1) == 11 * 12  # 132

    def test_charm_gen2(self):
        assert up_quark_mass_ratio(2) == 30 * 31  # 930

    def test_top_gen3(self):
        assert up_quark_mass_ratio(3) == 49 * 50  # 2450

    def test_down_gen1(self):
        assert down_quark_mass_ratio(1) == 12 * 13  # 156

    def test_strange_gen2(self):
        assert down_quark_mass_ratio(2) == 32 * 33  # 1056

    def test_bottom_gen3(self):
        assert down_quark_mass_ratio(3) == 52 * 53  # 2756


class TestProtonMass:
    def test_angular_momentum(self):
        assert proton_mass_ratio() == 42 * 43  # 1806

    def test_three_gen_decomposition(self):
        ratio = proton_mass_ratio_3gen()
        # 11³ + 19² + 12² = 1331 + 361 + 144 = 1836
        assert ratio == 1836.0


class TestBosonMasses:
    def test_w_boson(self):
        assert w_boson_mass_ratio() == 19**2 * (24 * 19 - 20)  # 157396

    def test_z_w_ratio(self):
        assert z_w_mass_ratio() == 9 / 8

    def test_higgs_w_ratio(self):
        assert np.isclose(higgs_w_mass_ratio(), 14 / 9)


class TestMassTable:
    def test_returns_list(self):
        table = mass_table()
        assert isinstance(table, list)
        assert len(table) >= 13

    def test_electron_first(self):
        table = mass_table()
        assert table[0]["particle"] == "electron"
        assert table[0]["predicted"] == 1.0

    def test_all_have_fields(self):
        table = mass_table()
        for row in table:
            assert "particle" in row
            assert "predicted" in row
            assert "measured" in row
            assert "error_pct" in row


# ──── Top-level import check ────


def test_top_level_imports():
    import lfm

    assert hasattr(lfm, "predict_all")
    assert hasattr(lfm, "mass_table")
