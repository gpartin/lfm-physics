"""Tests for lfm.constants — verify all derived values."""

import math

from lfm import constants as C


class TestFundamental:
    def test_chi0_is_19(self):
        assert C.CHI0 == 19.0

    def test_chi0_from_geometry(self):
        assert C.CHI0 == 3**C.D - 2**C.D

    def test_kappa(self):
        assert C.KAPPA == 1 / 63
        assert C.KAPPA == 1 / (4**C.D - 1)

    def test_lambda_h(self):
        assert math.isclose(C.LAMBDA_H, 4 / 31, rel_tol=1e-12)
        assert math.isclose(C.LAMBDA_H, C.D_ST / (2 * C.D_ST**2 - 1), rel_tol=1e-12)

    def test_epsilon_w(self):
        assert math.isclose(C.EPSILON_W, 0.1, rel_tol=1e-12)
        assert math.isclose(C.EPSILON_W, 2 / (C.CHI0 + 1), rel_tol=1e-12)

    def test_alpha_s(self):
        assert math.isclose(C.ALPHA_S, 2 / 17, rel_tol=1e-12)


class TestModeStructure:
    def test_modes_sum_to_chi0(self):
        assert C.N_DC_MODE + C.N_FACE_MODES + C.N_EDGE_MODES == C.CHI0

    def test_face_modes(self):
        assert C.N_FACE_MODES == 6

    def test_edge_modes(self):
        assert C.N_EDGE_MODES == 12

    def test_corner_modes(self):
        assert C.N_CORNER_MODES == 8


class TestPredictions:
    def test_alpha_em(self):
        # 1/137.088, measured 1/137.036 => 0.04% error
        assert math.isclose(1 / C.ALPHA_EM, 137.088, rel_tol=0.001)

    def test_omega_lambda(self):
        assert math.isclose(C.OMEGA_LAMBDA, 13 / 19, rel_tol=1e-12)

    def test_omega_matter(self):
        assert math.isclose(C.OMEGA_MATTER, 6 / 19, rel_tol=1e-12)

    def test_omega_sum_to_one(self):
        assert math.isclose(C.OMEGA_LAMBDA + C.OMEGA_MATTER, 1.0, rel_tol=1e-12)

    def test_sin2_theta_w(self):
        assert C.SIN2_THETA_W == 3 / 8

    def test_n_generations(self):
        assert C.N_GENERATIONS == 3

    def test_n_gluons(self):
        assert C.N_GLUONS == 8

    def test_n_efoldings(self):
        assert C.N_EFOLDINGS == 60

    def test_cabibbo(self):
        assert math.isclose(C.CABIBBO_ANGLE_SIN, 1 / math.sqrt(20), rel_tol=1e-12)


class TestNumerical:
    def test_cfl_safe(self):
        """DT_DEFAULT must be below the CFL limit."""
        assert C.DT_DEFAULT < C.CFL_19PT_MASSIVE

    def test_stencil_weights_sum(self):
        """6 faces × 1/3 + 12 edges × 1/6 + center = 0 (conservation)."""
        total = (
            6 * C.STENCIL_FACE_WEIGHT
            + 12 * C.STENCIL_EDGE_WEIGHT
            + C.STENCIL_CENTER_WEIGHT
        )
        assert math.isclose(total, 0.0, abs_tol=1e-12)

    def test_mexican_hat_period(self):
        """dt=0.02 gives ~16 samples per Mexican hat period."""
        samples_per_period = C.MEXICAN_HAT_PERIOD / C.DT_DEFAULT
        assert samples_per_period > 10, "Need at least 10 samples per MH period"
