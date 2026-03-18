"""Tests for lfm.core.evolver — backend-powered evolution loop."""

import numpy as np
import pytest

from lfm.config import FieldLevel, SimulationConfig
from lfm.core.evolver import Evolver


class TestEvolverInit:
    def test_default_creates_evolver(self):
        cfg = SimulationConfig(grid_size=16, e_amplitude=12.0)
        ev = Evolver(cfg, backend="cpu")
        assert ev.step == 0
        assert ev.backend.name == "numpy"

    def test_chi_initialized_to_chi0(self):
        cfg = SimulationConfig(grid_size=16, e_amplitude=12.0)
        ev = Evolver(cfg, backend="cpu")
        chi = ev.get_chi()
        np.testing.assert_allclose(chi, cfg.chi0)

    def test_psi_initialized_to_zero(self):
        cfg = SimulationConfig(grid_size=16, e_amplitude=12.0)
        ev = Evolver(cfg, backend="cpu")
        pr = ev.get_psi_real()
        np.testing.assert_allclose(pr, 0.0)

    def test_real_psi_imag_is_none(self):
        cfg = SimulationConfig(grid_size=16, e_amplitude=12.0)
        ev = Evolver(cfg, backend="cpu")
        assert ev.get_psi_imag() is None

    def test_complex_has_imag(self):
        cfg = SimulationConfig(
            grid_size=16, e_amplitude=12.0, field_level=FieldLevel.COMPLEX,
        )
        ev = Evolver(cfg, backend="cpu")
        pi = ev.get_psi_imag()
        assert pi is not None
        np.testing.assert_allclose(pi, 0.0)

    def test_color_shapes(self):
        cfg = SimulationConfig(
            grid_size=16, e_amplitude=12.0, field_level=FieldLevel.COLOR,
        )
        ev = Evolver(cfg, backend="cpu")
        pr = ev.get_psi_real()
        assert pr.shape == (3, 16, 16, 16)
        pi = ev.get_psi_imag()
        assert pi.shape == (3, 16, 16, 16)


class TestEvolverReal:
    @pytest.fixture
    def evolver(self):
        cfg = SimulationConfig(grid_size=16, e_amplitude=12.0)
        return Evolver(cfg, backend="cpu")

    def test_empty_universe_stable(self, evolver):
        evolver.evolve(20)
        np.testing.assert_allclose(evolver.get_psi_real(), 0.0, atol=1e-10)
        np.testing.assert_allclose(evolver.get_chi(), evolver.config.chi0, atol=1e-5)

    def test_step_counter(self, evolver):
        evolver.evolve(10)
        assert evolver.step == 10
        evolver.evolve(5)
        assert evolver.step == 15

    def test_callback_called(self, evolver):
        calls = []
        evolver.config.report_interval = 5
        evolver.evolve(10, callback=lambda ev, s: calls.append(s))
        assert calls == [5, 10]

    def test_energy_creates_chi_well(self, evolver):
        N = evolver.N
        c = N // 2
        E = np.zeros((N, N, N), dtype=np.float32)
        E[c, c, c] = 5.0
        evolver.set_psi_real(E)

        evolver.evolve(50)

        chi = evolver.get_chi()
        assert chi[c, c, c] < evolver.config.chi0, "Energy should create χ well"

    def test_energy_density(self, evolver):
        N = evolver.N
        c = N // 2
        E = np.zeros((N, N, N), dtype=np.float32)
        E[c, c, c] = 3.0
        evolver.set_psi_real(E)

        ed = evolver.get_energy_density()
        assert ed.shape == (N, N, N)
        np.testing.assert_allclose(ed[c, c, c], 9.0)


class TestEvolverComplex:
    @pytest.fixture
    def evolver(self):
        cfg = SimulationConfig(
            grid_size=16, e_amplitude=12.0, field_level=FieldLevel.COMPLEX,
        )
        return Evolver(cfg, backend="cpu")

    def test_empty_stable(self, evolver):
        evolver.evolve(20)
        np.testing.assert_allclose(evolver.get_psi_real(), 0.0, atol=1e-10)
        np.testing.assert_allclose(evolver.get_psi_imag(), 0.0, atol=1e-10)
        np.testing.assert_allclose(evolver.get_chi(), evolver.config.chi0, atol=1e-5)

    def test_complex_energy_density(self, evolver):
        N = evolver.N
        c = N // 2
        Pr = np.zeros((N, N, N), dtype=np.float32)
        Pi = np.zeros((N, N, N), dtype=np.float32)
        Pr[c, c, c] = 3.0
        Pi[c, c, c] = 4.0
        evolver.set_psi_real(Pr)
        evolver.set_psi_imag(Pi)

        ed = evolver.get_energy_density()
        np.testing.assert_allclose(ed[c, c, c], 25.0)  # 3²+4²

    def test_set_psi_imag_raises_for_real(self):
        cfg = SimulationConfig(grid_size=16, e_amplitude=12.0)
        ev = Evolver(cfg, backend="cpu")
        with pytest.raises(ValueError, match="imaginary"):
            ev.set_psi_imag(np.zeros((16, 16, 16)))


class TestEvolverColor:
    def test_color_evolves(self):
        cfg = SimulationConfig(
            grid_size=16, e_amplitude=12.0, field_level=FieldLevel.COLOR,
        )
        ev = Evolver(cfg, backend="cpu")
        ev.evolve(5)
        assert ev.step == 5
        chi = ev.get_chi()
        np.testing.assert_allclose(chi, cfg.chi0, atol=1e-5)


class TestEvolverSetChi:
    def test_set_and_get_chi(self):
        cfg = SimulationConfig(grid_size=16, e_amplitude=12.0)
        ev = Evolver(cfg, backend="cpu")

        new_chi = np.full((16, 16, 16), 18.5, dtype=np.float32)
        ev.set_chi(new_chi)

        chi = ev.get_chi()
        np.testing.assert_allclose(chi, 18.5)
