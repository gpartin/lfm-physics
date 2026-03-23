"""Tests for lfm.core.integrator — leapfrog time stepping."""

import numpy as np

from lfm.config import FieldLevel, SimulationConfig
from lfm.core.integrator import create_initial_state, step_leapfrog


class TestInitialState:
    def test_chi_is_chi0(self, small_config):
        state = create_initial_state(small_config)
        np.testing.assert_allclose(state.chi, small_config.chi0)

    def test_psi_is_zero(self, small_config):
        state = create_initial_state(small_config)
        np.testing.assert_allclose(state.psi, 0.0)

    def test_real_shape(self, small_config):
        state = create_initial_state(small_config)
        N = small_config.grid_size
        assert state.psi.shape == (N, N, N)
        assert state.chi.shape == (N, N, N)

    def test_complex_shape(self, small_complex_config):
        state = create_initial_state(small_complex_config)
        N = small_complex_config.grid_size
        assert state.psi.shape == (2, N, N, N)

    def test_color_shape(self):
        cfg = SimulationConfig(
            grid_size=16,
            field_level=FieldLevel.COLOR,
            e_amplitude=12.0,
        )
        state = create_initial_state(cfg)
        assert state.psi.shape == (3, 2, 16, 16, 16)

    def test_boundary_mask_exists_for_frozen(self, small_config):
        state = create_initial_state(small_config)
        assert state.boundary_mask is not None
        assert state.boundary_mask.shape == (16, 16, 16)

    def test_boundary_mask_freezes_corners(self, small_config):
        state = create_initial_state(small_config)
        # Corner (0,0,0) should be frozen
        assert state.boundary_mask[0, 0, 0]
        # Center should NOT be frozen
        c = small_config.grid_size // 2
        assert not state.boundary_mask[c, c, c]


class TestLeapfrog:
    def test_empty_universe_stable(self, small_config):
        """No energy → χ stays at χ₀, Ψ stays at 0."""
        state = create_initial_state(small_config)
        for _ in range(10):
            step_leapfrog(state, small_config)
        np.testing.assert_allclose(state.psi, 0.0, atol=1e-15)
        np.testing.assert_allclose(state.chi, small_config.chi0, atol=1e-10)

    def test_step_increments(self, small_config):
        state = create_initial_state(small_config)
        assert state.step == 0
        step_leapfrog(state, small_config)
        assert state.step == 1
        step_leapfrog(state, small_config)
        assert state.step == 2

    def test_energy_creates_chi_well(self, small_config):
        """A blob of energy should reduce χ below χ₀."""
        state = create_initial_state(small_config)
        N = small_config.grid_size
        c = N // 2

        # Place energy at center
        state.psi[c, c, c] = 5.0
        state.psi_prev[c, c, c] = 5.0

        for _ in range(50):
            step_leapfrog(state, small_config)

        # χ at center should have decreased from χ₀ = 19
        assert state.chi[c, c, c] < small_config.chi0

    def test_boundary_frozen(self, small_config):
        """Boundary stays at χ₀ even with energy present."""
        state = create_initial_state(small_config)
        N = small_config.grid_size
        c = N // 2
        state.psi[c, c, c] = 5.0
        state.psi_prev[c, c, c] = 5.0

        for _ in range(20):
            step_leapfrog(state, small_config)

        # Corner should still be χ₀
        assert state.chi[0, 0, 0] == small_config.chi0

    def test_complex_field_steps(self, small_complex_config):
        """Complex field integration doesn't crash."""
        state = create_initial_state(small_complex_config)
        N = small_complex_config.grid_size
        c = N // 2
        state.psi[0, c, c, c] = 3.0  # Real part
        state.psi_prev[0, c, c, c] = 3.0

        for _ in range(10):
            step_leapfrog(state, small_complex_config)

        assert state.step == 10
        assert np.isfinite(state.chi).all()
