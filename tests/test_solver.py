"""
Tests for the SCF eigenmode solver (Phase 2).

All tests use N=32 to keep run-time short.  The solver is validated
for the electron (the lightest, fastest-converging particle).

Physical invariants we check:
  - chi_min > 0 throughout  (Z2 vacuum flip = anti-confining = failure)
  - solver returns a SolitonSolution with correct metadata
  - energy_history is populated
  - N attribute matches the requested grid size
"""

import pytest

import lfm
from lfm.particles.solver import SolitonSolution, solve_eigenmode

# ── SolitonSolution dataclass ────────────────────────────────────────────────


class TestSolitonSolutionStructure:
    def test_fields_exist(self):
        import numpy as np

        N = 4
        sol = SolitonSolution(
            psi_r=np.zeros((N, N, N), dtype=np.float32),
            psi_i=None,
            chi=np.full((N, N, N), 19.0, dtype=np.float32),
            chi_min=19.0,
            energy=0.0,
            eigenvalue=19.0,
            converged=False,
            cycles=0,
            energy_history=[],
            particle=lfm.ELECTRON,
            N=N,
        )
        assert sol.N == N
        assert sol.particle is lfm.ELECTRON
        assert sol.chi_min == pytest.approx(19.0)


# ── Electron solver at N=32 ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def electron_sol():
    """Run the SCF solver once for the electron at N=32 (module-level cache)."""
    return solve_eigenmode(lfm.ELECTRON, N=32, verbose=False)


class TestElectronSolverN32:
    def test_returns_soliton_solution(self, electron_sol):
        assert isinstance(electron_sol, SolitonSolution)

    def test_grid_size_preserved(self, electron_sol):
        assert electron_sol.N == 32

    def test_chi_min_positive(self, electron_sol):
        """Critical Phase 0 constraint: chi must never go negative."""
        assert electron_sol.chi_min > 0.0, (
            f"Z2 vacuum flip detected: chi_min={electron_sol.chi_min:.3f}"
        )

    def test_chi_shape(self, electron_sol):
        assert electron_sol.chi.shape == (32, 32, 32)

    def test_psi_shape(self, electron_sol):
        assert electron_sol.psi_r.shape == (32, 32, 32)

    def test_energy_positive(self, electron_sol):
        assert electron_sol.energy > 0.0

    def test_energy_history_populated(self, electron_sol):
        assert len(electron_sol.energy_history) >= 2

    def test_eigenvalue_positive(self, electron_sol):
        assert electron_sol.eigenvalue > 0.0

    def test_particle_preserved(self, electron_sol):
        assert electron_sol.particle is lfm.ELECTRON

    def test_converged(self, electron_sol):
        assert electron_sol.converged, (
            f"Solver did not converge in {electron_sol.cycles} cycles; "
            f"chi_min={electron_sol.chi_min:.3f}  energy_hist={electron_sol.energy_history}"
        )

    def test_chi_below_chi0(self, electron_sol):
        import lfm.constants as c

        # The chi well must be genuinely deepened below chi0
        assert electron_sol.chi_min < c.CHI0


# ── Solver parameter validation ──────────────────────────────────────────────


class TestSolverParameterValidation:
    def test_rejects_too_few_steps(self):
        with pytest.raises(ValueError, match="steps_per_cycle"):
            solve_eigenmode(lfm.ELECTRON, N=16, steps_per_cycle=100)

    def test_default_position_is_centre(self, electron_sol):
        N = electron_sol.N
        cx = cy = cz = N // 2
        # chi minimum should be near grid centre
        chi = electron_sol.chi
        centre_chi = float(chi[cx, cy, cz])
        assert centre_chi <= electron_sol.chi_min + 2.0, (
            "chi minimum is far from grid centre — soliton may have drifted"
        )
