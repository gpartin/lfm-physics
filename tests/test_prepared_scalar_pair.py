from __future__ import annotations

import numpy as np

import lfm
from lfm.particles.prepared import (
    install_prepared_scalar_pair,
    prepared_mode_wave_hamiltonian,
)
from lfm.particles.solver import SolitonSolution


def _solution(grid_size: int, amplitude: float) -> SolitonSolution:
    coords = np.arange(grid_size, dtype=np.float32) - grid_size // 2
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    envelope = amplitude * np.exp(-(x * x + y * y + z * z) / 8.0)
    chi = lfm.CHI0 - 0.1 * envelope * envelope
    return SolitonSolution(
        psi_r=envelope.astype(np.float32),
        psi_i=None,
        chi=chi.astype(np.float32),
        chi_min=float(np.min(chi)),
        energy=float(np.sum(envelope * envelope)),
        eigenvalue=18.9,
        converged=True,
        cycles=1,
        particle=None,
        N=grid_size,
    )


def test_install_prepared_pair_sets_distinct_live_layers() -> None:
    config = lfm.SimulationConfig(
        grid_size=16,
        field_level=lfm.FieldLevel.COMPLEX,
        dt=0.02,
    )
    sim = lfm.Simulation(config, backend="cpu")
    solution_a = _solution(16, 1.0)
    solution_b = _solution(16, 0.5)
    install_prepared_scalar_pair(
        sim,
        solution_a,
        solution_b,
        position_a=(5.0, 8.0, 8.0),
        position_b=(11.0, 8.0, 8.0),
        velocity_a=(0.0, -0.01, 0.0),
        velocity_b=(0.0, 0.03, 0.0),
    )
    state = sim.phase_space_snapshot()
    assert state["psi_imag"] is not None
    assert not np.allclose(state["psi_real"], state["psi_real_prev"])
    assert not np.array_equal(state["chi"], state["chi_prev"])
    assert len(lfm.find_peaks(sim.energy_density, n=2, min_separation=3)) == 2


def test_prepared_mode_energy_increases_with_amplitude() -> None:
    low = _solution(16, 0.5)
    high = _solution(16, 1.0)
    low_energy = prepared_mode_wave_hamiltonian(low, dt=0.02)
    high_energy = prepared_mode_wave_hamiltonian(high, dt=0.02)
    assert high_energy > low_energy


def test_relative_pi_phase_reverses_second_mode_interference() -> None:
    config = lfm.SimulationConfig(
        grid_size=16,
        field_level=lfm.FieldLevel.COMPLEX,
        dt=0.02,
    )
    solution_a = _solution(16, 1.0)
    solution_b = _solution(16, 0.5)
    sim_same = lfm.Simulation(config, backend="cpu")
    install_prepared_scalar_pair(
        sim_same,
        solution_a,
        solution_b,
        position_a=(6.0, 8.0, 8.0),
        position_b=(10.0, 8.0, 8.0),
        velocity_a=(0.0, 0.0, 0.0),
        velocity_b=(0.0, 0.0, 0.0),
    )
    sim_opposite = lfm.Simulation(config, backend="cpu")
    install_prepared_scalar_pair(
        sim_opposite,
        solution_a,
        solution_b,
        position_a=(6.0, 8.0, 8.0),
        position_b=(10.0, 8.0, 8.0),
        velocity_a=(0.0, 0.0, 0.0),
        velocity_b=(0.0, 0.0, 0.0),
        phase_b=np.pi,
    )
    midpoint = (8, 8, 8)
    assert sim_same.energy_density[midpoint] > sim_opposite.energy_density[midpoint]
