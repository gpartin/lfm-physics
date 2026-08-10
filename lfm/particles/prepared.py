"""Prepared scalar-mode composition for live LFM simulations.

The helpers in this module create initial leapfrog layers from generic relaxed
scalar envelopes. They do not assign a particle catalog identity and do not
claim source-free formation or stability.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from lfm.analysis.energy import total_energy
from lfm.config import FieldLevel
from lfm.particles.solver import SolitonSolution, boost_fields

if TYPE_CHECKING:
    from lfm.simulation import Simulation


def _shift_mode(
    solution: SolitonSolution,
    position: tuple[float, float, float],
    chi0: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_size = solution.N
    center = grid_size // 2
    envelope = np.asarray(solution.psi_r, dtype=np.float32).copy()
    chi_delta = np.asarray(solution.chi, dtype=np.float32) - np.float32(chi0)
    for axis in range(3):
        shift = int(round(position[axis])) - center
        if shift:
            envelope = np.roll(envelope, shift, axis=axis)
            chi_delta = np.roll(chi_delta, shift, axis=axis)
    return envelope, np.float32(chi0) + chi_delta


def _time_harmonic_layers(
    envelope: np.ndarray,
    chi: np.ndarray,
    velocity: tuple[float, float, float],
    *,
    dt: float,
    omega: float,
    chi0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    speed_sq = float(sum(component * component for component in velocity))
    if speed_sq > 1.0e-30:
        return boost_fields(
            envelope,
            chi,
            velocity,
            dt=dt,
            omega=omega,
            chi0=chi0,
        )

    phase = float(omega * dt)
    zeros = np.zeros_like(envelope)
    real_prev = (envelope * math.cos(phase)).astype(np.float32)
    imag_prev = (envelope * math.sin(phase)).astype(np.float32)
    return envelope.copy(), zeros, real_prev, imag_prev, chi.copy()


def install_prepared_scalar_pair(
    sim: Simulation,
    solution_a: SolitonSolution,
    solution_b: SolitonSolution,
    *,
    position_a: tuple[float, float, float],
    position_b: tuple[float, float, float],
    velocity_a: tuple[float, float, float],
    velocity_b: tuple[float, float, float],
    phase_a: float = 0.0,
    phase_b: float = 0.0,
) -> None:
    """Install two generic prepared modes into an empty complex simulation."""
    if sim.step != 0:
        raise ValueError("prepared modes can only be installed at step zero")
    if sim.config.field_level != FieldLevel.COMPLEX:
        raise ValueError("prepared moving modes require a complex field")
    if solution_a.N != sim.config.grid_size or solution_b.N != sim.config.grid_size:
        raise ValueError("solution grid size must match the simulation")

    chi0 = float(sim.config.chi0)
    dt = float(sim.config.dt)
    env_a, chi_a = _shift_mode(solution_a, position_a, chi0)
    env_b, chi_b = _shift_mode(solution_b, position_b, chi0)
    layers_a = _time_harmonic_layers(
        env_a,
        chi_a,
        velocity_a,
        dt=dt,
        omega=float(solution_a.eigenvalue),
        chi0=chi0,
    )
    layers_b = _time_harmonic_layers(
        env_b,
        chi_b,
        velocity_b,
        dt=dt,
        omega=float(solution_b.eigenvalue),
        chi0=chi0,
    )

    def rotate(
        layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        phase: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cosine = math.cos(phase)
        sine = math.sin(phase)
        real = layers[0] * cosine - layers[1] * sine
        imag = layers[0] * sine + layers[1] * cosine
        real_prev = layers[2] * cosine - layers[3] * sine
        imag_prev = layers[2] * sine + layers[3] * cosine
        return real, imag, real_prev, imag_prev, layers[4]

    layers_a = rotate(layers_a, phase_a)
    layers_b = rotate(layers_b, phase_b)

    sim.set_psi_real(layers_a[0] + layers_b[0])
    sim.set_psi_imag(layers_a[1] + layers_b[1])
    sim.set_psi_real_prev(layers_a[2] + layers_b[2])
    sim.set_psi_imag_prev(layers_a[3] + layers_b[3])
    chi_current = chi0 + (chi_a - chi0) + (chi_b - chi0)
    chi_previous = chi0 + (layers_a[4] - chi0) + (layers_b[4] - chi0)
    sim.set_chi(chi_current)
    sim.set_chi_prev(chi_previous)


def prepared_mode_wave_hamiltonian(
    solution: SolitonSolution,
    *,
    dt: float,
    c: float = 1.0,
    chi0: float = 19.0,
) -> float:
    """Return the isolated time-harmonic wave Hamiltonian of one mode."""
    envelope = np.asarray(solution.psi_r, dtype=np.float32)
    real, imag, real_prev, imag_prev, _ = _time_harmonic_layers(
        envelope,
        np.asarray(solution.chi, dtype=np.float32),
        (0.0, 0.0, 0.0),
        dt=dt,
        omega=float(solution.eigenvalue),
        chi0=chi0,
    )
    return total_energy(
        real,
        real_prev,
        np.asarray(solution.chi, dtype=np.float32),
        dt,
        c,
        imag,
        imag_prev,
    )


__all__ = ["install_prepared_scalar_pair", "prepared_mode_wave_hamiltonian"]
