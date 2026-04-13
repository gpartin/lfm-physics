"""
Leapfrog Time Integrator
========================

Coupled GOV-01 + GOV-02 evolution using Verlet (leapfrog) integration.

GOV-01: Ψⁿ⁺¹ = 2Ψⁿ − Ψⁿ⁻¹ + Δt²[c²∇²Ψⁿ − (χⁿ)²Ψⁿ]
GOV-02: χⁿ⁺¹ = 2χⁿ − χⁿ⁻¹ + Δt²[c²∇²χⁿ − (κ/χ₀)χⁿ(|Ψⁿ|² − E₀²) − 4λ_H·χⁿ((χⁿ)²−χ₀²)]

v28.0: χ-dependent coupling (κ/χ₀)·χ replaces constant κ for exact Lagrangian
consistency and Noether-conserved Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.core.stencils import laplacian_19pt

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SimulationState:
    """Complete simulation state at a given time step.

    Stores current and previous fields for leapfrog integration.
    """

    # Wave field (Ψ) — current and previous
    psi: NDArray[np.floating]
    """Current Ψ field. Shape depends on FieldLevel:
    - REAL: (N, N, N)
    - COMPLEX: (2, N, N, N) — [real, imag]
    - COLOR: (n_colors, 2, N, N, N) — [color, real/imag, x, y, z]
    """
    psi_prev: NDArray[np.floating]
    """Previous Ψ field (same shape as psi)."""

    # Chi field — current and previous
    chi: NDArray[np.floating]
    """Current χ field. Shape: (N, N, N)."""
    chi_prev: NDArray[np.floating]
    """Previous χ field. Shape: (N, N, N)."""

    # Boundary mask
    boundary_mask: NDArray[np.bool_] | None = None
    """True where boundary is frozen. Shape: (N, N, N)."""

    step: int = 0
    """Current time step number."""


def create_initial_state(config: SimulationConfig) -> SimulationState:
    """Create a blank initial state with χ = χ₀ everywhere and Ψ = 0.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.

    Returns
    -------
    SimulationState
        Initial state ready for field initialization.
    """
    N = config.grid_size
    dtype = np.float64

    # Chi = chi0 everywhere
    chi = np.full((N, N, N), config.chi0, dtype=dtype)
    chi_prev = chi.copy()

    # Psi = 0
    if config.field_level == FieldLevel.REAL:
        psi = np.zeros((N, N, N), dtype=dtype)
    elif config.field_level == FieldLevel.COMPLEX:
        psi = np.zeros((2, N, N, N), dtype=dtype)
    else:  # COLOR
        psi = np.zeros((config.n_colors, 2, N, N, N), dtype=dtype)
    psi_prev = psi.copy()

    # Boundary mask
    mask = None
    if config.boundary_type == BoundaryType.FROZEN:
        mask = _spherical_boundary_mask(N, config.boundary_fraction)

    return SimulationState(
        psi=psi,
        psi_prev=psi_prev,
        chi=chi,
        chi_prev=chi_prev,
        boundary_mask=mask,
        step=0,
    )


def step_leapfrog(state: SimulationState, config: SimulationConfig) -> None:
    """Advance the simulation by one leapfrog step (in-place).

    Updates both Ψ and χ fields using GOV-01 and GOV-02.

    Parameters
    ----------
    state : SimulationState
        Current simulation state (modified in-place).
    config : SimulationConfig
        Simulation parameters.
    """
    dt2 = config.dt**2
    c2 = config.c**2

    if config.field_level == FieldLevel.REAL:
        _step_real(state, config, dt2, c2)
    elif config.field_level == FieldLevel.COMPLEX:
        _step_complex(state, config, dt2, c2)
    else:
        _step_color(state, config, dt2, c2)

    state.step += 1


def _step_real(
    state: SimulationState,
    config: SimulationConfig,
    dt2: float,
    c2: float,
) -> None:
    """Leapfrog step for real scalar field (gravity only)."""
    E = state.psi
    E_prev = state.psi_prev
    chi = state.chi
    chi_prev = state.chi_prev

    # GOV-01: E_next = 2E - E_prev + dt²[c²∇²E - χ²E]
    lap_E = laplacian_19pt(E)
    E_next = 2.0 * E - E_prev + dt2 * (c2 * lap_E - chi**2 * E)

    # Energy density for GOV-02
    energy_density = E**2

    # GOV-02 v28.0: chi_next = 2chi - chi_prev + dt²[c²∇²chi - (κ/χ₀)χ(E² - E₀²) - self_interaction]
    lap_chi = laplacian_19pt(chi)
    chi_source = (config.kappa / config.chi0) * chi * (energy_density - config.e0_sq)

    chi_accel = c2 * lap_chi - chi_source
    if config.lambda_self > 0:
        chi_accel -= 4 * config.lambda_self * chi * (chi**2 - config.chi0**2)

    chi_next = 2.0 * chi - chi_prev + dt2 * chi_accel

    # Apply frozen boundary conditions
    if state.boundary_mask is not None:
        E_next[state.boundary_mask] = 0.0
        chi_next[state.boundary_mask] = config.chi0

    # Swap: current → prev, next → current
    state.psi_prev = E
    state.psi = E_next
    state.chi_prev = chi
    state.chi = chi_next


def _step_complex(
    state: SimulationState,
    config: SimulationConfig,
    dt2: float,
    c2: float,
) -> None:
    """Leapfrog step for complex scalar field (gravity + EM)."""
    Pr = state.psi[0]  # Real part
    Pi = state.psi[1]  # Imaginary part
    Pr_prev = state.psi_prev[0]
    Pi_prev = state.psi_prev[1]
    chi = state.chi
    chi_prev = state.chi_prev

    # GOV-01 for real and imaginary parts
    lap_Pr = laplacian_19pt(Pr)
    lap_Pi = laplacian_19pt(Pi)
    chi_sq = chi**2

    Pr_next = 2.0 * Pr - Pr_prev + dt2 * (c2 * lap_Pr - chi_sq * Pr)
    Pi_next = 2.0 * Pi - Pi_prev + dt2 * (c2 * lap_Pi - chi_sq * Pi)

    # |Ψ|² = Pr² + Pi²
    energy_density = Pr**2 + Pi**2

    # GOV-02 v28.0
    lap_chi = laplacian_19pt(chi)
    chi_source = (config.kappa / config.chi0) * chi * (energy_density - config.e0_sq)
    chi_accel = c2 * lap_chi - chi_source
    if config.lambda_self > 0:
        chi_accel -= 4 * config.lambda_self * chi * (chi**2 - config.chi0**2)

    chi_next = 2.0 * chi - chi_prev + dt2 * chi_accel

    # Frozen boundary
    if state.boundary_mask is not None:
        Pr_next[state.boundary_mask] = 0.0
        Pi_next[state.boundary_mask] = 0.0
        chi_next[state.boundary_mask] = config.chi0

    # Swap
    new_psi = np.stack([Pr_next, Pi_next])
    state.psi_prev = state.psi
    state.psi = new_psi
    state.chi_prev = chi
    state.chi = chi_next


def _step_color(
    state: SimulationState,
    config: SimulationConfig,
    dt2: float,
    c2: float,
) -> None:
    """Leapfrog step for 3-color complex field (all four forces).

    v14: Color variance term −κ_c·f_c·Σ|Ψₐ|² in GOV-02
    v15: Cross-color coupling −ε_cc·χ²·(Ψₐ − Ψ̄) in GOV-01
    """
    psi = state.psi  # shape: (n_colors, 2, N, N, N)
    psi_prev = state.psi_prev
    chi = state.chi
    chi_prev = state.chi_prev
    chi_sq = chi**2

    n_colors = config.n_colors
    psi_next = np.empty_like(psi)

    # Energy density: colorblind sum Σₐ|Ψₐ|²
    energy_density = np.zeros_like(chi)
    # Per-color energy for f_c computation (v14)
    color_energy = np.zeros((n_colors,) + chi.shape, dtype=chi.dtype)

    # Pre-compute color average for cross-color coupling (v15)
    if config.epsilon_cc > 0:
        Pr_avg = np.mean(psi[:, 0], axis=0)  # (N,N,N)
        Pi_avg = np.mean(psi[:, 1], axis=0)

    for a in range(n_colors):
        Pr = psi[a, 0]
        Pi = psi[a, 1]
        Pr_prev = psi_prev[a, 0]
        Pi_prev = psi_prev[a, 1]

        # GOV-01 per color
        lap_Pr = laplacian_19pt(Pr)
        lap_Pi = laplacian_19pt(Pi)

        Pr_new = 2.0 * Pr - Pr_prev + dt2 * (c2 * lap_Pr - chi_sq * Pr)
        Pi_new = 2.0 * Pi - Pi_prev + dt2 * (c2 * lap_Pi - chi_sq * Pi)

        # v15: cross-color coupling −ε_cc·χ²·(Ψₐ − Ψ̄)
        if config.epsilon_cc > 0:
            Pr_new -= dt2 * config.epsilon_cc * chi_sq * (Pr - Pr_avg)
            Pi_new -= dt2 * config.epsilon_cc * chi_sq * (Pi - Pi_avg)

        psi_next[a, 0] = Pr_new
        psi_next[a, 1] = Pi_new

        ea = Pr**2 + Pi**2
        color_energy[a] = ea
        energy_density += ea

    # v14: compute normalized color variance f_c
    # f_c = [Σₐ|Ψₐ|⁴ / (Σₐ|Ψₐ|²)²] − 1/3
    color_variance_term = np.zeros_like(chi)
    if config.kappa_c > 0:
        sum_sq = np.sum(color_energy**2, axis=0)  # Σₐ|Ψₐ|⁴
        total_sq = energy_density**2  # (Σₐ|Ψₐ|²)²
        safe = total_sq > 1e-30
        f_c = np.where(safe, sum_sq / total_sq - 1.0 / n_colors, 0.0)
        color_variance_term = config.kappa_c * f_c * energy_density

    # GOV-02 v28.0 with colorblind source + color variance
    lap_chi = laplacian_19pt(chi)
    chi_source = (config.kappa / config.chi0) * chi * (energy_density - config.e0_sq)
    chi_accel = c2 * lap_chi - chi_source - color_variance_term
    if config.lambda_self > 0:
        chi_accel -= 4 * config.lambda_self * chi * (chi**2 - config.chi0**2)

    chi_next = 2.0 * chi - chi_prev + dt2 * chi_accel

    # Frozen boundary
    if state.boundary_mask is not None:
        psi_next[:, :, state.boundary_mask] = 0.0
        chi_next[state.boundary_mask] = config.chi0

    # Swap
    state.psi_prev = psi
    state.psi = psi_next
    state.chi_prev = chi
    state.chi = chi_next


def _spherical_boundary_mask(
    N: int,
    boundary_fraction: float,
) -> NDArray[np.bool_]:
    """Create a spherical frozen boundary mask.

    Points where r > (1 - boundary_fraction) * r_max are frozen.

    Parameters
    ----------
    N : int
        Grid size per axis.
    boundary_fraction : float
        Fraction of radius for the frozen shell.

    Returns
    -------
    ndarray of bool, shape (N, N, N)
        True where the boundary should be frozen.
    """
    center = N / 2.0
    r_max = N / 2.0
    r_freeze = (1.0 - boundary_fraction) * r_max

    coords = np.arange(N) - center + 0.5
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)

    return r_freeze < R
