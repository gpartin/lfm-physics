"""
Particle Motion Functions for LFM Solitons (Phase 3)
======================================================

Functions for:
  - Creating boosted soliton simulations from solved eigenmodes
  - Measuring soliton center-of-energy position
  - Measuring soliton velocity from momentum density

Critical Phase 0 Constraint
----------------------------
DO NOT use ``place_soliton(velocity=v) + equilibrate()`` for particle motion.
The chi equilibration step creates a well at the **initial** position, which
then traps the moving wave packet (chi self-trapping).  See research/p0_04b.

The correct approach implemented here:
  1. ``solve_eigenmode()``  → self-consistent {psi, chi} at rest
  2. ``boost_soliton_solution()`` → translate BOTH chi AND psi coherently,
     then apply velocity phase gradient to psi
  3. Simulate without re-equilibrating chi (the self-consistent well moves
     with the particle as the leapfrog evolves both fields jointly)

Usage::

    from lfm.particles.catalog import ELECTRON
    from lfm.particles.solver import solve_eigenmode
    from lfm.particles.motion import boost_soliton_solution, measure_center_of_energy

    sol = solve_eigenmode(ELECTRON, N=64)
    sim = boost_soliton_solution(sol, velocity=(0.04, 0.0, 0.0))
    sim.run(1000)
    pos = measure_center_of_energy(sim)
    print("x =", pos[0])          # should have advanced ~40 lattice cells
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import CHI0
from lfm.simulation import Simulation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lfm.particles.solver import SolitonSolution

# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def measure_center_of_energy(sim: Simulation) -> NDArray:
    """Compute the energy-weighted centre-of-mass position of the soliton.

    Uses potential energy density  eps(x) = chi(x)^2 * |Psi(x)|^2  as the
    weighting function.  This is a static-snapshot approximation (no kinetic
    term) which is fast and sufficient for tracking soliton centre positions.

    Parameters
    ----------
    sim : Simulation

    Returns
    -------
    ndarray, shape (3,)
        (cx, cy, cz) in lattice units (floating-point).
    """
    N = sim.config.grid_size
    chi = np.asarray(sim.chi, dtype=np.float64)
    pr = np.asarray(sim.psi_real, dtype=np.float64)
    pi_arr = sim.psi_imag
    pi = np.asarray(pi_arr, dtype=np.float64) if pi_arr is not None else None

    psi2 = pr**2
    if pi is not None:
        psi2 = psi2 + pi**2

    eps = chi**2 * psi2  # potential energy density (proxy for total mass)
    total = float(eps.sum())
    if total < 1e-30:
        # Empty field — return grid centre
        return np.array([N / 2.0, N / 2.0, N / 2.0])

    xs = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    cx = float((eps * X).sum() / total)
    cy = float((eps * Y).sum() / total)
    cz = float((eps * Z).sum() / total)
    return np.array([cx, cy, cz])


def measure_momentum_density(sim: Simulation) -> NDArray:
    """Compute the total energy-flux (momentum) of the soliton.

    Uses the stress-energy tensor momentum density:
        g(x) = -Re[ (dPsi/dt)* grad Psi ]

    For a static snapshot without the time derivative, we fall back to the
    phase-gradient proxy:  g ~= Im(Psi* grad Psi)  (valid for a slowly-
    varying envelope with a well-defined k vector).

    Parameters
    ----------
    sim : Simulation

    Returns
    -------
    ndarray, shape (3,)
        Integrated momentum (px, py, pz) in lattice units.
    """
    pr = np.asarray(sim.psi_real, dtype=np.float64)
    pi_arr = sim.psi_imag
    pi = np.asarray(pi_arr, dtype=np.float64) if pi_arr is not None else None

    if pi is None:
        # Real field: phase current = 0 (neutral particle, no net momentum in phase)
        # Use finite-difference dPsi/dt from prev snapshot if available
        pr_prev = getattr(sim, "_psi_r_prev", None)
        if pr_prev is not None:
            dpr = (pr - np.asarray(pr_prev, dtype=np.float64)) / sim.config.dt
            # g_x = -dpr * d(pr)/dx
            gx = -np.sum(dpr * np.gradient(pr, axis=0))
            gy = -np.sum(dpr * np.gradient(pr, axis=1))
            gz = -np.sum(dpr * np.gradient(pr, axis=2))
            return np.array([gx, gy, gz])
        # No prev snapshot: momentum undefined → return zeros
        return np.zeros(3)

    # Complex field: Im(Psi* grad Psi) = pr*grad(pi) - pi*grad(pr)
    gx = float(np.sum(pr * np.gradient(pi, axis=0) - pi * np.gradient(pr, axis=0)))
    gy = float(np.sum(pr * np.gradient(pi, axis=1) - pi * np.gradient(pr, axis=1)))
    gz = float(np.sum(pr * np.gradient(pi, axis=2) - pi * np.gradient(pr, axis=2)))
    return np.array([gx, gy, gz])


def measure_velocity(sim: Simulation) -> NDArray:
    """Estimate the soliton's current velocity vector.

    For a complex field, uses the ratio of total momentum to total energy.
    For a real field, uses the prev-snapshot time derivative.

    Returns
    -------
    ndarray, shape (3,)
        (vx, vy, vz) in lattice units per step.
    """
    chi = np.asarray(sim.chi, dtype=np.float64)
    pr = np.asarray(sim.psi_real, dtype=np.float64)
    pi_arr = sim.psi_imag
    pi = np.asarray(pi_arr, dtype=np.float64) if pi_arr is not None else None

    psi2 = pr**2
    if pi is not None:
        psi2 = psi2 + pi**2
    energy = float((chi**2 * psi2).sum())

    if energy < 1e-30:
        return np.zeros(3)

    momentum = measure_momentum_density(sim)
    # v ~ p/E  (natural units)
    return momentum / energy


# ---------------------------------------------------------------------------
# Boost function: eigenmode → moving simulation
# ---------------------------------------------------------------------------


def boost_soliton_solution(
    sol: SolitonSolution,
    velocity: tuple[float, float, float],
    position: tuple[int, int, int] | None = None,
    chi0: float = CHI0,
    c: float = 1.0,
) -> Simulation:
    """Create a moving-particle simulation from a solved eigenmode.

    **Physics: Polaron Self-Trapping**

    LFM solitons are *polaron-like*: the chi field (GOV-02) responds to
    |Psi|^2 on a timescale tau_chi ~ 1 / sqrt(kappa * A^2) ~ 1 time unit.
    At sub-c velocities, psi moves only sigma * v during that timescale,
    so chi forms its well around psi before psi can escape.  With fully
    coupled GOV-01+GOV-02, solitons at v < v_crit ~ 0.04c are self-trapped.

    **Approach**

    This function starts with *flat chi* (chi = chi0 everywhere) so that the
    soliton can travel freely in the short term.  For velocity-tracking
    experiments use ``sim.run(..., evolve_chi=False)`` so chi never traps the
    packet.  For long-time polaron dynamics (chi forms, soliton self-traps),
    run with ``evolve_chi=True``.

    Starting with flat chi avoids the chi-well-translation trap: if the solved
    chi well (chi_min << chi0) were placed at position and chi then evolved,
    the deep well would simply hold psi in place.

    Phase 0 constraint: v_max ~ 0.08c for reliable group-velocity tracking
    with frozen chi.

    Parameters
    ----------
    sol : SolitonSolution
        A converged eigenmode (typically ``sol.converged == True``).
        Must have ``sol.N`` and ``sol.particle``.
    velocity : (vx, vy, vz)
        Particle velocity in lattice units (|v| < c = 1).
    position : (cx, cy, cz) or None
        Target centre in the new simulation.  Defaults to grid centre.
    chi0 : float
        Background chi value (flat initial chi field).
    c : float
        Wave speed.

    Returns
    -------
    Simulation
        A freshly initialised simulation (flat chi, velocity-boosted psi).
        For velocity measurements call ``sim.run(steps, evolve_chi=False)``.

    Raises
    ------
    ValueError
        If ``|velocity| >= 0.8 * c``.
    """
    v_mag = math.sqrt(sum(vi**2 for vi in velocity))
    if v_mag >= 0.8 * c:
        raise ValueError(
            f"|velocity|={v_mag:.3f} >= 0.8c.  Phase 0 found v_max~0.08c for "
            "reliable lattice momentum encoding.  Use a smaller velocity."
        )

    N = sol.N
    particle = sol.particle

    # Default position: grid centre
    if position is None:
        half = N // 2
        position = (half, half, half)

    # --- Build simulation (flat chi) ---
    # chi starts at chi0 everywhere.  We do NOT place the eigenmode chi well
    # because that would hold psi stationary.  chi forms naturally around
    # the moving psi if evolve_chi=True is chosen.
    fl = FieldLevel(particle.field_level) if particle is not None else FieldLevel.REAL
    # Velocity encoding requires complex field (imaginary component).
    if v_mag > 0.0 and fl == FieldLevel.REAL:
        fl = FieldLevel.COMPLEX

    config = SimulationConfig(
        grid_size=N,
        field_level=fl,
        boundary_type=BoundaryType.FROZEN,
        chi0=chi0,
        c=c,
    )
    sim = Simulation(config)
    # chi is left at the default flat chi0.

    # --- Estimate amplitude and width from the solved eigenmode ---
    # We use a Gaussian wave packet (not the raw eigenmode profile) because:
    #  * eigenmode was shaped for chi_min < chi0 — wrong shape for flat chi=chi0
    #  * Gaussian correctly represents a free-particle traveling wave packet
    psi2_field = sol.psi_r.astype(np.float64) ** 2
    if sol.psi_i is not None:
        psi2_field += sol.psi_i.astype(np.float64) ** 2
    amp = float(np.sqrt(psi2_field.max()))

    # Width: RMS radius of |psi|^2 about its centre
    orig_half = N // 2
    xs = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    total_psi2 = float(psi2_field.sum())
    xc = float((psi2_field * X).sum() / total_psi2)
    r2 = (X - xc) ** 2 + (Y - orig_half) ** 2 + (Z - orig_half) ** 2
    sigma_rms = float(math.sqrt((psi2_field * r2).sum() / total_psi2))
    sigma_gauss = max(2.0, sigma_rms)

    # Phase from particle (electron=0, positron=pi)
    particle_phase = float(particle.phase) if particle is not None else 0.0

    # Use place_soliton to place a Gaussian wave packet at position with velocity
    sim.place_soliton(
        position=cast("tuple[float, float, float]", tuple(float(p) for p in position)),
        amplitude=amp,
        sigma=sigma_gauss,
        phase=particle_phase,
        velocity=velocity if v_mag > 0.0 else None,
    )
    return sim
