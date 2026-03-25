"""
LFM Particle Factory (Phase 5)
================================

Top-level factory function that creates a particle and places it into a
simulation in one call.  Combines the catalog, eigenmode solver, and motion
modules into a single convenient entry point.

Quick usage::

    from lfm import create_particle

    # Electron at rest with stable eigenmode
    placed = create_particle("electron")
    placed.sim.run(1000)

    # Electron moving at 0.04c
    placed = create_particle("electron", velocity=(0.04, 0.0, 0.0))
    placed.sim.run(5000)

    # Fast Gaussian seed (no eigenmode, instant)
    placed = create_particle("muon", use_eigenmode=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import CHI0
from lfm.particles.catalog import (
    Particle,
    amplitude_for_particle,
    get_particle,
    sigma_for_particle,
)
from lfm.particles.motion import boost_soliton_solution
from lfm.particles.solver import solve_eigenmode
from lfm.simulation import Simulation

# ---------------------------------------------------------------------------
# PlacedParticle dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlacedParticle:
    """Result of :func:`create_particle`.

    Attributes
    ----------
    sim : Simulation
        Simulation containing the particle.  Ready to call ``.run()``.
    particle : Particle
        Particle specification from the catalog.
    position : tuple[float, float, float]
        Centre of the particle in grid coordinates.
    velocity : tuple[float, float, float]
        Velocity in units of c (vx, vy, vz).
    energy : float
        Total energy metric at creation time (from ``sim.metrics()``).
    """

    sim: Simulation
    particle: Particle
    position: tuple
    velocity: tuple
    energy: float


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_particle(
    name: str,
    sim: Optional[Simulation] = None,
    N: int = 64,
    position: Optional[tuple] = None,
    velocity: tuple = (0.0, 0.0, 0.0),
    use_eigenmode: bool = True,
    chi0: float = CHI0,
    sigma: Optional[float] = None,
    amplitude: Optional[float] = None,
) -> PlacedParticle:
    """Create a particle and place it into a simulation.

    Parameters
    ----------
    name : str
        Particle name from the catalog:
        ``"electron"``, ``"positron"``, ``"muon"``, ``"proton"``, etc.
    sim : Simulation, optional
        Existing simulation to place the particle into.  Only used when
        ``use_eigenmode=False``; when ``use_eigenmode=True`` the eigenmode
        solver always creates a fresh simulation.  If ``None``, a new
        simulation is created (N×N×N, COMPLEX field level).
    N : int
        Grid size used when creating a new simulation.  Ignored if ``sim``
        is provided.
    position : tuple(float, float, float), optional
        Particle centre in grid coordinates.  Defaults to the grid centre
        ``(N/2, N/2, N/2)``.  Ignored when ``use_eigenmode=True`` (the
        eigenmode solver places the particle at centre automatically).
    velocity : tuple(float, float, float)
        Initial velocity in units of c: ``(vx, vy, vz)``.
        When non-zero and ``use_eigenmode=True`` the particle is boosted
        using :func:`boost_soliton_solution`.
    chi0 : float
        Background χ value for the simulation medium.  Defaults to
        ``CHI0 = 19`` (LFM vacuum).  Use a lower value (e.g. ``chi0=1.0``)
        for fast-propagation demonstrations where v_g ≈ 0.618 c is needed
        instead of the ~0.05 c vacuum group velocity.  Passed to
        :func:`solve_eigenmode` (eigenmode path) and :class:`SimulationConfig`
        (Gaussian seed path).
    sigma : float or None
        Override the catalog Gaussian width (in lattice cells).  If ``None``
        (default), the width is derived from the particle's mass ratio via
        :func:`~lfm.particles.catalog.sigma_for_particle`.  Useful for
        wave-optics experiments where a large illumination width is needed.
        Only used when ``use_eigenmode=False``.
    amplitude : float or None
        Override the catalog peak amplitude.  If ``None`` (default), derived
        from the particle mass ratio.  Only used when ``use_eigenmode=False``.
    -------
    PlacedParticle
        Dataclass with ``.sim``, ``.particle``, ``.position``,
        ``.velocity``, and ``.energy``.

    Raises
    ------
    ValueError
        If ``name`` is not a known particle.

    Examples
    --------
    >>> placed = create_particle("electron")
    >>> placed.sim.run(500)

    >>> placed = create_particle("muon", velocity=(0.05, 0, 0))
    >>> placed.sim.run(2000)
    """
    particle = get_particle(name)
    vx, vy, vz = velocity
    v_mag = float(np.sqrt(vx**2 + vy**2 + vz**2))

    if use_eigenmode:
        # ── Eigenmode path: solve SCF, then optionally boost ─────────────
        # Determine grid size from provided sim (if any) or argument N
        solve_N = sim.config.grid_size if sim is not None else N
        sol = solve_eigenmode(particle, N=solve_N, chi0=chi0)

        if v_mag > 0.0:
            out_sim = boost_soliton_solution(sol, velocity=velocity, chi0=chi0)
        else:
            # At rest: directly inject eigenmode fields (psi + self-consistent chi)
            # so the particle starts exactly in its eigenstate with zero velocity.
            fl = FieldLevel(particle.field_level)
            config = SimulationConfig(
                grid_size=solve_N,
                field_level=fl,
                boundary_type=BoundaryType.FROZEN,
                chi0=chi0,
            )
            out_sim = Simulation(config)
            psi_r = sol.psi_r.copy().astype(np.float32)
            # set_psi_real writes both current AND previous leapfrog buffers to
            # the same value (zero velocity initialisation).
            out_sim.psi_real = psi_r
            if sol.psi_i is not None and fl != FieldLevel.REAL:
                out_sim.psi_imag = sol.psi_i.copy().astype(np.float32)
            out_sim.chi = sol.chi.copy().astype(np.float32)
            # Correct the leapfrog prev buffer: for an amplitude-peak standing
            # wave  ψ(t)=A·cos(ωt),  ψ(−Δt)=A·cos(ω·Δt) ≠ A.
            # Using prev=curr gives a 2nd-order velocity error that grows.
            omega = float(sol.eigenvalue) if (sol.eigenvalue and sol.eigenvalue > 0) else 0.0
            if omega > 0.0:
                dt = out_sim.config.dt
                psi_lf_prev = (psi_r * float(np.cos(omega * dt))).astype(np.float32)
                out_sim._evolver.set_psi_real_prev(psi_lf_prev)
                # Also pre-populate the Simulation's user-facing cache so that
                # metrics() computes a non-zero (correct) kinetic energy before
                # the first run() call.
                out_sim._psi_r_prev = psi_lf_prev

        half = float(solve_N // 2)
        final_pos = (half, half, half)

    else:
        # ── Gaussian seed path: fast, no eigenmode ───────────────────────
        if sim is not None:
            out_sim = sim
        else:
            field_level = FieldLevel.COMPLEX if particle.field_level >= 1 else FieldLevel.REAL
            config = SimulationConfig(
                grid_size=N,
                field_level=field_level,
                boundary_type=BoundaryType.FROZEN,
                chi0=chi0,
            )
            out_sim = Simulation(config)
            out_sim.equilibrate()  # flat equilibrium before placing particle

        grid_N = out_sim.config.grid_size
        half = float(grid_N // 2)
        default_pos = (half, half, half)
        final_pos = tuple(position) if position is not None else default_pos

        amp = amplitude if amplitude is not None else amplitude_for_particle(particle, grid_N)
        sig = sigma if sigma is not None else sigma_for_particle(particle, grid_N)
        phase = float(getattr(particle, "phase", 0.0))

        # Determine velocity tuple for place_soliton (None if at rest)
        vel_arg = velocity if v_mag > 0.0 else None

        # Complex field needed if moving (phase gradient) or charged
        if v_mag > 0.0 and out_sim.config.field_level == FieldLevel.REAL:
            # Upgrade to complex simulation
            config2 = SimulationConfig(
                grid_size=grid_N,
                field_level=FieldLevel.COMPLEX,
                boundary_type=out_sim.config.boundary_type,
                chi0=chi0,
            )
            out_sim = Simulation(config2)
            out_sim.equilibrate()

        out_sim.place_soliton(
            position=final_pos,
            amplitude=amp,
            sigma=sig,
            phase=phase,
            velocity=vel_arg,
        )

    # Measure energy at creation time
    try:
        m = out_sim.metrics()
        energy = float(m.get("energy_total", 0.0))
    except Exception:
        energy = 0.0

    return PlacedParticle(
        sim=out_sim,
        particle=particle,
        position=final_pos,
        velocity=velocity,
        energy=energy,
    )
