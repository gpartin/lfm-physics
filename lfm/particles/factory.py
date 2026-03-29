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

import numpy as np

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import CHI0
from lfm.particles.catalog import (
    Particle,
    get_particle,
)
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
    sim: Simulation | None = None,
    N: int = 64,
    position: tuple | None = None,
    velocity: tuple = (0.0, 0.0, 0.0),
    use_eigenmode: bool = True,
    chi0: float = CHI0,
    sigma: float | None = None,
    amplitude: float | None = None,
) -> PlacedParticle:
    """Create a particle and place it into a simulation.

    Delegates to :meth:`Simulation.place_particle`, which handles
    eigenmode relaxation, phase-gradient velocity boost, and charge
    phase automatically.  The user does not need to understand the
    internal motion mechanism.

    Parameters
    ----------
    name : str
        Particle name from the catalog:
        ``"electron"``, ``"positron"``, ``"muon"``, ``"proton"``, etc.
    sim : Simulation, optional
        Existing simulation to place the particle into.  If ``None``,
        a new COMPLEX simulation is created.
    N : int
        Grid size used when creating a new simulation.  Ignored if ``sim``
        is provided.
    position : tuple(float, float, float), optional
        Particle centre in grid coordinates.  Defaults to the grid centre.
    velocity : tuple(float, float, float)
        Initial velocity in units of c: ``(vx, vy, vz)``.
    use_eigenmode : bool
        If ``True`` (default), uses eigenmode relaxation for a
        self-consistent bound state.  If ``False``, uses a fast Gaussian
        seed via :meth:`Simulation.place_soliton` (less accurate but instant).
    chi0 : float
        Background χ value.  Defaults to ``CHI0 = 19``.
    sigma : float or None
        Override the Gaussian width.  If ``None``, chosen automatically.
    amplitude : float or None
        Override the peak amplitude.  If ``None``, chosen automatically.

    Returns
    -------
    PlacedParticle
        Dataclass with ``.sim``, ``.particle``, ``.position``,
        ``.velocity``, and ``.energy``.

    Examples
    --------
    >>> placed = create_particle("electron")
    >>> placed.sim.run(500)

    >>> placed = create_particle("muon", velocity=(0.05, 0, 0))
    >>> placed.sim.run(2000)
    """
    particle = get_particle(name)

    # --- Create or reuse simulation ---
    if sim is not None:
        out_sim = sim
    else:
        fl = FieldLevel(max(particle.field_level, 1))  # at least COMPLEX

        # Auto-select timestep: moving particles need finer dt
        has_vel = sum(v**2 for v in velocity) > 1e-20
        if has_vel:
            from lfm.constants import DT_MOTION

            config = SimulationConfig(
                grid_size=N,
                field_level=fl,
                boundary_type=BoundaryType.FROZEN,
                chi0=chi0,
                dt=DT_MOTION,
            )
        else:
            config = SimulationConfig(
                grid_size=N,
                field_level=fl,
                boundary_type=BoundaryType.FROZEN,
                chi0=chi0,
            )
        out_sim = Simulation(config)

    grid_N = out_sim.config.grid_size
    half = float(grid_N // 2)
    final_pos: tuple[float, float, float] = (
        (float(position[0]), float(position[1]), float(position[2]))
        if position is not None
        else (half, half, half)
    )

    if use_eigenmode:
        # Eigenmode path: full physics via place_particle()
        out_sim.place_particle(
            particle,
            position=final_pos,
            velocity=velocity,
            amplitude=amplitude,
            sigma=sigma,
        )
    else:
        # Fast Gaussian seed path (no eigenmode relaxation)
        from lfm.particles.catalog import amplitude_for_particle, sigma_for_particle

        vx, vy, vz = velocity
        v_mag = float(np.sqrt(vx**2 + vy**2 + vz**2))
        amp = amplitude if amplitude is not None else amplitude_for_particle(particle, grid_N)
        sig = sigma if sigma is not None else sigma_for_particle(particle, grid_N)
        phase = float(getattr(particle, "phase", 0.0))
        vel_arg = velocity if v_mag > 0.0 else None

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


# ---------------------------------------------------------------------------
# Two-particle convenience constructor
# ---------------------------------------------------------------------------


def create_two_particles(
    name_a: str,
    name_b: str,
    separation: int = 16,
    N: int = 64,
    axis: int = 0,
    velocity_a: tuple = (0.0, 0.0, 0.0),
    velocity_b: tuple = (0.0, 0.0, 0.0),
    chi0: float = CHI0,
) -> tuple[PlacedParticle, PlacedParticle]:
    """Place two particles in a shared simulation for two-body experiments.

    Both particles are placed as eigenmode-relaxed solitons in a single
    N³ simulation.  Uses :meth:`Simulation.place_particle` for each,
    giving both proper self-consistent bound states and velocity boosts.

    The pair is centred in the grid:
      - particle A at (centre − separation/2) along ``axis``
      - particle B at (centre + separation/2) along ``axis``

    Parameters
    ----------
    name_a, name_b : str
        Particle names from the catalog (e.g. ``"up_quark"``).
    separation : int
        Centre-to-centre distance in lattice cells.
    N : int
        Grid size per axis.
    axis : int
        Spatial axis for the separation (0 = x, 1 = y, 2 = z).
    velocity_a, velocity_b : tuple(float, float, float)
        Initial velocities in units of c.
    chi0 : float
        Background χ value.

    Returns
    -------
    tuple(PlacedParticle, PlacedParticle)
        Both :class:`PlacedParticle` instances share the same ``.sim``.

    Examples
    --------
    >>> pa, pb = create_two_particles("up_quark", "up_quark", separation=12)
    >>> pa.sim.run(2000)   # evolves both together
    """
    particle_a = get_particle(name_a)
    particle_b = get_particle(name_b)

    # Choose field level: take the higher of the two particles, at least COMPLEX
    fl_int = max(particle_a.field_level, particle_b.field_level, 1)
    fl = FieldLevel(fl_int)

    config = SimulationConfig(
        grid_size=N,
        field_level=fl,
        boundary_type=BoundaryType.FROZEN,
        chi0=chi0,
    )
    shared_sim = Simulation(config)

    half = N // 2
    offset = separation // 2

    # Build 3-D position tuples
    pos_a_list = [float(half)] * 3
    pos_b_list = [float(half)] * 3
    pos_a_list[axis] = float(half - offset)
    pos_b_list[axis] = float(half + offset)
    pos_a = (pos_a_list[0], pos_a_list[1], pos_a_list[2])
    pos_b = (pos_b_list[0], pos_b_list[1], pos_b_list[2])

    shared_sim.place_particle(particle_a, position=pos_a, velocity=velocity_a)
    shared_sim.place_particle(particle_b, position=pos_b, velocity=velocity_b)

    try:
        m = shared_sim.metrics()
        total_energy_val = float(m.get("energy_total", 0.0))
    except Exception:
        total_energy_val = 0.0

    placed_a = PlacedParticle(
        sim=shared_sim,
        particle=particle_a,
        position=pos_a,
        velocity=velocity_a,
        energy=total_energy_val,
    )
    placed_b = PlacedParticle(
        sim=shared_sim,
        particle=particle_b,
        position=pos_b,
        velocity=velocity_b,
        energy=total_energy_val,
    )
    return placed_a, placed_b


# ---------------------------------------------------------------------------
# Collision factory
# ---------------------------------------------------------------------------


@dataclass
class CollisionSetup:
    """Result of :func:`create_collision`.

    Attributes
    ----------
    sim : Simulation
        Simulation containing both particles aimed at each other.
    particle_a, particle_b : Particle
        Particle specifications from the catalog.
    pos_a, pos_b : tuple[float, float, float]
        Initial centre positions in grid coordinates.
    vel_a, vel_b : tuple[float, float, float]
        Velocities in units of c.
    cm_energy : float
        Centre-of-mass energy as the sum of rest energies plus kinetic
        contribution from the imposed velocities.
    """

    sim: Simulation
    particle_a: Particle
    particle_b: Particle
    pos_a: tuple
    pos_b: tuple
    vel_a: tuple
    vel_b: tuple
    cm_energy: float


def create_collision(
    name_a: str,
    name_b: str,
    speed: float = 0.05,
    N: int = 64,
    separation: int | None = None,
    axis: int = 0,
    chi0: float = CHI0,
) -> CollisionSetup:
    """Set up a head-on particle collision in a shared simulation.

    Both particles are placed as eigenmode-relaxed solitons moving
    toward each other via :meth:`Simulation.place_particle`.  This
    handles eigenmode physics, phase-gradient boosts, and charge
    phases automatically.

    Parameters
    ----------
    name_a, name_b : str
        Particle names from the catalog.
    speed : float
        Speed of each particle (in units of c).  Both particles start with
        this magnitude, moving inward toward each other.  Must be in (0, 0.8].
    N : int
        Grid size per axis.
    separation : int or None
        Centre-to-centre distance in lattice cells.  If ``None``,
        defaults to ``N // 2``.
    axis : int
        Spatial axis along which particles approach (0=x, 1=y, 2=z).
    chi0 : float
        Background χ value.

    Returns
    -------
    CollisionSetup
        Ready-to-run simulation with both particles aimed at each other.
        Call ``setup.sim.run(steps)`` to evolve the collision.

    Examples
    --------
    >>> setup = create_collision("proton", "antiproton", speed=0.06, N=128)
    >>> setup.sim.run(10_000)
    >>> print(setup.sim.metrics())
    """
    if not 0 < speed <= 0.8:
        raise ValueError(f"speed must be in (0, 0.8], got {speed}")

    particle_a = get_particle(name_a)
    particle_b = get_particle(name_b)

    sep = separation if separation is not None else N // 2

    # Collision always needs at least COMPLEX for boost phase gradients
    fl_int = max(particle_a.field_level, particle_b.field_level, 1)
    fl = FieldLevel(fl_int)

    config = SimulationConfig(
        grid_size=N,
        field_level=fl,
        boundary_type=BoundaryType.FROZEN,
        chi0=chi0,
    )
    sim = Simulation(config)

    half = N // 2
    offset = sep // 2

    pos_a_list = [float(half)] * 3
    pos_b_list = [float(half)] * 3
    pos_a_list[axis] = float(half - offset)
    pos_b_list[axis] = float(half + offset)
    pos_a = (pos_a_list[0], pos_a_list[1], pos_a_list[2])
    pos_b = (pos_b_list[0], pos_b_list[1], pos_b_list[2])

    # Velocities: A moves toward B (+axis), B moves toward A (-axis)
    vel_a = [0.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]
    vel_a[axis] = +speed
    vel_b[axis] = -speed
    vel_a_t = (vel_a[0], vel_a[1], vel_a[2])
    vel_b_t = (vel_b[0], vel_b[1], vel_b[2])

    sim.place_particle(particle_a, position=pos_a, velocity=vel_a_t)
    sim.place_particle(particle_b, position=pos_b, velocity=vel_b_t)

    # Rough CM energy estimate
    cm_energy = float(particle_a.mass_ratio + particle_b.mass_ratio)

    return CollisionSetup(
        sim=sim,
        particle_a=particle_a,
        particle_b=particle_b,
        pos_a=pos_a,
        pos_b=pos_b,
        vel_a=vel_a_t,
        vel_b=vel_b_t,
        cm_energy=cm_energy,
    )
