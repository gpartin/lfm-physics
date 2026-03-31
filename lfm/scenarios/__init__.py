"""Scenario presets for LFM simulations.

Provides ready-made collections of celestial bodies (solitons) and
the ``place_bodies`` helper that places them in a simulation and
computes their orbital velocities automatically.

Quick usage::

    import lfm
    from lfm import CelestialBody, BodyType, solar_system, place_bodies

    cfg = lfm.SimulationConfig(grid_size=128, field_level=lfm.FieldLevel.REAL,
                                boundary_type=lfm.BoundaryType.FROZEN)
    sim = lfm.Simulation(cfg)
    bodies      = solar_system()
    body_omegas = place_bodies(sim, bodies, verbose=True)
"""

from lfm.scenarios.celestial import (
    BodyType,
    CelestialBody,
    black_hole_system,
    galaxy_core,
    place_bodies,
    solar_system,
)

__all__ = [
    "BodyType",
    "CelestialBody",
    "solar_system",
    "black_hole_system",
    "galaxy_core",
    "place_bodies",
]
