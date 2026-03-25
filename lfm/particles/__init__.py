"""
lfm.particles — Particle Catalog and Eigenmode Solver
======================================================

Provides the particle specification dataclass, the canonical particle
table, helper functions for simulation parameters, and (Phase 2) the
SCF eigenmode solver.

Quick usage::

    from lfm.particles import ELECTRON, MUON, PROTON
    from lfm.particles import get_particle, amplitude_for_particle

    p = get_particle("muon")
    amp = amplitude_for_particle(p, N=64)
"""

from lfm.particles.catalog import (
    ANTIMUON,
    ANTINEUTRON,
    ANTIPROTON,
    ANTITAU,
    CHARM_QUARK,
    DOWN_QUARK,
    # Pre-defined particle constants
    ELECTRON,
    MUON,
    NEUTRON,
    PARTICLES,
    PHOTON,
    POSITRON,
    PROTON,
    STRANGE_QUARK,
    TAU,
    UP_QUARK,
    Particle,
    amplitude_for_particle,
    get_particle,
    sigma_for_particle,
)
from lfm.particles.composite import (
    AtomState,
    MoleculeState,
    create_atom,
    create_molecule,
    nuclear_chi_well,
)
from lfm.particles.factory import PlacedParticle, create_particle
from lfm.particles.motion import (
    boost_soliton_solution,
    measure_center_of_energy,
    measure_momentum_density,
    measure_velocity,
)
from lfm.particles.solver import SolitonSolution, solve_eigenmode

__all__ = [
    "Particle",
    "PARTICLES",
    "get_particle",
    "amplitude_for_particle",
    "sigma_for_particle",
    # Leptons
    "ELECTRON",
    "POSITRON",
    "MUON",
    "ANTIMUON",
    "TAU",
    "ANTITAU",
    # Quarks
    "UP_QUARK",
    "DOWN_QUARK",
    "STRANGE_QUARK",
    "CHARM_QUARK",
    # Nucleons
    "PROTON",
    "ANTIPROTON",
    "NEUTRON",
    "ANTINEUTRON",
    # Bosons
    "PHOTON",
    # Phase 2: Eigenmode solver
    "SolitonSolution",
    "solve_eigenmode",
    # Phase 3: Particle motion
    "boost_soliton_solution",
    "measure_center_of_energy",
    "measure_momentum_density",
    "measure_velocity",
    # Phase 4: Composite systems
    "AtomState",
    "MoleculeState",
    "nuclear_chi_well",
    "create_atom",
    "create_molecule",
    # Phase 5: Factory
    "PlacedParticle",
    "create_particle",
]
