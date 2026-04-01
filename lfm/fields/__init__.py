"""Field initialization utilities for LFM simulations.

This module provides functions for placing wave-field sources on the
lattice (Gaussian solitons, multi-body arrangements, colour-field
configurations) and for computing the quasi-static χ equilibrium
(GOV-04 Poisson solve via FFT) that provides consistent initial
conditions for structure-formation experiments.

Typical usage
-------------
Most of these helpers are called indirectly through
:class:`~lfm.simulation.Simulation`::meth:`place_soliton` and
:meth:`~lfm.simulation.Simulation.equilibrate`.  Direct access is
useful when constructing batched initial conditions outside the
:class:`Simulation` API.
"""

from lfm.fields.arrangements import (
    disk_positions,
    disk_velocities,
    grid_positions,
    initialize_disk,
    sparse_positions,
    tetrahedral_positions,
)
from lfm.fields.boosted import boosted_soliton
from lfm.fields.equilibrium import (
    equilibrate_chi,
    equilibrate_from_fields,
    poisson_solve_fft,
)
from lfm.fields.light import spherical_phase_source
from lfm.fields.random import seed_noise, uniform_chi
from lfm.fields.soliton import gaussian_soliton, place_solitons, wave_kick
from lfm.fields.spinor import (
    apply_rotation_x,
    apply_rotation_z,
    gaussian_spinor,
    vortex_spinor,
)

__all__ = [
    "gaussian_soliton",
    "place_solitons",
    "wave_kick",
    "boosted_soliton",
    "poisson_solve_fft",
    "equilibrate_chi",
    "equilibrate_from_fields",
    "seed_noise",
    "uniform_chi",
    "tetrahedral_positions",
    "sparse_positions",
    "grid_positions",
    "disk_positions",
    "disk_velocities",
    "initialize_disk",
    # spinor
    "gaussian_spinor",
    "vortex_spinor",
    "apply_rotation_x",
    "apply_rotation_z",
    # light
    "spherical_phase_source",
]
