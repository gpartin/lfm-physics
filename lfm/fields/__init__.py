"""Field initialization: solitons, equilibrium, arrangements."""

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
from lfm.fields.random import seed_noise, uniform_chi
from lfm.fields.soliton import gaussian_soliton, place_solitons, wave_kick

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
]
