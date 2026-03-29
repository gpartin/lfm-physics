"""
Visualization — optional plotting for LFM simulations.
=======================================================

Requires ``matplotlib``.  Install with::

    pip install "lfm-physics[viz]"

Quick usage::

    import lfm
    from lfm.viz import plot_slice, plot_evolution

    sim = lfm.Simulation(lfm.SimulationConfig(grid_size=64))
    sim.place_soliton((32, 32, 32), amplitude=6.0)
    sim.equilibrate()
    sim.run(steps=5000)

    plot_slice(sim.chi)
    plot_evolution(sim.history)
"""

from lfm.viz.animation import animate_slice, animate_three_slices
from lfm.viz.collision import animate_collision_3d
from lfm.viz.evolution import plot_energy_components, plot_evolution
from lfm.viz.fields import plot_isosurface
from lfm.viz.galaxy import galaxy_summary_plot
from lfm.viz.projection import plot_projection, project_field
from lfm.viz.quantum import (
    animate_3d_slices,
    animate_double_slit,
    animate_double_slit_3d,
    plot_interference_pattern,
    render_3d_volume,
    volume_render_available,
)
from lfm.viz.radial import plot_radial_profile
from lfm.viz.slices import plot_chi_histogram, plot_slice, plot_three_slices
from lfm.viz.spacetime import spacetime_diagram
from lfm.viz.spectrum import plot_power_spectrum
from lfm.viz.sweep import plot_sweep
from lfm.viz.tracker import plot_trajectories

__all__ = [
    # static slices
    "plot_slice",
    "plot_three_slices",
    "plot_chi_histogram",
    # animations
    "animate_slice",
    "animate_three_slices",
    "animate_collision_3d",
    # projections
    "project_field",
    "plot_projection",
    # space-time
    "spacetime_diagram",
    # evolution / energy
    "plot_evolution",
    "plot_energy_components",
    # quantum / double-slit
    "plot_interference_pattern",
    "animate_double_slit",
    "animate_double_slit_3d",
    "animate_3d_slices",
    "render_3d_volume",
    "volume_render_available",
    # other
    "plot_radial_profile",
    "plot_isosurface",
    "plot_power_spectrum",
    "plot_trajectories",
    "plot_sweep",
    "galaxy_summary_plot",
]
