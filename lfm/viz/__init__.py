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

from lfm.viz.evolution import plot_energy_components, plot_evolution
from lfm.viz.fields import plot_isosurface
from lfm.viz.radial import plot_radial_profile
from lfm.viz.slices import plot_chi_histogram, plot_slice, plot_three_slices
from lfm.viz.spectrum import plot_power_spectrum
from lfm.viz.sweep import plot_sweep
from lfm.viz.tracker import plot_trajectories

__all__ = [
    "plot_slice",
    "plot_three_slices",
    "plot_chi_histogram",
    "plot_evolution",
    "plot_energy_components",
    "plot_radial_profile",
    "plot_isosurface",
    "plot_power_spectrum",
    "plot_trajectories",
    "plot_sweep",
]
