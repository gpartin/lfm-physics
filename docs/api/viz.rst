Visualization (``lfm.viz``)
===========================

Optional plotting and animation utilities.  Requires **matplotlib**::

    pip install "lfm-physics[viz]"

Module overview
---------------

.. automodule:: lfm.viz
   :members:
   :undoc-members:
   :show-inheritance:

Static field plots
------------------

.. autofunction:: lfm.viz.plot_slice
.. autofunction:: lfm.viz.plot_three_slices
.. autofunction:: lfm.viz.plot_chi_histogram
.. autofunction:: lfm.viz.plot_radial_profile
.. autofunction:: lfm.viz.plot_projection
.. autofunction:: lfm.viz.project_field
.. autofunction:: lfm.viz.plot_isosurface

Animations
----------

.. autofunction:: lfm.viz.animate_slice
.. autofunction:: lfm.viz.animate_three_slices
.. autofunction:: lfm.viz.animate_double_slit
.. autofunction:: lfm.viz.animate_double_slit_3d
.. autofunction:: lfm.viz.animate_3d_slices
.. autofunction:: lfm.viz.render_3d_volume
.. autofunction:: lfm.viz.volume_render_available

Evolution and energy
--------------------

.. autofunction:: lfm.viz.plot_evolution
.. autofunction:: lfm.viz.plot_energy_components
.. autofunction:: lfm.viz.spacetime_diagram
.. autofunction:: lfm.viz.plot_power_spectrum

Sweeps and trajectories
-----------------------

.. autofunction:: lfm.viz.plot_sweep
.. autofunction:: lfm.viz.plot_trajectories
.. autofunction:: lfm.viz.galaxy_summary_plot
