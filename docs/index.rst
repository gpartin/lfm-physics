lfm-physics
===========

*Simulate the universe from two equations.*

**lfm-physics** is a Python library for running Lattice Field Medium (LFM)
simulations — a computational physics framework in which gravity,
electromagnetism, dark matter, and structure formation all emerge from a pair of
coupled wave equations discretised on a 3D cubic lattice.

.. code-block:: python

   import lfm

   sim = lfm.Simulation(lfm.SimulationConfig(grid_size=64))
   sim.place_soliton((32, 32, 32), amplitude=6.0)
   sim.equilibrate()          # χ-field adjusts via GOV-02 → gravity well
   sim.run(steps=10_000)
   print(sim.metrics())

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/simulation
   api/config
   api/constants
   api/fields
   api/analysis
   api/units
   api/io
   api/core

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
