"""Low-level computation layer for the LFM leapfrog integrator.

This sub-package owns the performance-critical inner loop that
propagates Ψ and χ one timestep forward via the discrete GOV-01
+ GOV-02 leapfrog update.  It provides:

- **Evolver** — the main integration object that wraps backend
  (CPU / GPU) double-buffer leapfrog, the 19-point isotropic
  stencil, CFL validation, and optional SA-confinement fields.
- **Backends** — NumPy (CPU) and CuPy (GPU) wrappers behind a
  common interface so the rest of the library is backend-agnostic.
- **Boundary helpers** — frozen-boundary masks, interior masks,
  and ABSORBING PML layer support.

Most user code interacts with :class:`~lfm.simulation.Simulation`
instead of calling this layer directly.  Use :class:`Evolver` only
when you need fine-grained control (e.g. grid-checkpoint replay,
custom multi-grid schemes).
"""

from lfm.core.evolver import Evolver

__all__ = ["Evolver"]
