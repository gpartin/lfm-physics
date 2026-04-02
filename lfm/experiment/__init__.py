"""
Quantum-experiment components for LFM simulations.
===================================================

Pre-built setups for canonical quantum optics experiments using the
Lattice Field Medium.

Typical double-slit usage::

    import lfm
    from lfm.experiment import Barrier, Slit, DetectorScreen

    sim = lfm.Simulation(lfm.SimulationConfig(
        grid_size=64,
        field_level=lfm.FieldLevel.COMPLEX,
    ))

    # Build barrier with two slits (no which-path detectors)
    barrier = Barrier(
        sim,
        axis=2,
        slits=[Slit(center=24, width=4), Slit(center=40, width=4)],
    )

    # Fire wave packet toward the barrier
    sim.place_soliton((32, 32, 10), amplitude=4.0, velocity=(0, 0, 0.05))

    # Detector screen on the far side
    screen = DetectorScreen(sim, axis=2, position=55)

    # Run — barrier auto-enforced via callback
    snaps = sim.run_with_snapshots(
        steps=3000,
        snapshot_every=100,
        fields=["energy_density", "chi"],
        callback=lambda s, t: (barrier.step_callback(s, t), screen.record()),
    )

    # Plot interference pattern
    import lfm.viz as viz
    viz.plot_interference_pattern(screen.pattern)
"""

from lfm.experiment.barrier import Barrier, Slit
from lfm.experiment.collision import CollisionResult, collision
from lfm.experiment.common import ExperimentConfig, ExperimentResult, midplane_slice
from lfm.experiment.detector import DetectorScreen
from lfm.experiment.dispersion import Dispersion, dispersion
from lfm.experiment.double_slit import DoubleSlit, double_slit
from lfm.experiment.entanglement import (
    SPIN_CONFIGS,
    EntanglementResult,
    EntanglementSuiteResult,
    entanglement,
)
from lfm.experiment.ringdown import (
    DEFAULT_RINGDOWN_K_MODES,
    Next5FalsificationResult,
    QNMProjectionResult,
    next5_falsification_projection_v2,
    qnm_mode_projection_check,
)
from lfm.experiment.source import ContinuousSource

__all__ = [
    "Barrier",
    "Slit",
    "DetectorScreen",
    "ContinuousSource",
    "Dispersion",
    "dispersion",
    "double_slit",
    "DoubleSlit",
    "collision",
    "CollisionResult",
    "DEFAULT_RINGDOWN_K_MODES",
    "entanglement",
    "EntanglementResult",
    "EntanglementSuiteResult",
    "Next5FalsificationResult",
    "SPIN_CONFIGS",
    "QNMProjectionResult",
    "ExperimentConfig",
    "ExperimentResult",
    "midplane_slice",
    "next5_falsification_projection_v2",
    "qnm_mode_projection_check",
]
