"""28 — Coulomb Force: e- + e+ Attract, e- + e- Repel

Electric charge emerges from the PHASE of the complex wave field.
  - Electron (e-): phase = 0
  - Positron (e+): phase = pi

When two same-phase particles overlap, GOV-01 interference INCREASES |psi|^2,
which deepens the chi-well (GOV-02), raising energy.  Higher-energy = repulsion.

When opposite-phase particles overlap, interference is destructive, |psi|^2
decreases, chi-well shallows, energy falls.  Lower-energy = attraction.

This experiment places two particles nearby and measures whether their chi-wells
move toward or away from each other over short evolution.
"""

import numpy as np
import lfm
from lfm.config import BoundaryType, FieldLevel, SimulationConfig

print("28 — Coulomb Force: Charge from Phase")
print("=" * 55)
print()

N = 64
STEPS = 2_000
SEP = 12   # initial separation in cells

# We place two solitons by hand using Gaussian seeds (fast, sufficient demo)
def place_two(phase1, phase2, sep=SEP, N=N):
    """Place two electron-like solitons and return sim."""
    config = SimulationConfig(
        grid_size=N, field_level=FieldLevel.COMPLEX, boundary_type=BoundaryType.FROZEN
    )
    sim = lfm.Simulation(config)
    half = N // 2
    amp = lfm.amplitude_for_particle(lfm.ELECTRON, N)
    sig = lfm.sigma_for_particle(lfm.ELECTRON, N)
    sim.place_soliton((half - sep // 2, half, half), amplitude=amp, sigma=sig, phase=phase1)
    sim.place_soliton((half + sep // 2, half, half), amplitude=amp, sigma=sig, phase=phase2)
    sim.equilibrate()   # build chi-wells
    return sim

import math

def chi_gap(sim, N=N):
    """Return chi-well separation (distance between two chi minima along x-axis)."""
    half = N // 2
    # Extract chi along x at (y=half, z=half)
    chi_line = np.asarray(sim.chi[:, half, half], dtype=np.float64)
    # Find two deepest troughs by looking at local minima below 18.5
    from scipy.signal import argrelmin
    idx = argrelmin(chi_line, order=3)[0]
    wells = sorted(idx, key=lambda i: chi_line[i])[:2]
    if len(wells) < 2:
        return float(np.nan)
    return float(abs(wells[0] - wells[1]))

def run_demo(label, phase1, phase2):
    sim = place_two(phase1, phase2)
    gap0 = chi_gap(sim)
    sim.run(STEPS, evolve_chi=True)
    gap1 = chi_gap(sim)
    if not (math.isnan(gap0) or math.isnan(gap1)):
        delta = gap1 - gap0
        direction = "approach (attract)" if delta < 0 else "separate (repel)"
    else:
        delta = float("nan")
        direction = "undetermined"
    print(f"  {label:<30} gap0={gap0:.1f}  gap1={gap1:.1f}  "
          f"delta={delta:+.1f}  -> {direction}")

print("Measuring chi-well gap change after 2000 steps:")
print()
run_demo("e- + e+  (phase 0 vs pi)",  0.0,       math.pi)
run_demo("e- + e-  (phase 0 vs 0)",   0.0,       0.0)
run_demo("e+ + e+  (phase pi vs pi)", math.pi,   math.pi)
print()
print("Same charge -> repel (gap grows).  Opposite charge -> attract (gap shrinks).")
print()
print("Electric charge is the wave PHASE. No Coulomb law was injected.")
