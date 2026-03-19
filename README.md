# lfm-physics

[![Tests](https://github.com/gpartin/lfm-physics/actions/workflows/test.yml/badge.svg)](https://github.com/gpartin/lfm-physics/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/lfm-physics?v=2)](https://pypi.org/project/lfm-physics/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/lfm-physics?v=2)](https://pypi.org/project/lfm-physics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Simulate a universe from two equations.**

The Lattice Field Medium (LFM) framework runs two coupled wave equations on a
discrete 3D lattice — and gravity, electromagnetism, dark matter, and cosmic
structure all emerge from the dynamics.  No forces injected.  No constants
assumed.  Just a grid, two update rules, and initial noise.

## Install

```bash
pip install lfm-physics
```

GPU acceleration (NVIDIA):
```bash
pip install lfm-physics[gpu]
```

## Quick Start

```python
import lfm

# 1. Create a 64³ lattice — empty space has χ = 19 everywhere
sim = lfm.Simulation(lfm.SimulationConfig(grid_size=64))

# 2. Drop a soliton (energy blob) onto the grid
sim.place_soliton((32, 32, 32), amplitude=6.0)

# 3. Let the substrate settle into equilibrium
sim.equilibrate()

# 4. Run the two equations for 5 000 steps
sim.run(steps=5000)

# 5. Measure what emerged
m = sim.metrics()
print(f"χ_min = {m['chi_min']:.2f}")   # χ dropped — a gravity well!
print(f"Wells = {m['well_fraction']*100:.1f}%")

# 6. Look at the shape of gravity
profile = lfm.radial_profile(sim.chi, center=(32,32,32), max_radius=20)
# profile['r'] and profile['profile'] — does it fall like 1/r?
```

## Examples — Build a Universe in 12 Steps

Each example builds on the one before, from empty space to a simulated cosmos.
Run them in order:

| # | Example | What you'll see |
|---|---------|-----------------|
| 1 | [01_empty_space.py](examples/01_empty_space.py) | A grid with χ=19 everywhere — nothing happens (that's the point) |
| 2 | [02_first_particle.py](examples/02_first_particle.py) | Add energy → χ drops → a gravity well appears |
| 3 | [03_measuring_gravity.py](examples/03_measuring_gravity.py) | Measure χ(r) and check for 1/r falloff |
| 4 | [04_two_bodies.py](examples/04_two_bodies.py) | Two solitons attract — gravitational interaction emerges |
| 5 | [05_electric_charge.py](examples/05_electric_charge.py) | Phase = charge: same phase repels, opposite attracts |
| 6 | [06_dark_matter.py](examples/06_dark_matter.py) | Remove matter — the χ-well persists (substrate memory) |
| 7 | [07_matter_creation.py](examples/07_matter_creation.py) | Oscillate χ at 2χ₀ — matter appears from nothing |
| 8 | [08_universe.py](examples/08_universe.py) | Random noise on 64³ → wells + voids → cosmic structure |
| 9 | [09_hydrogen_atom.py](examples/09_hydrogen_atom.py) | Proton χ-well traps an electron — energy-level ladder emerges |
| 10 | [10_hydrogen_molecule.py](examples/10_hydrogen_molecule.py) | Two H atoms bond — bonding vs anti-bonding orbitals |
| 11 | [11_oxygen.py](examples/11_oxygen.py) | Heavier nucleus (8 electrons) — deeper well, richer structure |
| 12 | [12_fluid_dynamics.py](examples/12_fluid_dynamics.py) | 40-soliton gas → Euler equation from the stress-energy tensor |

```bash
cd examples
python 01_empty_space.py    # 30 seconds
# ... work through each one ...
python 08_universe.py       # the payoff
python 12_fluid_dynamics.py # fluid velocity from wave mechanics
```

**Interactive tutorials with visualisations:** https://emergentphysicslab.com/tutorials

## What is LFM?

Two coupled wave equations on a cubic lattice:

**GOV-01** — what matter does:
```
∂²Ψ/∂t² = c²∇²Ψ − χ²Ψ
```

**GOV-02** — what the substrate does:
```
∂²χ/∂t² = c²∇²χ − κ(|Ψ|² − E₀²)
```

The substrate field χ starts at 19 everywhere (derived from 3D lattice
geometry: 1 center + 6 face + 12 edge modes = 19).  Energy (|Ψ|²) pushes
χ down, creating wells.  Waves curve toward low χ.  That's gravity.

Complex phase differences in Ψ create interference — constructive (repulsion)
or destructive (attraction).  That's electromagnetism.

When matter leaves a region, χ doesn't snap back instantly — the well
persists.  That's dark matter.

No forces are coded in.  Everything emerges from the two update rules.

## Measurement Tools

The library includes tools to extract physics from your simulation:

```python
# Radial χ profile around a soliton
profile = lfm.radial_profile(sim.chi, center=(32,32,32), max_radius=20)

# Find the N brightest energy peaks
peaks = lfm.find_peaks(sim.energy_density, n=5)

# Track separation between two bodies over time
sep = lfm.measure_separation(sim.energy_density)

# Map lattice ticks to physical units (Gyr, Mpc)
scale = lfm.CosmicScale(box_mpc=100.0, grid_size=64)
print(scale.format_cosmic_time(50_000))  # "1.28 Gyr"
```

## Documentation

- [Interactive Tutorials](https://emergentphysicslab.com/tutorials) — step-by-step guides with live visualisations
- [LFM Physics Papers](https://zenodo.org/communities/lfm-physics)
- [Constants](https://github.com/gpartin/lfm-physics/blob/main/lfm/constants.py) — χ₀ = 19, κ = 1/63, and the rest
- [Contributing](CONTRIBUTING.md)

## License

MIT
