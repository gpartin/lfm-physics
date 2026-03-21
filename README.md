# lfm-physics

[![Tests](https://github.com/gpartin/lfm-physics/actions/workflows/test.yml/badge.svg)](https://github.com/gpartin/lfm-physics/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/lfm-physics)](https://pypi.org/project/lfm-physics/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/pypi/pyversions/lfm-physics)](https://pypi.org/project/lfm-physics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Lattice Field Medium** — a physics simulation library implementing the LFM framework.

Two governing equations. One integer (χ₀ = 19). All of physics.

LFM runs two coupled wave equations on a discrete 3D lattice.
Gravity, electromagnetism, the strong and weak forces, dark matter, and cosmic
structure all emerge from the dynamics — no forces injected, no constants
assumed.  Just a grid, two update rules, and initial noise.

## Install

```bash
pip install lfm-physics
```

GPU acceleration (NVIDIA, optional):
```bash
pip install "lfm-physics[gpu]"
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

## The Equations

Everything in LFM follows from two coupled wave equations evaluated on a
3D cubic lattice with a 19-point stencil (6 face + 12 edge neighbours):

**GOV-01** — matter wave equation (Klein-Gordon on the lattice):

```
Ψⁿ⁺¹ = 2Ψⁿ − Ψⁿ⁻¹ + Δt²[ c²∇²Ψⁿ − (χⁿ)²Ψⁿ ]
```

**GOV-02** — substrate field equation (complete, v14):

```
χⁿ⁺¹ = 2χⁿ − χⁿ⁻¹ + Δt²[ c²∇²χⁿ
        − κ(Σₐ|Ψₐⁿ|² + ε_W·jⁿ − E₀²)          ← gravity + weak
        − 4λ_H·χⁿ((χⁿ)² − χ₀²)                  ← Higgs self-interaction
        − κ_c·f_c·Σₐ|Ψₐⁿ|² ]                     ← color screening
```

| Term | Force | What it does |
|------|-------|--------------|
| `κ·Σ\|Ψ\|²` | **Gravity** | Energy density creates χ wells — waves curve toward low χ |
| `ε_W·j` | **Weak** | Momentum current j breaks parity symmetry in χ |
| `4λ_H·χ(χ²−χ₀²)` | **Higgs** | Mexican-hat potential makes χ₀ = 19 a dynamical attractor |
| `κ_c·f_c·Σ\|Ψ\|²` | **Strong** | Color variance f_c gives extra χ deepening for non-singlet states |
| Phase interference | **EM** | Same phase → constructive → repel; opposite → destructive → attract |

### Field Levels

| Level | Field | Components | Forces | Use for |
|-------|-------|------------|--------|---------|
| `REAL` | E ∈ ℝ | 1 real | Gravity | Cosmology, dark matter |
| `COMPLEX` | Ψ ∈ ℂ | 1 complex | Gravity + EM | Atoms, charged particles |
| `COLOR` | Ψₐ ∈ ℂ³ | 3 complex | All four | Full multi-force simulations |

```python
# Gravity-only cosmology (fastest)
cfg = lfm.SimulationConfig(grid_size=128, field_level=lfm.FieldLevel.REAL)

# Electromagnetism + gravity
cfg = lfm.SimulationConfig(grid_size=64, field_level=lfm.FieldLevel.COMPLEX)

# All four forces
cfg = lfm.SimulationConfig(grid_size=64, field_level=lfm.FieldLevel.COLOR)
```

### Constants — All Derived from χ₀ = 19

| Constant | Symbol | Value | Origin |
|----------|--------|-------|--------|
| `CHI0` | χ₀ | 19 | 3D lattice modes: 1 + 6 + 12 = 19 |
| `KAPPA` | κ | 1/63 ≈ 0.0159 | Unit coupling on 4³ − 1 = 63 modes |
| `LAMBDA_H` | λ_H | 4/31 ≈ 0.129 | z₂ lattice geometry |
| `EPSILON_W` | ε_W | 2/(χ₀+1) = 0.1 | Weak mixing angle factorisation |
| `KAPPA_C` | κ_c | κ/3 = 1/189 | Color variance coupling |
| `ALPHA_S` | α_s | 2/17 ≈ 0.118 | Strong coupling at M_Z |
| `EPSILON_CC` | ε_cc | 2/17 | Cross-color coupling (GOV-01 v15) |
| `ALPHA_EM` | α | 11/(480π) ≈ 1/137.1 | Fine-structure constant |
| `OMEGA_LAMBDA` | Ω_Λ | 13/19 ≈ 0.684 | Dark energy fraction |
| `OMEGA_MATTER` | Ω_m | 6/19 ≈ 0.316 | Matter fraction |

## Examples — Build a Universe in 14 Steps

Each example builds on the one before, from empty space to a simulated cosmos:

| # | Example | What you'll see |
|---|---------|-----------------|
| 1 | [01_empty_space.py](examples/01_empty_space.py) | A grid with χ = 19 everywhere — nothing happens |
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
| 13 | [13_weak_force.py](examples/13_weak_force.py) | Turn ε_W on/off and measure parity asymmetry from χ + j |
| 14 | [14_strong_force.py](examples/14_strong_force.py) | Color fields — measure confinement proxy via χ line integrals |

```bash
cd examples
python 01_empty_space.py        # start here
python 08_universe.py           # the payoff — cosmic structure from noise
python 14_strong_force.py       # all four forces active
```

**Interactive tutorials with visualisations:**
[emergentphysicslab.com/tutorials](https://emergentphysicslab.com/tutorials)

## Measurement & Analysis

```python
# Radial χ profile around a soliton
profile = lfm.radial_profile(sim.chi, center=(32,32,32), max_radius=20)

# Find the N brightest energy peaks
peaks = lfm.find_peaks(sim.energy_density, n=5)

# Track separation between two bodies over time
sep = lfm.measure_separation(sim.energy_density)

# Energy conservation drift
drift = lfm.energy_conservation_drift(sim)

# Fluid velocity from the stress-energy tensor
fields = lfm.fluid_fields(psi_real, psi_imag, chi, dt, c=1.0)
# fields['velocity_x'], fields['pressure'], fields['energy_density']

# Momentum density (Noether current — sources weak force)
j = lfm.momentum_density(psi_real, psi_imag, dx=1.0)

# Color variance (strong force diagnostic)
fc = lfm.color_variance(psi_real_color, psi_imag_color)

# Confinement proxy — χ line integral between peaks
proxy = lfm.confinement_proxy(sim.chi, pos_a, pos_b)

# Fourier power spectrum P(k) of any 3D field
spec = lfm.power_spectrum(sim.chi, bins=50)
# spec['k'], spec['power']

# Track energy peaks across a run
trajectories = lfm.track_peaks(sim, steps=5000, interval=200, n_peaks=3)
```

## Parameter Sweeps

```python
# Sweep amplitude from 2 to 10 and record chi_min at each value
config = lfm.SimulationConfig(grid_size=32)
results = lfm.sweep(config, param="amplitude", values=[2, 4, 6, 8, 10],
                    steps=3000, metric_names=["chi_min", "well_fraction"])
for r in results:
    print(f"amp={r['amplitude']:.0f}  chi_min={r['chi_min']:.2f}")
```

## Visualisation *(New in 0.3.0)*

Install with: `pip install "lfm-physics[viz]"`

```python
from lfm.viz import (
    plot_slice,          # 2D slice through a 3D field
    plot_three_slices,   # XY + XZ + YZ panels
    plot_chi_histogram,  # distribution of χ values
    plot_evolution,      # time-series of metrics
    plot_energy_components,  # stacked kinetic / gradient / potential
    plot_radial_profile, # χ(r) with 1/r reference overlay
    plot_isosurface,     # 3D voxel rendering
    plot_power_spectrum, # P(k) from Fourier analysis
    plot_trajectories,   # peak motion in x-y / x-z / y-z
    plot_sweep,          # sweep results line plot
)

# Example: slice through the chi field at z = 32
fig, ax = plot_slice(sim.chi, axis=2, index=32, title="χ mid-plane")
fig.savefig("chi_slice.png")

# Three-panel overview
fig = plot_three_slices(sim.chi, title="χ field")

# Time evolution dashboard
fig = plot_evolution(sim.history)

# Radial profile with 1/r fit
fig, ax = plot_radial_profile(sim.chi, center=(32,32,32))
```

## Checkpoints & Units

```python
# Save / resume a simulation
sim.save_checkpoint("my_run.npz")
sim2 = lfm.Simulation.load_checkpoint("my_run.npz")

# Map lattice ticks to physical units
scale = lfm.CosmicScale(box_mpc=100.0, grid_size=64)
print(scale.format_cosmic_time(50_000))   # "1.28 Gyr"

# Planck-resolution mode
ps = lfm.PlanckScale.at_planck_resolution(grid_size=256)
print(ps.cells_per_planck)                # 1.0 exactly
```

## Low-Level API — Evolver

For maximum performance or custom evolution loops:

```python
from lfm import SimulationConfig, FieldLevel, Evolver

cfg = SimulationConfig(grid_size=128, field_level=FieldLevel.COLOR)
evo = Evolver(cfg, backend="auto")  # auto-selects GPU if available

# Inject initial conditions
evo.set_psi_real(my_array)
evo.set_chi(my_chi)

# Evolve 100 000 steps
evo.evolve(100_000)

# Extract fields as NumPy arrays
chi = evo.get_chi()
psi = evo.get_psi_real()
```

## GPU Support

The library automatically uses your NVIDIA GPU when CuPy is installed:

```python
print(lfm.gpu_available())  # True if CuPy + CUDA detected

# Force CPU even if GPU is available
sim = lfm.Simulation(cfg, backend="cpu")
```

Typical speedup: **50-200×** for N ≥ 64 grids on modern NVIDIA GPUs.

## Documentation

- [Interactive Tutorials](https://emergentphysicslab.com/tutorials) — step-by-step guides with live visualisations
- [LFM Physics Papers](https://zenodo.org/communities/lfm-physics) — 84+ published papers
- [Constants Reference](https://github.com/gpartin/lfm-physics/blob/main/lfm/constants.py) — χ₀ = 19 and everything derived from it
- [Changelog](CHANGELOG.md) — version history
- [Contributing](CONTRIBUTING.md)

## License

MIT
