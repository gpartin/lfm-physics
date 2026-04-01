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

GPU acceleration (choose the wheel matching your environment):
```bash
pip install "lfm-physics[gpu-cuda12x]"  # CUDA 12.x (RTX 30/40-series)
pip install "lfm-physics[gpu-cuda11x]"  # CUDA 11.x (RTX 20-series, Ampere)
pip install "lfm-physics[gpu-rocm]"     # AMD ROCm 5.x
pip install "lfm-physics[gpu]"          # alias for gpu-cuda12x
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

**GOV-02** — substrate field equation (complete, v17):

```
χⁿ⁺¹ = 2χⁿ − χⁿ⁻¹ + Δt²[ c²∇²χⁿ
        − κ(Σₐ|Ψₐⁿ|² + ε_W·jⁿ − E₀²)          ← gravity + weak
        − 4λ_H·χⁿ((χⁿ)² − χ₀²)                  ← Higgs self-interaction
        − κ_c·f_c·Σₐ|Ψₐⁿ|²                       ← color screening (v14)
        − ε_cc·χ²·(Ψₐ − Ψ̄)  (in GOV-01)         ← cross-color coupling (v15)
        − κ_string·CCV·Σₐ|Ψₐⁿ|²                  ← color current variance (v16)
        − κ_tube·SCV·Σₐ|Ψₐⁿ|² ]                  ← flux tube confinement (v17)
```

**S_a auxiliary** — Helmholtz-smoothed color density (v17):

```
(γ + D·k²) S̃ₐ(k) = γ · FT[|Ψₐ|²](k)

SCV = [Σₐ S̃ₐ² / (Σₐ S̃ₐ)²] − 1/3     (smoothed color variance)
```

All five S_a parameters are derived from χ₀ = 19:
γ = ε_W = 0.1, L = β₀ = 7, D = γL² = 4.9, κ_tube = 30κ ≈ 0.476.

| Term | Force | What it does |
|------|-------|--------------|
| `κ·Σ\|Ψ\|²` | **Gravity** | Energy density creates χ wells — waves curve toward low χ |
| `ε_W·j` | **Weak** | Momentum current j breaks parity symmetry in χ |
| `4λ_H·χ(χ²−χ₀²)` | **Higgs** | Mexican-hat potential makes χ₀ = 19 a dynamical attractor |
| `κ_c·f_c·Σ\|Ψ\|²` | **Strong (screening)** | Color variance f_c gives extra χ deepening for non-singlet states |
| `ε_cc·χ²·(Ψₐ−Ψ̄)` | **Strong (mixing)** | Cross-color coupling: equalises colors, favours hadrons over quarks |
| `κ_tube·SCV` | **Strong (confinement)** | Helmholtz-smoothed color variance → linear potential between quarks |
| Phase interference | **EM** | Same phase → constructive → repel; opposite → destructive → attract |

### Field Levels

| Level | Field | Components | Forces | Use for |
|-------|-------|------------|--------|---------|
| `REAL` | E ∈ ℝ | 1 real | Gravity | Cosmology, dark matter |
| `COMPLEX` | Ψ ∈ ℂ | 1 complex | Gravity + EM | Atoms, charged particles |
| `COLOR` | Ψₐ ∈ ℂ³ | 3 complex | All four | Full multi-force simulations |

```python
# Config presets (recommended) — one call, all parameters set correctly:
cfg = lfm.gravity_only(grid_size=128)   # Real E, κ only
cfg = lfm.gravity_em(grid_size=64)      # Complex Ψ, κ + phase
cfg = lfm.full_physics(grid_size=64)    # Color Ψₐ, all v17 terms

# Or manual field-level selection:
cfg = lfm.SimulationConfig(grid_size=128, field_level=lfm.FieldLevel.REAL)
cfg = lfm.SimulationConfig(grid_size=64, field_level=lfm.FieldLevel.COMPLEX)
cfg = lfm.SimulationConfig(grid_size=64, field_level=lfm.FieldLevel.COLOR)
```

### Constants — All Derived from χ₀ = 19

| Constant | Symbol | Value | Origin |
|----------|--------|-------|--------|
| `CHI0` | χ₀ | 19 | 3D lattice modes: 1 + 6 + 12 = 19 |
| `KAPPA` | κ | 1/63 ≈ 0.0159 | Unit coupling on 4³ − 1 = 63 modes |
| `LAMBDA_H` | λ_H | 4/31 ≈ 0.129 | z₂ lattice geometry |
| `EPSILON_W` | ε_W | 2/(χ₀+1) = 0.1 | Weak mixing angle factorisation |
| `KAPPA_C` | κ_c | κ/3 = 1/189 | Color variance coupling (v14) |
| `ALPHA_S` | α_s | 2/17 ≈ 0.118 | Strong coupling at M_Z |
| `EPSILON_CC` | ε_cc | 2/17 | Cross-color coupling (v15) |
| `KAPPA_STRING` | κ_string | κ_c = 1/189 | CCV string coupling (v16) |
| `KAPPA_TUBE` | κ_tube | 30κ ≈ 0.476 | SCV flux tube coupling (v17) |
| `SA_GAMMA` | γ | ε_W = 0.1 | Helmholtz decay rate |
| `SA_L` | L | β₀ = 7 | Flux tube coherence length |
| `SA_D` | D | γL² = 4.9 | Helmholtz diffusion coefficient |
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

**Advanced & showcase examples (15–37):** cover double-slit interference, Kepler orbits, gravitational waves, Stern-Gerlach spin deflection, 720° spinor periodicity, Bell inequalities, and more. See [examples/](examples/) for the full list.

```bash
cd examples
python 01_empty_space.py        # start here
python 08_universe.py           # the payoff — cosmic structure from noise
python 14_strong_force.py       # all four forces active
python 37_spin_entanglement_showcase.py --small   # spin correlations + 3-D movie
```

**Interactive tutorials with visualisations:**
[emergentphysicslab.com/tutorials](https://emergentphysicslab.com/tutorials)

## Particle Physics

The `lfm.particles` sub-package provides a catalog of 69 particles
(all Standard Model fermions, gauge bosons, and common hadrons),
an SCF eigenmode solver, and a one-call factory that drops a
physically correct soliton into a ready-to-run simulation.

### Particle Catalog

```python
from lfm.particles.catalog import PARTICLES, get_particle

# List all 69 particles with quantum numbers
for name, p in PARTICLES.items():
    print(f"{name:16s}  mass_ratio={p.mass_ratio:10.1f}  l={p.l}  charge={p.charge}")
# electron, positron, muon, antimuon, tau, antitau,
# up, down, strange, charm, bottom, top (+ antiparticles),
# proton, neutron, pion±/⁰, kaon, D, B, Λ, Σ, Ξ, Ω, ...
# W±, Z, Higgs, photon, gluon, neutrinos
```

See [examples/24_particle_catalog.py](examples/24_particle_catalog.py) for the full table.

### Electron at Rest

```python
import lfm

# Eigenmode solver finds the stable self-consistent soliton (~10 000 steps)
placed = lfm.create_particle("electron")
print(f"chi_min = {placed.sim.metrics()['chi_min']:.3f}")  # < 19 — well formed

# Run 1 000 coupled GOV-01 + GOV-02 steps
placed.sim.run(1000)
```

The solver runs the Self-Consistent Field (SCF) algorithm:
1. Place a Gaussian seed; Poisson-equilibrate χ
2. Evolve Ψ only (χ frozen) — radiation escapes, bound mode survives
3. Re-equilibrate χ; iterate until energy converges (<5% change)
4. Verify stability with 500 coupled GOV-01+GOV-02 steps

See [examples/25_electron_at_rest.py](examples/25_electron_at_rest.py).

### Moving Particles

```python
# Boost to 4% of lightspeed — wraps the eigenmode in a Gaussian wave packet
e = lfm.create_particle("electron", velocity=(0.04, 0.0, 0.0))
e.sim.run(5000)

# Compare electron vs muon at same velocity
mu = lfm.create_particle("muon", velocity=(0.04, 0.0, 0.0))
```

See [examples/26_electron_traverse.py](examples/26_electron_traverse.py) and
[examples/27_electron_vs_muon.py](examples/27_electron_vs_muon.py).

### Coulomb Force (Phase = Charge)

```python
# theta=0 → negative charge (electron)
# theta=pi → positive charge (positron)
# Same phase repels; opposite phase attracts — from GOV-01 interference alone
```

See [examples/28_coulomb_force.py](examples/28_coulomb_force.py).

### Hydrogen Atom and Molecule

```python
import lfm

# H atom: proton χ-well traps psi wave — returns bound=True, fraction_near_nucleus
atom = lfm.create_atom("H", N=64, steps=10000)
print(f"bound={atom.bound}  fraction={atom.fraction_near_nucleus:.3f}")

# H2 molecule: two proton wells at bond_length separation
mol = lfm.create_molecule("H2", N=64, bond_length=16.0)
print(f"bond_stable={mol.bond_stable}")
```

See [examples/29_hydrogen_atom.py](examples/29_hydrogen_atom.py) and
[examples/30_hydrogen_molecule.py](examples/30_hydrogen_molecule.py).

### Fast Gaussian Seed (no eigenmode solver)

```python
# Skip the SCF solver for quick prototyping — Gaussian blob, instant
placed = lfm.create_particle("muon", use_eigenmode=False)
placed.sim.run(500)
```

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

The library automatically uses your GPU when CuPy is installed and a
compatible accelerator is detected:

```python
print(lfm.gpu_available())  # True if CuPy + CUDA/ROCm detected

# Force CPU even if GPU is available
sim = lfm.Simulation(cfg, backend="cpu")
```

Install the correct CuPy wheel for your hardware:

| Hardware | Install command |
|---|---|
| NVIDIA CUDA 12.x (RTX 30/40-series) | `pip install "lfm-physics[gpu-cuda12x]"` |
| NVIDIA CUDA 11.x (RTX 20-series) | `pip install "lfm-physics[gpu-cuda11x]"` |
| AMD ROCm 5.0 | `pip install "lfm-physics[gpu-rocm]"` |
| Convenience alias (CUDA 12.x) | `pip install "lfm-physics[gpu]"` |

If you are unsure which CUDA version you have, run `nvcc --version` or
`nvidia-smi`.  On CPU-only machines, no GPU package is needed; NumPy is
used automatically.

Typical speedup: **50–200×** for N ≥ 64 grids on modern NVIDIA GPUs.

## Project Structure

```
lfm/                Library source code (import as `import lfm`)
  core/             Leapfrog evolver, backends (NumPy + CuPy/CUDA), config
  analysis/         Post-processing: spectra, spinors, chi statistics
  experiment/       High-level experiment runners (double-slit, collision, …)
  particles/        Soliton solvers, tracking, collision helpers
  viz/              Plotting and animation utilities (matplotlib + ffmpeg)

examples/           37 numbered tutorial scripts (01 → 37)
  outputs/          Auto-created results (PNG, MP4) – not committed

experiments/        Quantitative validation experiments (paper evidence)

tests/              pytest suite (`pytest tests/`)

docs/               Sphinx documentation source

benchmarks/         Performance benchmarking scripts

research/           Exploratory scripts tied to paper drafts

scripts/            Development scratch scripts (prefixed with `_`)
                    Not part of the public API; may be incomplete or outdated

tools/              Developer utilities (linting helpers, release scripts)
```

## Papers & Citations

All 84+ LFM papers are open access at
**[zenodo.org/communities/lfm-physics](https://zenodo.org/communities/lfm-physics)**.

### Citing this library

```bibtex
@software{partin2026lfm,
  author    = {Partin, Greg},
  title     = {{lfm-physics}: {Lattice Field Medium} Physics Simulation Library},
  version   = {1.3.0},
  date      = {2026-03-31},
  license   = {MIT},
  url       = {https://github.com/gpartin/lfm-physics},
  doi       = {10.5281/zenodo.lfm-physics},
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included in the repository
(GitHub and Zenodo both pick it up automatically).

### Key results covered by this library

| Paper | What it showed |
|-------|----------------|
| [LFM-045](https://zenodo.org/communities/lfm-physics) | Complete framework: GOV-01 + GOV-02 derive gravity, EM, strong and weak forces from two equations |
| [LFM-048](https://zenodo.org/communities/lfm-physics) | Spin-½ and the Dirac equation emerge from Lorentz symmetry of GOV-01 solutions |
| [LFM-050](https://zenodo.org/communities/lfm-physics) | Kepler T² ∝ r³ reproduced to 0.04 % error from GOV-01 + GOV-02 alone |
| [LFM-051/052](https://zenodo.org/communities/lfm-physics) | Hydrogen atom and molecular binding emerge as standing-wave eigenmodes |
| [LFM-065](https://zenodo.org/communities/lfm-physics) | Electric charge = wave phase θ; Coulomb force from phase interference |
| [LFM-070](https://zenodo.org/communities/lfm-physics) | Bell-inequality violation (S ≈ 2√2) from χ-geometry correlations |
| [LFM-071](https://zenodo.org/communities/lfm-physics) | Dark energy Ω_Λ = 13/19 = 0.684 derived from lattice mode counting (0.12 % error) |
| [LFM-075](https://zenodo.org/communities/lfm-physics) | Higgs self-coupling λ_H = 4/31 derived from second-shell lattice geometry |
| [LFM-085](https://zenodo.org/communities/lfm-physics) | Quantum entanglement reproduced without hidden variables via non-separable χ correlations |

### Experimental validation in this library

The `lfm` package ships the code that generated, or is directly descended from,
the experiments in those papers:

```python
import lfm, lfm.config_presets as presets

# reproduce Paper 050 – Kepler orbits
cfg = presets.gravity_only(grid_size=64)
sim = lfm.Simulation(cfg)
sim.place_soliton((32, 32, 16), amplitude=6.0)
sim.equilibrate()
sim.run(steps=10_000)
print(sim.metrics())          # kepler_error < 0.04 %

# reproduce Paper 065 – Coulomb repulsion from phase
cfg = presets.gravity_em(grid_size=64)
sim = lfm.Simulation(cfg)
e1 = sim.place_particle("electron", position=(20, 32, 32))
e2 = sim.place_particle("electron", position=(44, 32, 32))
sim.run(steps=2_000)
print(e1.force, e2.force)     # opposite signs ✓
```

## Documentation

- [Interactive Tutorials](https://emergentphysicslab.com/tutorials) — step-by-step guides with live visualisations
- [LFM Physics Papers](https://zenodo.org/communities/lfm-physics) — 84+ published papers
- [Constants Reference](https://github.com/gpartin/lfm-physics/blob/main/lfm/constants.py) — χ₀ = 19 and everything derived from it
- [Changelog](CHANGELOG.md) — version history
- [Contributing](CONTRIBUTING.md)

## License

MIT
