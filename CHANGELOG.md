# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.2.1] - 2026-03-20

### Added
- GOV-02 v14: color variance term `κ_c·f_c·Σ|Ψₐ|²` with `KAPPA_C = 1/189`
- GOV-01 v15: cross-color coupling `ε_cc·χ²·(Ψₐ − Ψ̄)` with `EPSILON_CC = 2/17`
- `color_variance()` analysis function for color field diagnostics
- `momentum_density()`, `weak_parity_asymmetry()`, `confinement_proxy()` observables
- `continuity_residual()`, `fluid_fields()` for hydrodynamic analysis
- Constants: `KAPPA_C`, `EPSILON_CC`, `ALPHA_S`, `N_COLORS`
- Config: `kappa_c`, `epsilon_cc`, `n_colors` parameters
- Example 13: weak force (parity asymmetry)
- Example 14: strong force (color screening and confinement proxy)
- Website tutorials 13-14 (emergentphysicslab.com)

### Fixed
- Ruff lint: import sorting, unused imports, line lengths
- `PlanckScale` type annotation in `SimulationConfig.planck_scale`
- Version alignment between `pyproject.toml` and `__init__.py`

## [0.1.3] - 2026-03-19

### Added
- `PlanckScale.at_planck_resolution(grid_size, dt)` — classmethod that sets 1 cell = 1 Planck length exactly
- `PlanckScale.is_planck_resolution` — bool property, True when cells_per_planck == 1.0
- `PlanckScale.cell_size_m` — cell size in metres (Planck length × cells_per_planck)
- `PlanckScale.step_to_planck_ticks(step)` — convert simulation steps to Planck time ticks
- `PlanckScale.planck_ticks_to_step(ticks)` — inverse conversion
- `PlanckScale.step_to_seconds(step)` — convert steps to SI seconds
- Sphinx documentation source tree (`docs/`) with furo theme and autodoc
- Read the Docs configuration (`.readthedocs.yaml`)
- Benchmark suite (`benchmarks/`) with evolver and field-ops benchmarks
- `lfm.io` module with module-level `save_checkpoint` / `load_checkpoint` wrappers

### Removed
- `lfm/formulas/` stub directory (untracked, had no content)

## [0.1.2] - 2026-03-18

### Added
- `Simulation.save_checkpoint(path)` — full state persistence (fields, step counter, config, metric history) to compressed `.npz`
- `Simulation.load_checkpoint(path, backend)` — classmethod to restore a simulation from checkpoint, ready for `run()`
- `Simulation.history` property — list of metric snapshots recorded during `run()`
- `CosmicScale` unit mapper: converts lattice cells/steps to Mpc/Gyr with Hubble calibration
- `PlanckScale` unit mapper: default observable-universe scale and Planck-resolution mode

## [0.1.1] - 2026-03-18

### Fixed
- Corrected author email in package metadata

## [0.1.0] - 2026-03-18

### Added

**M0 — Project scaffolding**
- Package structure with `pyproject.toml` (hatchling)
- CI/CD: GitHub Actions for testing (Python 3.10-3.12) and PyPI publishing
- MIT license, `.gitignore`, `py.typed` marker

**M1 — Constants & Config**
- `lfm.constants` — All LFM fundamental constants derived from chi_0 = 19
- `lfm.config` — `SimulationConfig` dataclass with CFL validation
- `FieldLevel` and `BoundaryType` enums

**M2 — Core numerics**
- `lfm.core.stencils` — 7-point and 19-point isotropic Laplacian operators
- `lfm.core.integrator` — Leapfrog time integration (GOV-01 + GOV-02)
- Supports REAL, COMPLEX, and COLOR field levels

**M3 — Backend system**
- `lfm.core.backends.protocol` — Backend Protocol (runtime-checkable)
- `lfm.core.backends.numpy_backend` — CPU backend (NumPy)
- `lfm.core.backends.cupy_backend` — GPU backend (CuPy + CUDA kernels)
- Auto-detect GPU with graceful CPU fallback

**M4 — Evolver & Fields**
- `lfm.core.evolver` — Double-buffered evolution loop
- `lfm.fields.soliton` — Gaussian soliton placement with phase
- `lfm.fields.equilibrium` — FFT Poisson equilibration (GOV-04)
- `lfm.fields.random` — Random noise seeding
- `lfm.fields.arrangements` — Spatial distribution helpers

**M5 — Analysis**
- `lfm.analysis.energy` — 3-component energy decomposition (kinetic, gradient, potential)
- `lfm.analysis.structure` — Well/void fraction, cluster counting, interior mask
- `lfm.analysis.metrics` — All-in-one snapshot telemetry

**M6 — Formulas**
- `lfm.formulas.predictions` — 41+ physics predictions from chi_0 = 19
- `lfm.formulas.masses` — Complete particle mass table (leptons, quarks, bosons)

**M7 — Simulation facade**
- `lfm.Simulation` — High-level API: place_soliton, equilibrate, run, metrics
- Callback support for monitoring long runs
- Metric history recording
- Checkpoint save/load for resumable simulations

**Examples**
- `examples/cosmic_structure_formation.py` — 256^3 universe simulation
- `examples/soliton_gravity.py` — Single soliton in a chi well
- `examples/em_from_phase.py` — Coulomb force from wave phase interference
- `examples/parametric_resonance.py` — Matter creation from oscillating chi
