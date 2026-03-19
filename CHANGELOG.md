# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
