# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.2.1] - 2026-03-27

### Fixed
- **Which-path detector: χ-blocking replaced with Ψ-amplitude damping** (`lfm/experiment/barrier.py`):
  the old implementation raised slit χ proportional to detector strength, which physically
  blocked wave propagation rather than measuring it.  The corrected mechanism leaves slit χ
  at χ₀ (fully open) and applies per-step Ψ amplitude scaling `pr *= (1-γ)` at detector
  slit cells, where γ is calibrated so `(1-γ)^transit_steps = (1-strength)`.  V3/V4
  re-runs confirm fringes weaken continuously with strength rather than vanishing abruptly.
- **`_det_scale_masks` reshape bug** (`lfm/experiment/barrier.py`): `_td(scale_cpu)` returned
  a flat `(N³,)` CuPy array; added `.reshape(N, N, N)` so the mask broadcasts correctly
  against the `(N, N, N)` Ψ arrays without a `ValueError`.
- **`Barrier.apply()` OOM crash at N≥256** (`lfm/experiment/barrier.py`): the old
  element-wise blend implementation (`active[:] = active * (1-bm) + h*bm`) created
  3-4 temporary N³ float32 arrays per line on every simulation step.  At N=256
  COMPLEX (6+ × 64 MB resident arrays) this exhausted the CuPy memory pool, causing
  a fatal `cudaErrorMemoryAllocation` after ~50k steps.
- **`Barrier.apply()` 4× throughput regression**: the same blend operations moved
  ~2.5 GB of GPU memory per step, reducing throughput from ~264 steps/s to ~60 steps/s.

### Changed
- `Barrier.__init__` now pre-builds lightweight enforcement structures:
  - `_barrier_slab_sl`: axis-specific slice tuple for the barrier slab.
  - `_slit_restore_sls`: list of `(slice, chi_value)` pairs for each slit/detector.
  - `_psi_barrier_mask`: pre-built float32 device array (0 at barrier, 1 elsewhere).
- `Barrier.apply()` replaced with a **fill+slab** strategy:
  1. `buf.fill(chi0)` — write-only fill (no source array read).
  2. `buf[slab_sl] = height` — overwrite barrier slab.
  3. Per-slit slice restore for transparent / detector openings.
  4. `pr *= psi_barrier_mask` — in-place Ψ zeroing, no temporaries.
- **Net result**: ~256 MB memory traffic per step (vs ~2.5 GB before), 144 steps/s
  at N=256 COMPLEX (was 60 steps/s), no more OOM.


### Added
- **Double-slit experiment module** (`lfm/experiment/double_slit.py`): `double_slit()` —
  self-contained 3D double-slit setup returning geometry, barrier, and source objects;
  supports near-field / far-field / wave-packet modes and full which-path detector variants.
- **`23_double_slit.py` multi-variant CLI**: `--variant` now accepts multiple values
  (`--variant 3 4 5 6 7 8`) so subsets of the 8 variants can be run in one invocation.
- **Remote backend** (`lfm/core/backends/remote_backend.py`, `job_schema.py`):
  `RemoteBackend` dispatches `SimulationJob` payloads to the WaveGuard cloud GPU endpoint
  (`POST /v1/simulate_job`); credentials via `LFM_SIMULATE_API_KEY` env var or
  `lfm.configure_remote()`.  All live tests skip in CI when the key is absent.
- **`CITATION.cff`**: machine-readable citation metadata for Zenodo/GitHub.
- **Viz API docs** (`docs/api/viz.rst`): Sphinx autodoc page for `lfm.viz`.
- **Color confinement scaffolding** (`lfm.analysis.confinement`): `confinement_proxy()`, `color_current_variance()` — characterise SU(3) colour physics
- **Collider diagnostics** (`lfm.analysis.collider`): `collider_event_display()`, `compute_impact_parameter()` — head-on soliton scattering analysis
- **Boosted soliton interface** (`lfm.fields.boosted`): `boosted_soliton()` with phase-gradient encoding for scatter experiments
- **2D parameter sweeps** (`lfm.sweep.sweep_2d`) — all (v1, v2) combinations with nested metrics collection
- Library Quality Plan (`LIBRARY_QUALITY_PLAN.md`) and comprehensive audit of all 10 gaps

### Changed
- Classifier upgraded to `Development Status :: 5 - Production/Stable`
- Docs version aligned to `1.2.0` in `conf.py`
- Documentation URL added to PyPI metadata

## [1.1.0] - 2026-03-23

### Added
- **Metric analysis** (`lfm.analysis.metric`): `effective_metric_00`, `metric_perturbation`, `time_dilation_factor`, `gravitational_potential`, `schwarzschild_chi`
- **Phase/charge analysis** (`lfm.analysis.phase`): `phase_field`, `charge_density`, `phase_coherence`, `coulomb_interaction_energy`
- **Angular momentum** (`lfm.analysis.angular_momentum`): `angular_momentum_density`, `total_angular_momentum`, `precession_rate`
- **GitHub governance**: issue/PR templates, SECURITY.md, CODE_OF_CONDUCT.md
- **Pre-commit hooks**: `.pre-commit-config.yaml` with ruff linting and formatting
- **Docs build**: `docs/Makefile` and `docs/make.bat`

### Fixed
- GPU NaN bug: CUDA color kernel threshold mismatch (`psi_sq_total > 1e-30f`)
- GPU boundary masking: added missing Ψ boundary masking in color CUDA kernel
- Color variance `RuntimeWarning`: safe denominator pattern for zero-field input
- Schwarzschild chi `RuntimeWarning`: guard `np.where` before `np.sqrt`

### Improved
- Test suite: 202 → 307 tests (all passing); coverage 64 % → 91 %

## [1.0.0] - 2026-03-22

### Added
- **Visualisation module** (`lfm.viz`): `plot_slice`, `plot_three_slices`, `plot_chi_histogram`, `plot_evolution`, `plot_energy_components`, `plot_radial_profile`, `plot_isosurface`, `plot_power_spectrum`, `plot_trajectories`, `plot_sweep`
- **Power spectrum analyser** (`lfm.analysis.spectrum`): `power_spectrum()` — radially-averaged FFT P(k)
- **Particle tracker** (`lfm.analysis.tracker`): `track_peaks()`, `flatten_trajectories()`
- **Parameter sweep runner** (`lfm.sweep.sweep`)
- **GOV-02 v14**: color variance term `κ_c·f_c·Σ|Ψₐ|²` with `KAPPA_C = 1/189`
- **GOV-01 v15**: cross-color coupling `ε_cc·χ²·(Ψₐ − Ψ̄)` with `EPSILON_CC = 2/17`
- Constants: `KAPPA_C`, `EPSILON_CC`, `ALPHA_S`, `N_COLORS`
- Config: `kappa_c`, `epsilon_cc`, `n_colors` parameters
- Docs: `troubleshooting.md`, `primer.md`
- Benchmark suite (`benchmarks/`) with evolver and field-ops benchmarks
- Read the Docs configuration (`.readthedocs.yaml`)

### Changed
- First stable API release — version number jump to `1.0.0` reflects production readiness

## [0.4.1] - 2026-03-20

### Fixed
- CI lint gate now catches formatting/import issues earlier via a dedicated quality job in `test.yml`
- Workflow matrix stability improved by running quality checks before test matrix fan-out
- GitHub Actions versions updated (`actions/checkout@v5`, `actions/setup-python@v6`) to address Node 20 deprecation warnings

### Improved
- Contributor workflow now installs and uses pre-commit by default (`pip install -e ".[dev]"` includes `pre-commit`)
- Pre-commit guidance clarified in `CONTRIBUTING.md` (`pre-commit install`, `pre-commit run`)
- Pre-commit hook scope narrowed to maintained library/test/workflow paths to avoid unrelated legacy formatting debt

## [0.4.0] - 2026-03-22

### Added
- **Metric analysis** (`lfm.analysis.metric`): `effective_metric_00`, `metric_perturbation`, `time_dilation_factor`, `gravitational_potential`, `schwarzschild_chi` — extract spacetime geometry from χ field
- **Phase/charge analysis** (`lfm.analysis.phase`): `phase_field`, `charge_density`, `phase_coherence`, `coulomb_interaction_energy` — electromagnetic observables from complex fields
- **Angular momentum** (`lfm.analysis.angular_momentum`): `angular_momentum_density`, `total_angular_momentum`, `precession_rate` — orbital and spin analysis
- **Boosted solitons** (`lfm.fields.boosted`): `boosted_soliton` — momentum-encoded solitons with phase gradient (complex) or time-derivative kick (real) for scattering experiments
- **2D parameter sweeps** (`lfm.sweep`): `sweep_2d()` — run all (v1, v2) combinations and collect metrics
- **GitHub governance**: issue/PR templates, SECURITY.md, CODE_OF_CONDUCT.md
- **Pre-commit hooks**: `.pre-commit-config.yaml` with ruff linting and formatting
- **Docs build**: `docs/Makefile` and `docs/make.bat` for local Sphinx builds
- **Coverage config**: `pyproject.toml` [tool.coverage.*] sections (excluding viz/ and kernel_source.py)

### Fixed
- **GPU NaN bug**: CUDA color kernel threshold mismatch (`psi_sq_total > 1e-30f` → `total_sq > 1e-30f`) that caused FTZ underflow to 0/0=NaN
- **GPU boundary masking**: Added missing Psi boundary masking in color CUDA kernel
- **Color variance RuntimeWarning**: Safe denominator pattern to avoid division-by-zero warning on zero-field input
- **Schwarzschild chi RuntimeWarning**: Compute `np.where(r > r_s, 1.0 - r_s/r, 0.0)` before `np.sqrt` to avoid invalid-value warning

### Improved
- Test suite: 202 → **307 tests** (all passing)
- Coverage: 64% → **91%** (excluding viz/ and CUDA kernel strings)
- Docs version: aligned at 0.4.0 across pyproject.toml, lfm/__init__.py, docs/conf.py

## [0.3.0] - 2026-03-21

### Added
- **Visualisation module** (`lfm.viz`): 10 plotting functions for rapid exploration
  - `plot_slice`, `plot_three_slices`, `plot_chi_histogram` — 2D field slices
  - `plot_evolution`, `plot_energy_components` — time-series dashboards
  - `plot_radial_profile` — χ(r) with 1/r reference overlay
  - `plot_isosurface` — 3D voxel rendering of χ wells/voids
  - `plot_power_spectrum` — Fourier P(k) visualisation
  - `plot_trajectories` — peak motion scatter plots
  - `plot_sweep` — parameter sweep line plots
- **Power spectrum analyser** (`lfm.analysis.spectrum`): `power_spectrum()` — radially-averaged FFT P(k) for any 3D field
- **Particle tracker** (`lfm.analysis.tracker`): `track_peaks()` and `flatten_trajectories()` — follow energy-density maxima across timesteps
- **Parameter sweep runner** (`lfm.sweep`): `sweep()` — run a batch of simulations varying one parameter, collect metrics
- **Docs**: `docs/troubleshooting.md` — common errors (NaN, CFL, slow, imports) with fixes
- **Docs**: `docs/primer.md` — "LFM in Five Minutes" physics primer

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
