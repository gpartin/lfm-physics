# lfm-physics Library — Gap Closure Project Plan

**Created**: 2026-03-22  
**Purpose**: Close all identified gaps between the `lfm-physics` library and canonical LFM physics (copilot-instructions.md).  
**Order**: Dependencies first, then ROI descending.

---

## P001 — constants.py: Add S_a / v16 constants
**Status**: DONE  
**Priority**: CRITICAL (prerequisite for everything)  
**Effort**: Low  
**ROI**: Unblocks all v16 work  

Add to `lfm/constants.py`:
- `SA_GAMMA = EPSILON_W = 0.1` — S_a decay rate, derived: γ = ε_W (same physics as weak coupling)
- `SA_L = CHI0 - 12 = 7` — S_a diffusion range in lattice units, derived: L = β₀
- `SA_D = SA_GAMMA * SA_L**2 = 4.9` — S_a diffusion coefficient, derived: D = γL²
- `Z2_COORD = 2 * D_ST**2 = 32` — Second coordination shell on 4D lattice
- `RANK_SU3 = N_COLORS - 1 = 2` — Rank of SU(3) gauge group  
- `KAPPA_TUBE = (Z2_COORD - RANK_SU3) * KAPPA = 30/63 ≈ 0.476` — Smoothed color variance coupling
- `KAPPA_STRING = KAPPA_C = KAPPA / 3 = 1/189` — Color current variance coupling (v15-style)
- `BETA_0 = int(SA_L)` — QCD β₀ coefficient (same as SA_L)

---

## P002 — config.py: Add S_a / v16 parameters
**Status**: DONE  
**Priority**: CRITICAL (prerequisite for all SA simulations)  
**Effort**: Low  
**ROI**: Enables confinement simulations  

Add to `SimulationConfig`:
- `kappa_tube: float = 0.0` — SCV coupling (0 = off, backward compatible; KAPPA_TUBE for full v16)
- `kappa_string: float = 0.0` — CCV coupling (0 = off; KAPPA_STRING for v15-style)
- `sa_gamma: float = SA_GAMMA` — S_a decay rate (only used when kappa_tube > 0)
- `sa_d: float = SA_D` — S_a diffusion coefficient (only used when kappa_tube > 0)

Add derived property:
- `@property sa_enabled: bool` — `kappa_tube > 0`

---

## P003 — numpy_backend.py: Add CCV + SCV + S_a diffusion
**Status**: DONE  
**Priority**: CRITICAL  
**Effort**: High  
**ROI**: Makes v16 confinement work on CPU (development/testing path)  

Update `NumpyBackend.step_color()`:
- Add params: `kappa_string`, `kappa_tube`, `sa_fields_in`, `sa_fields_out`, `sa_gamma`, `sa_d`
- Compute per-color currents `j_{a,x/y/z}` for each color a
- Compute CCV = `Σ_d [ Σ_a j²_{a,d} - (1/3)(Σ_a j_{a,d})² ]` for `kappa_string > 0`
- Read SCV from `sa_fields_in`: `SCV = Σ_a S_a² - (1/3)(Σ_a S_a)²` for `kappa_tube > 0`
- Add `- kappa_string * ccv - kappa_tube * scv` to GOV-02 chi acceleration
- Before returning: if `kappa_tube > 0`, update each `S_a` via:
  `S_a^{n+1} = S_a + dt * (D * Lap(S_a) + |Ψ_a|² - γ * S_a)`
  (7-pt Laplacian is sufficient for diffusion, dt not dt2)

---

## P004 — kernel_source.py: Add v16 CUDA kernels
**Status**: DONE  
**Priority**: HIGH (GPU path for confinement)  
**Effort**: High  
**ROI**: Full GPU-accelerated v16 confinement (10-100× faster than CPU)  

- Update `EVOLUTION_KERNEL_SRC` to accept `kappa_string`, `kappa_tube`, `Sa_in` params
- Add CCV computation to the existing per-color loop
- Add SCV read from `Sa_in `
- Add `- kappa_string * ccv - kappa_tube * scv` to GOV-02
- Add new `SA_DIFFUSION_KERNEL_SRC` string constant (ported from `lfm_universe_simulator.py`)

---

## P005 — cupy_backend.py: Wire v16 kernels
**Status**: DONE  
**Priority**: HIGH  
**Effort**: Medium  
**ROI**: GPU confinement simulations  

- Compile `SA_DIFFUSION_KERNEL_SRC` as `SA_DIFFUSION_KERNEL`
- Update `step_color()` to accept and pass `kappa_string`, `kappa_tube`, `sa_fields`
- Call both kernels in sequence when `kappa_tube > 0`

---

## P006 — evolver.py: S_a fields state management
**Status**: DONE  
**Priority**: HIGH  
**Effort**: Medium  
**ROI**: Transparent S_a lifecycle management  

- Add `sa_fields` array to evolver state (shape `(3, N, N, N)`, float32, initialized to zeros)
- Pass `sa_fields` to `step_color` when `config.sa_enabled`
- Expose `get_sa_fields()` / `set_sa_fields(arr)` methods

---

## P007 — simulation.py: SA properties + velocity in place_soliton
**Status**: DONE  
**Priority**: HIGH (two independent improvements)  
**Effort**: Medium  
**ROI**: Exposes S_a + enables collision/galaxy work  

**Part A — S_a access:**
- Add `@property sa_fields` → `get_sa_fields()` result
- Add `@sa_fields.setter` → `set_sa_fields()`
- Update `save_checkpoint()` to include `sa_fields` if `sa_enabled`
- Update `load_checkpoint()` to restore `sa_fields`

**Part B — velocity in `place_soliton`:**
- Add `velocity: tuple[float, float, float] | None = None` param to `place_soliton()`
- When velocity is not None, call `boosted_soliton()` internally instead of `gaussian_soliton()`
- Same for `place_solitons()` — add `velocities: list[tuple] | None = None` param

---

## P008 — analysis/confinement.py: New confinement analysis module
**Status**: DONE  
**Priority**: HIGH (canonical physics tool)  
**Effort**: Medium  
**ROI**: Enables publication-quality confinement measurements  

New file `lfm/analysis/confinement.py`:
- `smoothed_color_variance(sa_fields)` → `NDArray` shape (N,N,N) — SCV at each lattice point
- `color_current_variance(psi_r, psi_i, n_colors=3)` → `NDArray` shape (N,N,N) — CCV
- `string_tension(chi_profiles, separations)` → `tuple[float, float]` — (σ, R²) from linear fit
- `flux_tube_profile(chi, sa_fields, p0, p1, samples=64)` → `dict` — chi and SCV along connecting line
- `measure_chi_midpoint(chi, p0, p1)` → `float` — chi value at midpoint between two points

Update `lfm/analysis/__init__.py` to export all four.

---

## P009 — analysis/observables.py: Add rotation_curve
**Status**: DONE  
**Priority**: HIGH (rotating galaxy simulation requires this)  
**Effort**: Medium  
**ROI**: Enables galaxy rotation curve demo (dark matter smoking gun)  

Add to `lfm/analysis/observables.py`:
- `rotation_curve(energy_density, center=None, n_bins=32, max_radius=None)` → `dict`  
  Returns `{r, v_circ, mass_enclosed, chi_min_profile}`  
  Method: bin energy in spherical shells, compute v² = GM_enc/r from `M_enc ∝ Σ|Ψ|²`
- `keplerian_velocity(chi, center, r, c=1.0)` → `float`  
  v_K = sqrt(-d(chi_potential)/dr * r) at radius r

---

## P010 — analysis/metric.py: Add horizon finder
**Status**: DONE  
**Priority**: MEDIUM  
**Effort**: Low  
**ROI**: Enables BH formation experiments  

Add to `lfm/analysis/metric.py`:
- `find_apparent_horizon(chi, chi0=CHI0, fraction=0.5)` → `dict`  
  Returns `{exists: bool, radius: float, center: tuple, chi_at_horizon: float}`  
  Criterion: outermost shell where `chi/chi0 < fraction`
- `horizon_mass(r_horizon, c=1.0)` → `float`  
  From Schwarzschild: `M = r_s * c² / 2` with `r_s = r_horizon`

---

## P011 — fields/arrangements.py: Galactic disk initializer
**Status**: DONE  
**Priority**: MEDIUM  
**Effort**: Medium  
**ROI**: Enables rotating galaxy simulation out-of-the-box  

Add to `lfm/fields/arrangements.py`:
- `disk_positions(N, n_particles, center=None, r_min=5, r_max=None, random_seed=42)` → `NDArray` shape (n,3)  
  Uniform random placement in disk plane (z=center), radii in [r_min, r_max]
- `disk_velocities(positions, chi, center=None, c=1.0)` → `NDArray` shape (n,3)  
  Keplerian circular orbital velocities from chi gradient. Direction: tangential to radius vector.
- `initialize_disk(sim, n_particles, amplitude=None, r_min=5, r_max=None, random_seed=42)` → `None`  
  Convenience function: calls disk_positions, equilibrates chi, calls disk_velocities,  
  then places each soliton with the computed velocity (requires P007 velocity support).

---

## P012 — analysis/tracker.py: Collision event detection
**Status**: DONE  
**Priority**: MEDIUM  
**Effort**: Low  
**ROI**: Makes collision experiments easy to analyze  

Add to `lfm/analysis/tracker.py`:
- `detect_collision_events(trajectories, min_sep=3.0)` → `list[dict]`  
  Input: output of `track_peaks()`. Returns `[{step, i, j, sep, type}]`  
  where `type` is `"approach"` (first time sep < min_sep) or `"merge"` (peak disappears).
- `compute_impact_parameter(traj_i, traj_j)` → `float`  
  Minimum perpendicular distance between two straight-line trajectory extrapolations.

---

## P013 — planning.py: New use_case_preset entries
**Status**: DONE  
**Priority**: MEDIUM  
**Effort**: Low  
**ROI**: Discoverability of new features  

Add to `UseCaseName` literal type and `use_case_preset()` function:
- `"rotating_galaxy"` → 128³ REAL, frozen boundary, sigma=grid/24
- `"particle_collision"` → 64³ COMPLEX, two boosted solitons, frozen boundary
- `"string_tension"` → 64³ COLOR, kappa_c=KAPPA_C, kappa_tube=10*KAPPA, lambda_self=LAMBDA_H

---

## P014 — __init__.py: Export all new symbols
**Status**: DONE  
**Priority**: LOW (polish, but needed for usability)  
**Effort**: Low  
**ROI**: Clean public API  

Add to `lfm/__init__.py`:
- From `lfm.constants`: `SA_GAMMA`, `SA_L`, `SA_D`, `KAPPA_TUBE`, `KAPPA_STRING`, `BETA_0`, `Z2_COORD`, `RANK_SU3`
- From `lfm.analysis.confinement`: `smoothed_color_variance`, `color_current_variance`, `string_tension`, `flux_tube_profile`, `measure_chi_midpoint`
- From `lfm.analysis.observables`: `rotation_curve`, `keplerian_velocity`
- From `lfm.analysis.metric`: `find_apparent_horizon`, `horizon_mass`
- From `lfm.fields.arrangements`: `disk_positions`, `disk_velocities`, `initialize_disk`
- From `lfm.analysis.tracker`: `detect_collision_events`, `compute_impact_parameter`

---

## P015 — tests/test_sa_fields.py: Tests for new features
**Status**: DONE  
**Priority**: HIGH (required for confidence in new physics)  
**Effort**: High  
**ROI**: Prevents regressions, validates correctness  

New `tests/test_sa_fields.py` file covering:
- S_a field initialization (zeros)
- S_a diffusion update (source → S_a grows to near |Ψ|²)
- SCV vanishes for color singlet (equal S_a)
- SCV maximum for single-color (one S_a non-zero)
- kappa_tube=0 backward compatibility (identical to v14/v15 results)
- Chi deepens between colored sources when kappa_tube > 0
- save/load checkpoint round-trip with sa_fields
- `place_soliton(velocity=...)` produces boosted field
- `rotation_curve()` returns correct shape
- `find_apparent_horizon()` finds horizon in Schwarzschild chi profile
- `disk_positions()` returns positions in correct range
- `detect_collision_events()` finds approach event

---

## P016 — examples/17_confinement_v16.py: Flux tube demonstration
**Status**: DONE  
**Priority**: HIGH  
**Effort**: Medium  
**ROI**: Canonical demonstration of confinement  

Script structure:
1. Create 64³ simulation, FieldLevel.COLOR, kappa_c=KAPPA_C, kappa_tube=10*KAPPA
2. Place two colored solitons (color_a=0 dominated) at separation=20
3. Equilibrate chi
4. Run 15,000 steps
5. Measure: chi midpoint vs control (kappa_tube=0), flux_tube_profile, string_tension
6. Plot: chi slice through tube, profile, string tension vs separation
7. Assert: H₀ rejected (chi depression > 0.3 threshold)

---

## P017 — examples/18_particle_collision.py: Head-on collision
**Status**: DONE  
**Priority**: MEDIUM  
**Effort**: Medium  
**ROI**: Dramatic demo of LFM particle dynamics  

Script structure:
1. Create 64³ simulation, FieldLevel.COMPLEX
2. Place two solitons with opposite velocities on collision course
3. Equilibrate
4. Run with track_peaks callback
5. Detect collision event
6. Measure: energy before/after, secondary fragments, chi landscape
7. Plot: trajectory, chi evolution, energy conservation

---

## P018 — examples/19_rotating_galaxy.py: Flat rotation curve
**Status**: DONE  
**Priority**: MEDIUM  
**Effort**: Medium  
**ROI**: Most iconic LFM dark matter prediction  

Script structure:
1. Create 128³ simulation, FieldLevel.REAL
2. Place central massive soliton (galaxy core)
3. Place disk solitons with Keplerian velocities (use initialize_disk)
4. Run for several orbital periods
5. Measure: rotation_curve at t=0 vs t=late
6. Show: inner curve Keplerian, outer curve flatter (chi memory = dark matter halo)
7. Plot: v(r) curves, chi slice

---

## Summary Table

| ID | File | Description | Status | Priority |
|----|------|-------------|--------|----------|
| P001 | `lfm/constants.py` | S_a / v16 constants | DONE | CRITICAL |
| P002 | `lfm/config.py` | SA params in SimulationConfig | DONE | CRITICAL |
| P003 | `lfm/core/backends/numpy_backend.py` | CCV + SCV + SA diffusion | DONE | CRITICAL |
| P004 | `lfm/core/backends/kernel_source.py` | v16 CUDA kernels | DONE | HIGH |
| P005 | `lfm/core/backends/cupy_backend.py` | Wire v16 kernels | DONE | HIGH |
| P006 | `lfm/core/evolver.py` | SA fields state | DONE | HIGH |
| P007 | `lfm/simulation.py` | SA properties + velocity | DONE | HIGH |
| P008 | `lfm/analysis/confinement.py` | New confinement analysis | DONE | HIGH |
| P009 | `lfm/analysis/observables.py` | rotation_curve | DONE | HIGH |
| P010 | `lfm/analysis/metric.py` | Horizon finder | DONE | MEDIUM |
| P011 | `lfm/fields/arrangements.py` | Galactic disk init | DONE | MEDIUM |
| P012 | `lfm/analysis/tracker.py` | Collision detection | DONE | MEDIUM |
| P013 | `lfm/planning.py` | New presets | DONE | MEDIUM |
| P014 | `lfm/__init__.py` | Export new symbols | DONE | LOW |
| P015 | `tests/test_sa_fields.py` | Tests for new features | DONE | HIGH |
| P016 | `examples/17_confinement_v16.py` | Flux tube demo | DONE | HIGH |
| P017 | `examples/18_particle_collision.py` | Collision demo | DONE | MEDIUM |
| P018 | `examples/19_rotating_galaxy.py` | Galaxy rotation curve | DONE | MEDIUM |
| P019 | `paper_experiments/collider_lhc_style.py` | LHC-style 3×3 collision event sweep | DONE | LOW |
| P020 | `paper_experiments/sparc_rotation_curve_comparison.py` | SPARC 175-galaxy chi-memory analysis | DONE | LOW |
| P021 | Website tutorial 17 | `17-string-tension-confinement/page.tsx` | DONE | LOW |
| P022 | Website tutorial 18 | `18-particle-collision/page.tsx` | DONE | LOW |
| P023 | Website tutorial 19 | `19-rotating-galaxy/page.tsx` | DONE | LOW |
| P024 | Website tutorials index | Add entries 17/18/19, "Nineteen scripts", badge update | DONE | LOW |
| P025 | Website tutorial 16 nav | Update right-nav link to → tutorial 17 | DONE | LOW |

---

## Website Tutorial Status (as of P021–P025)

All 19 tutorials exist at `src/app/tutorials/NN-slug/page.tsx` in the LFM website repo.

| # | Slug | Level | Status |
|---|------|-------|--------|
| 01–16 | various | Beginner/Intermediate/Advanced | pre-existing |
| 17 | `17-string-tension-confinement` | Advanced | ADDED P021 |
| 18 | `18-particle-collision` | Advanced | ADDED P022 |
| 19 | `19-rotating-galaxy` | Advanced | ADDED P023 |

Navigation chain: 01 ↔ 02 ↔ … ↔ 16 ↔ 17 ↔ 18 ↔ 19 → All tutorials

---

## Potential Future Features (P026+)

Identified during P019–P025 implementation:

| ID | Feature | Notes |
|----|---------|-------|
| P026 | `lfm.rotation_curve_fit(sparc_row)` | Auto-fit tau to match one SPARC galaxy |
| P027 | `lfm.collider_event_display(result)` | Rich ASCII event display with peak trajectory art |
| P028 | `lfm.galaxy_summary_plot(sim, sparc_gid)` | Side-by-side LFM vs SPARC v(r) figure |
| P029 | `lfm.sparc_load(path)` | Official SPARC loader in lfm.analysis |
| P030 | `initialize_disk` polar offset parameter | Add `b_cells` to place solitons with non-zero ecliptic inclination |

---

## v1.1.0 Features (P031–P047) — COMPLETED

| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| P031 | `Simulation.run_with_snapshots()` | ✅ DONE | Block-loop design; fields=["chi","psi_real","psi_imag","energy_density"] |
| P032 | `lfm.viz.animate_slice()` | ✅ DONE | FuncAnimation, optional GIF/MP4 save via pillow/ffmpeg |
| P033 | `lfm.viz.animate_three_slices()` | ✅ DONE | Three mid-plane panels in one animation |
| P034 | `lfm.viz.project_field()` | ✅ DONE | sum/mean/max column-density projection |
| P035 | `lfm.viz.plot_projection()` | ✅ DONE | Log/linear 2-D projection image with colorbar |
| P036 | `lfm.viz.spacetime_diagram()` | ✅ DONE | χ(x,t) heatmap from snapshot sequence |
| P037 | `lfm.correlation_function()` | ✅ DONE | Wiener–Khinchin FFT-based two-point ξ(r) |
| P038 | `lfm.matter_power_spectrum()` | ✅ DONE | Dimensionless Δ²(k) = k³P(k)/(2π²) |
| P039 | `lfm.halo_mass_function()` | ✅ DONE | scipy.ndimage.label connected-components halos |
| P040 | `lfm.void_statistics()` | ✅ DONE | High-χ void finder + size distribution |
| P041 | `lfm.gravitational_wave_strain()` | ✅ DONE | h = (χ − χ₀)/χ₀ strain field |
| P042 | `lfm.gw_quadrupole()` | ✅ DONE | Reduced traceless quadrupole tensor I_ij |
| P043 | `lfm.gw_power()` | ✅ DONE | Peters-formula L_GW from snapshot sequence |
| P044 | `examples/20_gravitational_waves.py` | ✅ DONE | Binary inspiral + GW chirp + spacetime diagram |
| P045 | `examples/21_wave_interference.py` | ✅ DONE | Double-source interference, phase sweep |
| P046 | `examples/22_soliton_modes.py` | ✅ DONE | Breathing-mode FFT vs ω_H = √(8λ_H)·χ₀ |
| P047 | `tests/test_v11_features.py` (50 tests) | ✅ DONE | 405/405 total passing |

---

## Derivation Rationale for S_a Parameters

All four S_a parameters are DERIVED from χ₀ = 19 (Session 143):

| Parameter | Formula | Value | Derivation |
|-----------|---------|-------|------------|
| γ (decay rate) | ε_W = 2/(χ₀+1) | 0.1 | Same lattice physics as weak helicity coupling |
| L (range) | β₀ = χ₀−12 | 7 | Center + face modes = axially independent confinement range |
| D (diffusion) | γ·L² | 4.9 | Follows algebraically from γ and L |
| κ_tube | (z₂ − rank_G)·κ | 30/63 | z₂ = 2D_st² = 32 neighbor channels, minus rank(SU(3)) = 2 |

CFL stability check: `dt < 1/(6D) = 1/29.4 ≈ 0.034`. Our dt=0.02 satisfies this with 1.7× margin.

---

## Key Experimental Results (from LFM_CONFINEMENT_MECHANISM.md)

- Static 1D analysis: Linear potential R² = 0.989
- Dynamic 3D test (Session 143): R² = 0.882, σ = 0.0275 lattice units
- H₀ REJECTED (delta_chi > 0.3 threshold at separation=12,16,20,24)
- At κ_tube = 30κ: 28-unit chi depression (but unstable >15k steps without Mexican hat)
- At κ_tube = 10κ: Stable 20k+ steps, clear tube signal, H₀ rejected
- Recommended default for experiments: `kappa_tube = 10 * KAPPA` (safe) or `kappa_tube = KAPPA_TUBE` with `lambda_self = LAMBDA_H`
