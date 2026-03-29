# LFM Particle Inventory Audit & Implementation Plan

**Date**: 2026-03-28
**Status**: Implementation in progress
**Library**: `lfm-physics` (c:\Papers\lfm-physics\)

---

## 1. Philosophy: Particles in LFM

LFM is a **substrate hypothesis**. Two governing equations (GOV-01 + GOV-02)
run on a discrete lattice and all physics emerges. Particles are NOT
fundamental objects — they are **standing-wave eigenmodes** trapped in
self-consistent χ-wells.

**What the catalog provides:**
- **Recipes**: Initial conditions (amplitude, sigma, phase, field level) that
  produce stable eigenmodes after equilibration via the SCF solver
- **Expected properties**: Mass ratio, charge, spin — these should EMERGE
  from the simulation, not be imposed
- **Composition**: For composite particles, the constituent quark content
  so they can be assembled from fundamental building blocks

**What the simulation does:**
- Eigenmode solver seeds a Gaussian, equilibrates χ via Poisson (GOV-04),
  evolves GOV-01 only (radiation escapes, bound mode stays), re-equilibrates
- The RESULT is a self-consistent {Ψ, χ} eigenstate
- Properties like mass (χ_min depth), charge (phase), spin (angular momentum
  in the solution) are MEASURED from the evolved field, not hardcoded

**Key principle**: The catalog entries are like "recipes in a cookbook" —
they tell you the ingredients and procedure, but the food (particle) is
what emerges from the cooking (simulation).

---

## 2. Current Coverage

### 2.1 Existing Catalog (15 particles)

| # | Name | Symbol | Category | mass_ratio | charge | phase | field_level |
|---|------|--------|----------|------------|--------|-------|-------------|
| 1 | electron | e⁻ | lepton | 1.0 | -1 | 0.0 | COMPLEX |
| 2 | positron | e⁺ | lepton | 1.0 | +1 | π | COMPLEX |
| 3 | muon | μ⁻ | lepton | 206.768 | -1 | 0.0 | COMPLEX |
| 4 | antimuon | μ⁺ | lepton | 206.768 | +1 | π | COMPLEX |
| 5 | tau | τ⁻ | lepton | 3477.23 | -1 | 0.0 | COMPLEX |
| 6 | antitau | τ⁺ | lepton | 3477.23 | +1 | π | COMPLEX |
| 7 | up_quark | u | quark | 4.34 | +2/3 | π | COLOR |
| 8 | down_quark | d | quark | 9.22 | -1/3 | 0.0 | COLOR |
| 9 | strange_quark | s | quark | 183.71 | -1/3 | 0.0 | COLOR |
| 10 | charm_quark | c | quark | 2491.7 | +2/3 | π | COLOR |
| 11 | proton | p | baryon | 1836.15 | +1 | π | COMPLEX |
| 12 | antiproton | p̄ | baryon | 1836.15 | -1 | 0.0 | COMPLEX |
| 13 | neutron | n | baryon | 1838.68 | 0 | 0.0 | REAL |
| 14 | antineutron | n̄ | baryon | 1838.68 | 0 | 0.0 | REAL |
| 15 | photon | γ | boson | 0.0 | 0 | 0.0 | REAL |

### 2.2 Missing Particles (47 to add)

#### A. Fundamental Fermions — Quarks (6 missing)

| Name | Symbol | mass_ratio | charge | phase | field_level | Notes |
|------|--------|------------|--------|-------|-------------|-------|
| anti_up_quark | ū | 4.34 | -2/3 | 0.0 | COLOR | flip charge/phase of up |
| anti_down_quark | d̄ | 9.22 | +1/3 | π | COLOR | flip charge/phase of down |
| anti_strange_quark | s̄ | 183.71 | +1/3 | π | COLOR | flip charge/phase of strange |
| anti_charm_quark | c̄ | 2491.7 | -2/3 | 0.0 | COLOR | flip charge/phase of charm |
| bottom_quark | b | 8189.7 | -1/3 | 0.0 | COLOR | 3rd gen down-type |
| top_quark | t | 338,274 | +2/3 | π | COLOR | 3rd gen up-type |

Plus their antiparticles:

| Name | Symbol | mass_ratio | charge | phase | field_level |
|------|--------|------------|--------|-------|-------------|
| anti_bottom_quark | b̄ | 8189.7 | +1/3 | π | COLOR |
| anti_top_quark | t̄ | 338,274 | -2/3 | 0.0 | COLOR |

#### B. Fundamental Fermions — Neutrinos (6 missing)

| Name | Symbol | mass_ratio | charge | phase | field_level | Notes |
|------|--------|------------|--------|-------|-------------|-------|
| electron_neutrino | νₑ | ~0.004 | 0 | 0.0 | REAL | Nearly massless |
| muon_neutrino | ν_μ | ~0.017 | 0 | 0.0 | REAL | Nearly massless |
| tau_neutrino | ν_τ | ~0.030 | 0 | 0.0 | REAL | Nearly massless |
| anti_electron_neutrino | ν̄ₑ | ~0.004 | 0 | 0.0 | REAL | CPT partner |
| anti_muon_neutrino | ν̄_μ | ~0.017 | 0 | 0.0 | REAL | CPT partner |
| anti_tau_neutrino | ν̄_τ | ~0.030 | 0 | 0.0 | REAL | CPT partner |

Note: Neutrino masses are tiny (< 1 eV). In LFM, they are neutral, nearly
massless eigenmodes. They require REAL field level (no charge). Their tiny
mass means they are barely-trapped modes — the eigenmode solver may not
converge for them at practical grid sizes. They may need special handling
(effectively massless propagating waves with tiny χ confinement).

#### C. Gauge Bosons (4 missing)

| Name | Symbol | mass_ratio | charge | spin | field_level | Notes |
|------|--------|------------|--------|------|-------------|-------|
| w_plus | W⁺ | 157,294 | +1 | 1 | COMPLEX | Massive gauge boson |
| w_minus | W⁻ | 157,294 | -1 | 1 | COMPLEX | Massive gauge boson |
| z_boson | Z⁰ | 178,450 | 0 | 1 | REAL | Massive neutral gauge |
| gluon | g | 0.0 | 0 | 1 | COLOR | Massless color-octet |

#### D. Scalar Boson (1 missing)

| Name | Symbol | mass_ratio | charge | spin | field_level | Notes |
|------|--------|------------|--------|------|-------------|-------|
| higgs | H⁰ | 244,890 | 0 | 0 | REAL | χ excitation (Mexican hat) |

In LFM, the Higgs is a χ-field oscillation mode: ω_H = √(8λ_H)·χ₀ ≈ 19.30.
It's not a Ψ-field soliton but a localized χ-perturbation.

#### E. Light Mesons (8 missing, built from quarks)

| Name | Symbol | Quarks | mass_ratio | charge | phase | Stable? |
|------|--------|--------|------------|--------|-------|---------|
| pion_plus | π⁺ | ud̄ | 273.13 | +1 | π | No (τ=26ns) |
| pion_minus | π⁻ | ūd | 273.13 | -1 | 0.0 | No |
| pion_zero | π⁰ | (uū−dd̄)/√2 | 263.89 | 0 | 0.0 | No (τ=8.4e-17s) |
| kaon_plus | K⁺ | us̄ | 966.12 | +1 | π | No (τ=12.4ns) |
| kaon_minus | K⁻ | ūs | 966.12 | -1 | 0.0 | No |
| kaon_zero | K⁰ | ds̄ | 974.55 | 0 | 0.0 | No |
| eta | η | flavor mix | 1073.2 | 0 | 0.0 | No |
| rho | ρ⁰ | (uū−dd̄)/√2 | 1513.4 | 0 | 0.0 | No |

#### F. Strange Baryons (4 missing, built from quarks)

| Name | Symbol | Quarks | mass_ratio | charge | Stable? |
|------|--------|--------|------------|--------|---------|
| lambda_baryon | Λ⁰ | uds | 2183.5 | 0 | No |
| sigma_plus | Σ⁺ | uus | 2327.6 | +1 | No |
| sigma_minus | Σ⁻ | dds | 2343.1 | -1 | No |
| xi_minus | Ξ⁻ | dss | 2578.5 | -1 | No |

#### G. Charm Mesons (4 missing)

| Name | Symbol | Quarks | mass_ratio | charge |
|------|--------|--------|------------|--------|
| d_plus | D⁺ | cd̄ | 3659.1 | +1 |
| d_minus | D⁻ | c̄d | 3659.1 | -1 |
| d_zero | D⁰ | cū | 3649.7 | 0 |
| j_psi | J/ψ | cc̄ | 6057.0 | 0 |

#### H. Bottom Mesons (4 missing)

| Name | Symbol | Quarks | mass_ratio | charge |
|------|--------|--------|------------|--------|
| b_plus | B⁺ | ub̄ | 10,339 | +1 |
| b_minus | B⁻ | ūb | 10,339 | -1 |
| b_zero | B⁰ | db̄ | 10,339 | 0 |
| upsilon | Υ | bb̄ | 18,518 | 0 |

#### I. Charm/Bottom Baryons (4 missing)

| Name | Symbol | Quarks | mass_ratio | charge |
|------|--------|--------|------------|--------|
| lambda_c | Λ_c⁺ | udc | 4479.8 | +1 |
| xi_c_plus | Ξ_c⁺ | usc | 4832.0 | +1 |
| lambda_b | Λ_b⁰ | udb | 11,004 | 0 |
| omega_baryon | Ω⁻ | sss | 3276.8 | -1 |

---

## 3. Coverage Summary

| Category | Existing | Missing | Total | % Done |
|----------|----------|---------|-------|--------|
| Charged Leptons | 6 | 0 | 6 | 100% |
| Neutrinos | 0 | 6 | 6 | 0% |
| Quarks (inc. anti) | 4 | 8 | 12 | 33% |
| Gauge Bosons | 1 (γ) | 4 | 5 | 20% |
| Higgs | 0 | 1 | 1 | 0% |
| Light Mesons | 0 | 8 | 8 | 0% |
| Strange Baryons | 0 | 4 | 4 | 0% |
| Charm Mesons | 0 | 4 | 4 | 0% |
| Bottom Mesons | 0 | 4 | 4 | 0% |
| Heavy Baryons | 0 | 4 | 4 | 0% |
| Nucleons | 4 | 0 | 4 | 100% |
| **TOTAL** | **15** | **47** | **62** | **24%** |

---

## 4. Implementation Plan

### Phase 1: Complete Fundamental Particles (21 entries)
Add to `catalog.py`:
- 8 antiquarks + bottom/top quarks (COLOR level, flip charge/phase for anti)
- 6 neutrinos (REAL level, nearly massless)
- W±, Z⁰ (massive gauge bosons)
- Gluon (massless, COLOR level)
- Higgs (REAL level, χ-mode excitation)

These are all point-like fundamentals that the eigenmode solver can handle
directly (or that represent special LFM modes like Higgs = χ oscillation).

### Phase 2: Add `quark_content` field to Particle dataclass
Add optional `quark_content: tuple[str, ...] | None` field to enable
composite particle construction from constituents.

### Phase 3: Meson Factory
Extend `composite.py` with `create_meson(name)`:
- Place quark + antiquark at specified separation
- Equilibrate combined χ-well
- Let GOV-01 + GOV-02 evolve to find bound state

### Phase 4: Baryon Factory  
Extend `composite.py` with `create_baryon(name)`:
- Place 3 quarks in triangular arrangement
- Same equilibration procedure

### Phase 5: Collision Setup (Particle Smasher)
Extend `factory.py` with `create_collision(name_a, name_b, energy, ...)`:
- Unified API for throwing any two particles at each other
- Automatic grid sizing, velocity calculation from CM energy
- Post-collision analysis (product identification)

---

## 5. Movement Abstraction

### Current State
- `boost_soliton_solution(sol, velocity)` — works for eigenmode-solved particles
- `measure_center_of_energy(sim)` — measure position via χ²·|Ψ|² weighting  
- `measure_velocity(sim)` — measure velocity via momentum/energy

### Issues
1. Only works with `SolitonSolution` (eigenmode solver output)
2. Starts with flat χ (loses self-consistent chi structure for tracking)
3. No unified "move any particle" API
4. No support for composite particle movement
5. Practical velocity limit ~0.08c (dispersion at higher v)

### Design Decision
Movement is already **generic** in principle — all particles are Ψ-field
solitons in χ-wells, and they all move the same way (boost = Lorentz-like
phase gradient). The existing `create_two_particles()` function already
provides the core smasher setup. What we need:

1. **Unified collision factory** (`create_collision`) that:
   - Accepts any particle names + collision energy
   - Computes velocities from CM frame kinematics
   - Places particles with correct separation
   - Returns a ready-to-run simulation

2. **Post-collision analysis** tools to identify products

---

## 6. LFM Philosophy: No Conflicts

**Does the substrate hypothesis conflict with any particle?**

No. Every particle in the Standard Model maps to an LFM construct:

| SM Particle Type | LFM Representation |
|------------------|-------------------|
| Fermion (lepton/quark) | Standing-wave eigenmode in χ-well (GOV-01 bound state) |
| Photon | Propagating Ψ wave in vacuum (massless, χ₀ background) |
| W±, Z⁰ | Massive eigenmode (spin-1 from vector Ψ structure) |
| Gluon | Color-carrying propagating mode (massless, COLOR level) |
| Higgs | χ-field oscillation mode (Mexican hat: ω_H ≈ 19.30) |
| Meson (qq̄) | Two-soliton bound state (quark + antiquark in shared χ-well) |
| Baryon (qqq) | Three-soliton bound state (3 quarks in shared χ-well) |
| Neutrino | Nearly-massless eigenmode (barely trapped, REAL field) |

The key insight: **all particles emerge from the same two equations**.
The catalog just specifies the initial conditions that produce each type.
Unstable particles (mesons, heavy baryons) will naturally decay in
simulation — this is a FEATURE, not a bug. Their catalog entry records
the expected eigenmode that forms transiently.

---

## 7. Mass Derivation Reference

All particle masses in LFM derive from angular momentum quantization:
**m/m_e = l(l+1)** where l = τ·χ₀ + offset

| Particle | l | l(l+1) | Measured m/m_e | Error |
|----------|---|--------|----------------|-------|
| electron | 0 | 0 → 1 (special) | 1.0 | exact |
| muon | 14 | 210 | 206.768 | 1.6% |
| tau | 59 | 3540 | 3477.23 | 1.8% |
| proton | 42 | 1806 | 1836.15 | 1.6% |

For the catalog, we use the **measured** mass_ratio values (experimental
data), not the LFM predictions. The LFM mass formulas are derivation
results to be tested, not inputs.
