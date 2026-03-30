# LFM Particle Validation Plan

**Goal**: Take every cataloged particle and validate it behaves like its real-world counterpart using ONLY GOV-01 + GOV-02. No external physics injected. Set conditions → run simulation → measure emergence → compare to known physics.

**Philosophy**: LFM is a substrate hypothesis. We don't program particles to have spin or decay — we place solitons with the right field level and coupling constants, evolve GOV-01/GOV-02, and *measure* whether spin-like behavior, decay, and correct interactions emerge. A failure is data, not a bug to fix by cheating.

---

## Status Legend

- [ ] **TODO** — Not started
- [~] **IN PROGRESS** — Work underway
- [x] **DONE** — Completed and verified

---

## Phase 0: Infrastructure & Config Presets

Before any particle experiments, we need reusable config presets, measurement tools, and the correct confinement kernel.

### Task 0.0: Implement v17 Helmholtz Kernel for S_a (PREREQUISITE)
**Files**: `lfm/core/backends/numpy_backend.py`, `lfm/core/backends/cupy_backend.py`
**Status**: [x] DONE
**Depends on**: Nothing

**Why first**: The current S_a implementation uses forward-Euler diffusion, which reaches the same steady state as the v17 Helmholtz kernel but has CFL stability constraints and transient lag. The Helmholtz kernel is the canonical confinement mechanism from `LFM_CONFINEMENT_MECHANISM.md` and must be in place before ANY color/confinement experiments (Phase 3+).

**Implementation — NumPy backend** (`step_color`):

Replace the current Euler diffusion loop for S_a:
```python
# CURRENT (v16 Euler diffusion):
# dS_a/dt = D∇²S_a + γ(|Ψ_a|² - S_a)
```

With FFT-based Helmholtz solve (quasi-static, instant response):
```python
# v17 Helmholtz kernel (per-color, per-step):
for a in range(3):
    psi_sq_a = psi_a_real[a]**2 + psi_a_imag[a]**2
    psi_sq_hat = np.fft.rfftn(psi_sq_a)
    S_a[a] = np.fft.irfftn(helmholtz_filter * psi_sq_hat, s=psi_sq_a.shape)
```

Where `helmholtz_filter` is precomputed once:
```python
# Precompute k² grid and Helmholtz filter at sim init:
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
kz = np.fft.rfftfreq(N, d=dx) * 2 * np.pi
k_sq = kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2
helmholtz_filter = sa_gamma / (sa_gamma + sa_d * k_sq)
```

**Implementation — CuPy backend**: Same approach using `cupyx.scipy.fft.rfftn`/`irfftn`.

**Properties preserved**:
- Same steady state as Euler diffusion (proven 0.0000% RMS error)
- No CFL stability constraint (FFT is unconditionally stable)
- No persistent S_a state arrays needed (computed fresh each step)
- Smoothing length L = √(D/γ) = √(4.9/0.1) = 7 lattice units (= β₀ = χ₀ - 12)

**Acceptance**: 
1. Place a single colored soliton, run 1000 steps with old Euler and new Helmholtz
2. Compare S_a fields at step 1000: RMS difference < 0.01%
3. Run existing color confinement tests — must still pass

---

### Task 0.1: Create Config Presets Module
**File**: `lfm/config_presets.py`
**Status**: [x] DONE
**Depends on**: Nothing

Create named preset configs that turn on the right physics terms for each tier:

```python
from lfm.config import SimulationConfig
from lfm.constants import (KAPPA, LAMBDA_H, EPSILON_W, KAPPA_C,
                           EPSILON_CC, KAPPA_STRING, KAPPA_TUBE,
                           SA_GAMMA, SA_D, CHI0)

def gravity_only() -> SimulationConfig:
    """Level 0: Real E, gravity only. For neutral massive particles."""
    return SimulationConfig()  # all defaults

def gravity_em() -> SimulationConfig:
    """Level 1: Complex Ψ, gravity + EM. For charged leptons, photons."""
    return SimulationConfig(
        field_level=1,
        lambda_self=LAMBDA_H,  # Mexican hat for vacuum stability
    )

def full_physics() -> SimulationConfig:
    """Level 2: 3-color Ψ_a, all four forces. For quarks, hadrons."""
    return SimulationConfig(
        field_level=2,
        lambda_self=LAMBDA_H,
        kappa_c=KAPPA_C,
        epsilon_cc=EPSILON_CC,
        kappa_string=KAPPA_STRING,
        kappa_tube=KAPPA_TUBE,
        sa_gamma=SA_GAMMA,
        sa_d=SA_D,
    )
```

**Acceptance**: Import presets, verify all coupling constants match the canonical values from `LFM_CONFINEMENT_MECHANISM.md`.

---

### Task 0.2: Create Measurement Toolkit
**File**: `lfm/analysis/measurements.py`
**Status**: [x] DONE
**Depends on**: Nothing

Standard measurement functions that extract physics from evolved fields WITHOUT injecting any external physics. Every experiment will call these.

```python
def measure_force_profile(sim, particle_positions, axis='r'):
    """Measure force between two solitons as function of separation.
    
    Method: Run GOV-01/GOV-02 for many steps at each separation.
    Measure chi gradient at midpoint → force = -dχ/dr.
    Returns: separations[], forces[] arrays.
    """

def measure_binding_energy(sim):
    """Total energy of system minus sum of isolated particle energies.
    Negative = bound state."""

def measure_angular_momentum_spectrum(sim):
    """Decompose field into spherical harmonics Y_lm.
    Returns: dict of {(l,m): amplitude}.
    Tells us effective spin content of a soliton."""

def measure_oscillation_frequency(sim, steps, probe_point):
    """FFT of E(t) at a probe point.
    Returns: dominant frequency ω.
    Compare to χ_local for mass verification: m = ℏω/c² ~ χ."""

def measure_lifetime(sim, steps, energy_threshold=0.1):
    """Run sim and track when localized energy drops below threshold.
    Returns: step at which particle "decayed" (energy dispersed).
    Stable particle → lifetime = ∞ (never drops)."""

def measure_phase_winding(sim):
    """For complex fields: compute ∮∇θ·dl around soliton.
    Returns: winding number (integer = charge quantization)."""

def measure_color_fraction(sim):
    """For color fields: compute f_c = [Σ|Ψ_a|⁴/(Σ|Ψ_a|²)²] - 1/3.
    Returns: f_c at soliton center.
    0 = balanced (singlet/hadron), 2/3 = single color (free quark)."""

def measure_scattering_angle(sim, steps):
    """Track two soliton centers of mass over time.
    Returns: asymptotic deflection angle, impact parameter."""
```

**Acceptance**: Unit tests for each function using synthetic field configurations.

---

### Task 0.3: Create Experiment Runner Framework
**File**: `lfm/experiment/runner.py`
**Status**: [x] DONE
**Depends on**: Task 0.1, 0.2

Standardized experiment runner that enforces the hypothesis framework from copilot-instructions:

```python
@dataclass
class ExperimentResult:
    name: str
    h0_description: str       # Null hypothesis
    h1_description: str       # Alternative hypothesis
    success_criterion: str    # How we decide
    measurements: dict        # Raw numbers
    h0_rejected: bool         # Did physics emerge?
    lfm_only_verified: bool   # No external physics?
    notes: str

def run_experiment(
    name: str,
    setup_fn: Callable,       # Creates sim + places particles
    measure_fn: Callable,     # Extracts measurements from evolved sim
    evaluate_fn: Callable,    # Compares measurements to expectations
    steps: int = 10000,
) -> ExperimentResult:
    """Standard experiment runner.
    
    1. setup_fn() → Simulation (configured with correct preset)
    2. sim.run(steps)
    3. measure_fn(sim) → measurements dict
    4. evaluate_fn(measurements) → (h0_rejected, notes)
    5. Return ExperimentResult
    """
```

**Acceptance**: Run a trivial experiment (single electron, verify it doesn't explode) using the framework.

---

## Phase 1: Single-Particle Validation (Does each particle EXIST stably?)

For each particle: place it alone, evolve GOV-01/GOV-02, measure if it persists as a localized soliton with the right mass and stability.

### Task 1.1: Electron Stability & Mass
**File**: `tests/validation/test_electron.py`
**Status**: [x] DONE
**Depends on**: Phase 0

**Setup**: `gravity_em()` config, place electron (phase=0, field_level=1) at center of N=64 grid.
**Run**: 50,000 steps with dt=0.02.
**Measure**:
- Energy retained (%) — should be >90% for stable particle
- Oscillation frequency ω via FFT — should match χ_well (mass)
- Phase winding number — should be 0 (charge = -1 from phase=0)
- χ_min (well depth) — should be self-consistent with eigenmode

**H₀**: Electron soliton disperses (energy <50% retained after 50k steps).
**H₁**: Electron is a stable bound state (energy >90% retained).
**Criterion**: REJECT H₀ if energy retention >90%.

---

### Task 1.2: Positron Stability & Charge Sign
**File**: `tests/validation/test_positron.py`
**Status**: [x] DONE
**Depends on**: Task 1.1

Same as electron but with phase=π. Verify:
- Same mass (same ω) — charge shouldn't affect mass
- Opposite phase from electron — this IS the charge difference
- Same stability (energy retention >90%)

**H₀**: Positron behaves differently from electron (different mass or stability).
**H₁**: Positron = electron with opposite phase (same mass, same stability).

---

### Task 1.3: Muon Stability & Mass Ratio
**File**: `tests/validation/test_muon.py`
**Status**: [x] DONE
**Depends on**: Task 1.1

**Setup**: `gravity_em()` config, place muon (l=14, significantly heavier than electron).
**Measure**:
- Mass ratio via ω_muon/ω_electron — should be ~206.8 (l(l+1)=210)
- Stability: should be LESS stable than electron (muon decays in real physics)
- Track energy over time — does it slowly leak? (would indicate instability/decay)

**H₀**: Muon has same mass as electron.
**H₁**: Muon is measurably heavier with mass ratio near l(l+1)=210.

---

### Task 1.4: Neutrino (Massless/Near-Massless Stability)
**File**: `tests/validation/test_neutrino.py`
**Status**: [x] DONE
**Depends on**: Task 1.1

**Setup**: `gravity_only()` config (neutrino is neutral), place neutrino (l=0, near-massless).
**Measure**:
- Does it propagate at ~c? (massless → v≈c)
- Does it stay localized or disperse? (real physics: neutrinos propagate freely)
- Energy density profile over time

**H₀**: Neutrino behaves like a massive trapped soliton.
**H₁**: Neutrino propagates freely at ~c (not trapped).

---

### Task 1.5: Proton Stability
**File**: `tests/validation/test_proton.py`
**Status**: [x] DONE
**Depends on**: Phase 0

**Setup**: `gravity_em()` config (proton is composite but simulated as single soliton at level 1).
**Measure**:
- Mass ratio ω_proton/ω_electron — should be ~1836
- Stability: proton should be EXTREMELY stable (lifetime >10³⁴ years)
- Energy retention after 100k steps

**Note**: This tests the proton as a single soliton. Phase 3 will test it as a 3-quark bound state.

---

### Task 1.6: Neutron Stability (Should Decay!)
**File**: `tests/validation/test_neutron.py`
**Status**: [x] DONE
**Depends on**: Task 1.5

**Setup**: `gravity_only()` config (neutron is neutral, field_level=0).
**Measure**:
- Mass ratio — should be slightly heavier than proton (~1838.7)
- Stability: free neutron SHOULD be unstable (τ ≈ 880s in real physics)
- Track energy: does it slowly leak out? (Would indicate emergent decay)
- If energy redistributes, measure final products — do they look like proton + electron + neutrino?

**H₀**: Neutron is as stable as proton.
**H₁**: Neutron shows some instability (energy leakage over time).

---

### Task 1.7: Sweep All 69 Particles (Automated Mass & Stability)
**File**: `tests/validation/test_particle_sweep.py`
**Status**: [x] DONE
**Depends on**: Tasks 1.1-1.6

Automated sweep over entire catalog:
1. For each particle: create with appropriate config preset (based on field_level)
2. Evolve 20,000 steps
3. Measure: energy retention %, oscillation frequency ω, χ_min
4. Compute mass ratio from ω relative to electron ω
5. Compare to catalog's mass_ratio (from l(l+1))
6. Report: table of all 69 particles with measured vs expected mass ratio, stability flag

**Output**: CSV/JSON with columns: name, field_level, expected_mass_ratio, measured_mass_ratio, error_pct, energy_retained_pct, stable_flag

**Acceptance**: >80% of particles have measured mass within 10% of l(l+1) prediction. All particles flagged "stable=True" retain >90% energy.

---

## Phase 2: Two-Particle Interactions (Do they attract/repel correctly?)

### Task 2.1: Electron-Electron Repulsion (EM Like-Charge)
**File**: `tests/validation/test_ee_repulsion.py`
**Status**: [x] DONE
**Depends on**: Task 1.1

**Setup**: `gravity_em()` config, two electrons (both phase=0) separated by 20 lattice units.
**Run**: 30,000 steps.
**Measure**:
- Force profile (attractive or repulsive?)
- Force vs separation — should show repulsion that weakens with distance
- Separation over time — should increase (repulsion)

**H₀**: Same-phase solitons attract or show no interaction.
**H₁**: Same-phase solitons repel via constructive interference increasing energy.

---

### Task 2.2: Electron-Positron Attraction (EM Opposite-Charge)
**File**: `tests/validation/test_ep_attraction.py`
**Status**: [x] DONE
**Depends on**: Task 2.1

**Setup**: `gravity_em()` config, electron (phase=0) + positron (phase=π) separated by 20 units.
**Measure**:
- Force profile — should be attractive
- Separation over time — should decrease (attraction)
- Energy in overlap region — destructive interference reduces |Ψ|²

**H₀**: Opposite-phase solitons repel or show no interaction.
**H₁**: Opposite-phase solitons attract via destructive interference lowering energy.

---

### Task 2.3: Electron-Positron Annihilation
**File**: `tests/validation/test_annihilation.py`
**Status**: [x] DONE
**Depends on**: Task 2.2

**Setup**: Same as 2.2 but start closer (separation ~8-10 units) so they merge.
**Measure**:
- Do they form a bound state (positronium)?
- Does the bound state decay (energy radiates outward)?
- Total energy conservation during process
- Radiation pattern — is energy released as outgoing waves?

**H₀**: Electron and positron merge into a stable blob.
**H₁**: They annihilate — localized energy converts to outgoing radiation.

---

### Task 2.4: Gravitational Attraction (Two Neutral Massive Particles)
**File**: `tests/validation/test_gravity_attraction.py`
**Status**: [x] DONE
**Depends on**: Task 1.5

**Setup**: `gravity_only()` config, two heavy real-field solitons (e.g., neutron-mass) separated by 20 units.
**Measure**:
- Separation over time — should decrease (gravity attracts)
- Force profile vs separation — should follow ~1/r² at large r
- Compare gravitational vs electromagnetic force strengths (gravity should be MUCH weaker)

**H₀**: Neutral solitons show no interaction.
**H₁**: Neutral solitons attract via shared χ-well deepening (gravity).

---

### Task 2.5: Coulomb Force Law Measurement (1/r² Test)
**File**: `tests/validation/test_coulomb_law.py`
**Status**: [x] DONE
**Depends on**: Task 2.1

**Setup**: `gravity_em()` config, two electrons at separations r = 10, 15, 20, 25, 30.
**Measure**: Force at each separation (from energy gradient or acceleration).
**Fit**: F ∝ r^(-n). 
**Expected**: n ≈ 2 in 3D (Coulomb law).

**H₀**: Force does not follow 1/r².
**H₁**: Force follows 1/r² (Coulomb law emerges from phase interference in 3D).

---

### Task 2.6: Charge Quantization Test
**File**: `tests/validation/test_charge_quantization.py`
**Status**: [x] DONE
**Depends on**: Task 2.1

**Setup**: `gravity_em()` config, particles with intermediate phases (θ = 0, π/4, π/2, 3π/4, π).
**Measure**: Force between each pair, phase stability over time.
**Expected**: 
- Only θ=0 and θ=π should be stable (integer charges)
- Intermediate phases should drift toward 0 or π
- This demonstrates charge quantization is EMERGENT

**H₀**: All phases are equally stable.
**H₁**: Only θ=0 and θ=π are dynamically stable (charge is quantized).

---

## Phase 3: Color Physics & Confinement (Do quarks confine?)

### Task 3.1: ~~Implement v17 Helmholtz Kernel for S_a~~ → MOVED to Task 0.0
**Status**: [x] DONE (moved to Task 0.0 as prerequisite)

---

### Task 3.2: Single Quark in Isolation
**File**: `tests/validation/test_single_quark.py`
**Status**: [x] DONE
**Depends on**: Task 3.1

**Setup**: `full_physics()` config, place single up quark (field_level=2, color="r" → only Ψ_0 excited).
**Measure**:
- f_c value — should be 2/3 (single color = maximum color variance)
- SCV value — should be large (S_a smoothing spreads color info, SCV > 0)
- χ well depth — should be DEEPER than equivalent colorless soliton (κ_c·f_c·2/3 adds to gravity)
- Stability — does the quark remain localized?

**H₀**: Quark behaves identically to a colorless soliton.
**H₁**: Quark has measurably deeper χ well and non-zero color variance.

---

### Task 3.3: Color Singlet (Three Equal Colors)
**File**: `tests/validation/test_color_singlet.py`
**Status**: [x] DONE
**Depends on**: Task 3.2

**Setup**: `full_physics()` config, place soliton with equal energy in all 3 colors (Ψ_0 = Ψ_1 = Ψ_2).
**Measure**:
- f_c value — should be 0 (balanced = singlet)
- SCV value — should be 0 (S_a equal for all colors → variance vanishes)
- χ well depth — should match gravity-only prediction (no color correction)

**H₀**: Singlet and single-color have same χ well.
**H₁**: Singlet has shallower well (no color variance contribution).

---

### Task 3.4: Quark-Antiquark Flux Tube (Meson Analog)
**File**: `tests/validation/test_flux_tube.py`
**Status**: [x] DONE
**Depends on**: Task 3.1, 3.2

**Setup**: `full_physics()` config, N=64, two colored solitons separated by 20 units:
- Soliton A: Ψ_0 excited (red quark)
- Soliton B: Ψ_1 excited (green antiquark — different color)

**Run**: 20,000 steps.
**Measure**:
- χ profile between the two quarks — is there a "tube" depression?
- SCV in the region between quarks — should be positive (color mismatch)
- Energy in tube region vs separation — should grow linearly (string tension)

**H₀**: No χ depression exists between separated quarks (tube integral = 0).
**H₁**: Linear χ depression forms between quarks (R² > 0.8 for tube energy vs separation).

Reference: Dynamic 3D test in `LFM_CONFINEMENT_MECHANISM.md` got R² = 0.882.

---

### Task 3.5: Flux Tube at Multiple Separations (String Tension Measurement)
**File**: `tests/validation/test_string_tension.py`
**Status**: [x] DONE
**Depends on**: Task 3.4

**Setup**: Same as 3.4 but at separations d = 10, 14, 18, 22, 26.
**Measure**: Tube energy at each separation.
**Fit**: E_tube = σ × d + constant.
**Expected**: R² > 0.85, σ > 0 (positive string tension).

**H₀**: Tube energy does not grow with separation (σ = 0).
**H₁**: Tube energy grows linearly (σ > 0, confinement demonstrated).

---

### Task 3.6: Three-Quark System (Baryon Analog)
**File**: `tests/validation/test_baryon.py`
**Status**: [x] DONE
**Depends on**: Task 3.4

**Setup**: `full_physics()` config, N=64, three colored solitons in a triangle:
- Soliton A: Ψ_0 (red)
- Soliton B: Ψ_1 (green)
- Soliton C: Ψ_2 (blue)

**Run**: 30,000 steps.
**Measure**:
- Do they form a bound state? (cluster near each other)
- What is f_c at the center? (should approach 0 if balanced = baryon → singlet)
- Energy of the bound state vs 3× isolated quark energy (binding energy)
- Compare stability to single quark — baryon should be MORE stable

**H₀**: Three quarks fly apart (no binding).
**H₁**: Three quarks bind into a stable cluster with f_c → 0 (color singlet).

---

### Task 3.7: Quark vs Hadron Stability Comparison
**File**: `tests/validation/test_quark_hadron_stability.py`
**Status**: [x] DONE
**Depends on**: Task 3.2, 3.3, 3.6

Run all three configurations (free quark, meson pair, baryon triplet).
**Measure**: f_c ordering and soliton survival.
**Expected ordering**: f_c(quark) > f_c(meson) > f_c(baryon)
(More balanced colors → lower f_c, consistent with QCD singlet ordering.)

**Note (Session 145)**: Original plan tested energy retention, but ψ-energy retention > 1.0 during
early χ→ψ transient transfer invalidates short-timescale retention comparisons. The v15 stress test
(Session 140b) used ε_cc over 3000 steps — at CI budget (500 steps) the transient dominates.
Rewritten to test f_c ordering (structural) and survival (localization) instead.

**H₀**: All configurations have identical f_c.
**H₁**: f_c ordering follows QCD: quark > meson > baryon.

---

## Phase 4: Emergent Behavior Validation (Does known physics emerge?)

### Task 4.1: Hydrogen Atom (Electron Orbiting Proton)
**File**: `tests/validation/test_hydrogen_atom.py`
**Status**: [ ] TODO
**Depends on**: Phase 2

**Setup**: `gravity_em()` config, N=128, place proton (phase=π, +charge) at center, electron (phase=0, -charge) at offset.
**Run**: 100,000 steps.
**Measure**:
- Does electron form a bound orbit/standing wave around proton?
- Binding energy — compare to known hydrogen ground state
- Angular momentum spectrum of the electron field — look for l=0 (s-orbital) dominance
- Radial probability density |Ψ(r)|² — should peak at Bohr-like radius

**H₀**: Electron disperses or collapses into proton.
**H₁**: Electron forms a stable bound state around proton.

Reference: `lfm_3d_atom.py` already has some code for this.

---

### Task 4.2: Hydrogen Energy Levels (Excited States)
**File**: `tests/validation/test_hydrogen_spectrum.py`
**Status**: [ ] TODO
**Depends on**: Task 4.1

**Setup**: Same as 4.1 but seed electron at different l values (l=0,1,2).
**Measure**:
- ω for each l-state — should show discrete energy levels
- Energy ratios — should approach E_n ∝ 1/n² pattern
- Transition: perturb l=1 state, measure radiation frequency — should be ω₂₁ = E₂ - E₁

**H₀**: All initial conditions give same energy.
**H₁**: Discrete energy levels emerge with E_n ∝ 1/n².

---

### Task 4.3: Pair Creation from Parametric Resonance
**File**: `tests/validation/test_pair_creation.py`
**Status**: [ ] TODO
**Depends on**: Phase 1

**Setup**: `gravity_em()` config, oscillating χ at frequency Ω = 2χ₀ = 38 (parametric resonance condition).
Start with E = tiny random noise (machine epsilon level).
**Run**: 50,000 steps.
**Measure**:
- Does |Ψ|² grow exponentially? (Mathieu instability)
- Do localized structures (particles) form from the amplified noise?
- Compare to static χ control (should NOT grow)

**H₀**: Noise stays at machine epsilon regardless of χ oscillation frequency.
**H₁**: |Ψ|² grows exponentially only at Ω ≈ 2χ₀ (parametric matter creation).

Reference: 256³ GPU sim showed 1.78×10²¹× growth at Ω=2χ₀.

---

### Task 4.4: Wave Equation Dispersion (Verify GOV-01 Gives Klein-Gordon)
**File**: `tests/validation/test_dispersion.py`
**Status**: [ ] TODO
**Depends on**: Phase 0

**Setup**: Any config, uniform χ = χ₀, inject monochromatic plane wave with known k.
**Run**: Enough steps to measure ω from phase velocity.
**Measure**: ω² = c²k² + χ₀² (Klein-Gordon dispersion).
**Do this for multiple k values and fit.**

**H₀**: ω² ≠ c²k² + χ₀².
**H₁**: ω² = c²k² + χ₀² (KG dispersion confirmed).

---

### Task 4.5: Gravitational Lensing
**File**: `tests/validation/test_lensing.py`
**Status**: [ ] TODO
**Depends on**: Phase 2

**Setup**: `gravity_em()` config, heavy soliton at center creating deep χ-well. Send wave packet past it off-axis.
**Measure**:
- Deflection angle of wave packet
- Compare to GR prediction: θ = 4GM/(c²b) where b = impact parameter

**H₀**: Wave packet travels in a straight line regardless of χ-well.
**H₁**: Wave packet deflects toward χ-well (gravitational lensing).

---

### Task 4.6: Gravitational Time Dilation
**File**: `tests/validation/test_time_dilation.py`
**Status**: [ ] TODO
**Depends on**: Phase 1

**Setup**: Two identical oscillating solitons — one near a massive object (deep χ-well), one far away (χ ≈ χ₀).
**Measure**: Oscillation frequency of each over time.
**Expected**: Soliton in deeper χ-well oscillates at different rate (time dilation).

ω(r) ~ χ(r), and χ is lower near mass → lower frequency → time runs slower near mass.

**H₀**: Both solitons oscillate at same frequency.
**H₁**: Soliton near mass oscillates at lower frequency (gravitational redshift).

---

### Task 4.7: Dark Matter Halo (χ Memory Test)
**File**: `tests/validation/test_dark_matter_halo.py`
**Status**: [ ] TODO
**Depends on**: Phase 2

**Setup**: `gravity_only()` config with GOV-03 (τ-averaging), place massive soliton, let it evolve, then REMOVE the soliton (set E=0).
**Measure**: 
- Does χ-well persist after matter is removed? (χ memory)
- How long does the reduced-χ region last? (controlled by τ)
- Does a test particle orbit the phantom well?

**H₀**: χ returns to χ₀ immediately when matter is removed.
**H₁**: χ-well persists for ~τ steps (dark matter halo analog).

---

## Phase 5: Multi-Particle Systems & Composite Validation

### Task 5.1: Proton as Three Quarks (Not Just a Single Soliton)
**File**: `tests/validation/test_proton_composite.py`
**Status**: [ ] TODO
**Depends on**: Phase 3

**Setup**: `full_physics()` config, N=128, place three quarks (u, u, d) with correct colors.
**Run**: 100,000 steps.
**Measure**:
- Do they bind into a single cluster?
- Mass of the cluster vs sum of quark masses (binding energy should be ~99% of proton mass — most mass is from binding, not quarks)
- f_c at cluster center → should approach 0 (color singlet)
- Compare cluster mass to single-soliton proton from Task 1.5

**H₀**: Three quarks don't bind (fly apart or merge into featureless blob).
**H₁**: Three quarks form bound state with mass >> sum of constituent quark masses.

---

### Task 5.2: Pion (Meson) from Quark + Antiquark
**File**: `tests/validation/test_pion_composite.py`
**Status**: [ ] TODO
**Depends on**: Phase 3

**Setup**: `full_physics()` config, up quark + down antiquark.
**Measure**:
- Binding, mass, stability
- Compare to single-soliton pion from catalog

---

### Task 5.3: Scattering: Electron + Electron → Electron + Electron
**File**: `tests/validation/test_ee_scattering.py`
**Status**: [ ] TODO
**Depends on**: Phase 2

**Setup**: `gravity_em()` config, two electrons moving toward each other.
**Measure**:
- Do they scatter (deflect) or pass through?
- Scattering angle vs impact parameter
- Energy conservation during scattering
- Compare to Rutherford/Mott scattering cross-section

---

### Task 5.4: Scattering: Electron + Positron → Radiation
**File**: `tests/validation/test_ep_scattering.py`  
**Status**: [ ] TODO
**Depends on**: Task 2.3

Controlled annihilation experiment:
- Vary collision energy
- Measure radiation output direction and energy
- Look for threshold behaviors

---

### Task 5.5: Nuclear Binding (Deuterium)
**File**: `tests/validation/test_deuterium.py`
**Status**: [ ] TODO
**Depends on**: Task 5.1

**Setup**: `full_physics()`, one proton (uud) + one neutron (udd) nearby.
**Measure**: Do they form a bound state (deuterium)? What's the binding energy?

---

## Phase 6: Systematic Validation Report

### Task 6.1: Generate Validation Report
**File**: `scripts/generate_validation_report.py`
**Status**: [ ] TODO
**Depends on**: All above

Run every experiment in Phases 1-5. Generate markdown report:

```
# LFM Particle Validation Report
## Date: YYYY-MM-DD
## Summary: X/Y experiments H₀ rejected
## 
## Per-experiment results table
## Per-particle mass comparison table
## Force law fits
## Confinement measurements
## Binding energy measurements
## Overall assessment
```

---

### Task 6.2: CI Integration
**File**: `tests/validation/conftest.py` + GitHub Actions update
**Status**: [ ] TODO  
**Depends on**: Task 6.1

Mark fast experiments (<30s) as CI-runnable. Slow experiments (>5min) marked `@pytest.mark.slow`.
Add `make validate-particles` target.

---

## Dependency Graph

```
Phase 0 (Infrastructure)
  ├── 0.1 Config Presets
  ├── 0.2 Measurement Toolkit
  └── 0.3 Experiment Runner ← 0.1 + 0.2

Phase 1 (Single Particle) ← Phase 0
  ├── 1.1 Electron
  ├── 1.2 Positron ← 1.1
  ├── 1.3 Muon ← 1.1
  ├── 1.4 Neutrino ← 1.1
  ├── 1.5 Proton
  ├── 1.6 Neutron ← 1.5
  └── 1.7 Full Sweep ← 1.1-1.6

Phase 2 (Two-Particle) ← Phase 1
  ├── 2.1 e-e Repulsion ← 1.1
  ├── 2.2 e-p+ Attraction ← 2.1
  ├── 2.3 Annihilation ← 2.2
  ├── 2.4 Gravity ← 1.5
  ├── 2.5 Coulomb 1/r² ← 2.1
  └── 2.6 Charge Quantization ← 2.1

Phase 3 (Color/Confinement) ← Phase 0
  ├── 3.1 Helmholtz Kernel ← Backend
  ├── 3.2 Single Quark ← 3.1
  ├── 3.3 Color Singlet ← 3.2
  ├── 3.4 Flux Tube ← 3.1, 3.2
  ├── 3.5 String Tension ← 3.4
  ├── 3.6 Baryon ← 3.4
  └── 3.7 Stability Ordering ← 3.2, 3.3, 3.6

Phase 4 (Emergent Physics) ← Phases 1-2
  ├── 4.1 Hydrogen Atom ← Phase 2
  ├── 4.2 Energy Levels ← 4.1
  ├── 4.3 Pair Creation ← Phase 1
  ├── 4.4 Dispersion ← Phase 0
  ├── 4.5 Lensing ← Phase 2
  ├── 4.6 Time Dilation ← Phase 1
  └── 4.7 Dark Matter ← Phase 2

Phase 5 (Composites) ← Phases 2-3
  ├── 5.1 Proton from Quarks ← Phase 3
  ├── 5.2 Pion from Quarks ← Phase 3
  ├── 5.3 e-e Scattering ← Phase 2
  ├── 5.4 e-p+ Scattering ← 2.3
  └── 5.5 Deuterium ← 5.1

Phase 6 (Report) ← All
  ├── 6.1 Validation Report
  └── 6.2 CI Integration
```

---

## Guiding Principles (NEVER VIOLATE)

1. **GOV-01 + GOV-02 ONLY** — No Coulomb, no Newton, no QED, no external potentials
2. **Set conditions, measure outcomes** — We don't program particles to spin; we measure if spin-like behavior emerges
3. **Failures are data** — If an experiment fails, that IS the scientific result. Do not inject physics to make it pass.
4. **LFM-ONLY constraint verification** — Every experiment must verify no external physics was used
5. **Hypothesis framework** — Every experiment has H₀, H₁, explicit success criteria
6. **Use the library** — All experiments use `lfm.Simulation`, `lfm.SimulationConfig`, measurement toolkit
7. **Reproducible** — Fixed random seeds, deterministic evolution, version-pinned dependencies
