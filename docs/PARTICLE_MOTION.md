# LFM Particle Motion: How Solitons Move on the Lattice

## Overview

Particles in LFM are **standing waves trapped in self-consistent χ-wells**.
Making them move is non-trivial because the wave equation (GOV-01) couples
to the χ field (GOV-02), and the χ-well acts as a potential trap that resists
displacement.

This document describes the mechanism that **makes particles move correctly**
and is implemented in `Simulation.place_particle()`.

---

## The Three-Step Motion Mechanism

### Step 1: Eigenmode Relaxation (`relax_eigenmode`)

A bare Gaussian blob is NOT a particle — it radiates and dissolves.  A true
particle is a **self-consistent bound state**: a standing wave Ψ sitting
inside a χ-well that it creates through GOV-02.

The Poisson-relaxation hybrid solver alternates:
1. **Poisson solve**: ∇²χ = (κ/c²)|Ψ|² → self-consistent χ from current Ψ
2. **Imaginary-time relax**: evolve Ψ in frozen χ → lowest eigenmode

After ~5-20 cycles, this converges to a stationary eigenmode with frequency ω
and a well-defined χ-well depth.

### Step 2: Phase-Gradient Boost (`boost_fields`)

A stationary eigenmode Ψ₀(x) is converted to a moving soliton by encoding
momentum as a spatial phase gradient:

```
Ψ(x) = Ψ₀(x) · exp(ik·x)
```

where `k = γ·ω·v/c²` is the wavevector encoding the desired velocity.

This also requires:
- **Shifted prev buffer**: Ψ(t = -Δt) has its envelope physically shifted
  backward by v·Δt to initialise the correct leapfrog velocity
- **Boosted frequency**: ω' = γ·ω (relativistic time dilation)

### Step 3: Poisson χ Equilibration (`equilibrate`)

After one or more particles are placed, the combined |Ψ|² field is used to
Poisson-solve a fresh χ field:

```
∇²χ = (κ/c²)(|Ψ|² − E₀²)
```

For moving particles, χ_prev is also computed with an advective correction:
dχ/dt = −v·∇χ, so χ(t = -Δt) ≈ χ(t=0) + v·Δt·∇χ.  This ensures consistent
leapfrog initialisation.

---

## Critical Parameters for Motion

| Parameter | Stationary | Moving | Why |
|-----------|-----------|--------|-----|
| **Amplitude** | Catalog default (8-14) | **3.0** | Deep wells (amp ≥ 8) completely pin the soliton |
| **Sigma** | Catalog default (≥ 3.0) | **≥ 5.0** | Narrow solitons (σ < 5) suffer Peierls-Nabarro lattice pinning |
| **Well depth** | Deep (OK) | **< 0.12** | Depth = (χ₀ − χ_min)/χ₀; deeper = more pinned |
| **dt** | 0.02 (default) | 0.005 | Smaller dt resolves the boosted phase gradient better |

### Velocity-Depth Empirical Law

```
depth < 0.10  →  ~75-80% of requested velocity retained
depth  0.10-0.20  →  50-70% velocity (degraded)
depth > 0.23  →  PINNED (<10% velocity)
```

At amp=3.0, sig=5.0: depth ≈ 0.097 → **75% velocity** ✓

---

## The Sign Bug (Historical)

In `boost_fields()`, the envelope shift for the prev buffer uses
`scipy.ndimage.shift()`.  The correct call is:

```python
shift_vec = (-vx * dt, -vy * dt, -vz * dt)  # MINUS sign
```

The minus sign means: "at time t = -Δt, the envelope was Δt earlier,
so it was shifted BACKWARD relative to its current position."

Using `+vx*dt` (wrong sign) causes the soliton to decelerate instead of
maintaining velocity.

---

## User API

Users should **never** need to understand the above mechanism.  The
`Simulation.place_particle()` method handles everything:

```python
from lfm import Simulation, SimulationConfig, FieldLevel

config = SimulationConfig(grid_size=64, field_level=FieldLevel.COMPLEX)
sim = Simulation(config)

# Place particles — eigenmode + boost + charge phase all handled automatically
sim.place_particle("proton",     position=(32, 32, 24), velocity=(0, 0, 0.1))
sim.place_particle("antiproton", position=(32, 32, 40), velocity=(0, 0, -0.1))

# Run — auto-equilibrates χ before first step
sim.run(steps=10_000)
```

Higher-level factory functions also work:

```python
from lfm import create_particle, create_two_particles, create_collision

# Single particle
placed = create_particle("electron", velocity=(0.05, 0, 0))
placed.sim.run(5000)

# Two-body system
pa, pb = create_two_particles("up_quark", "up_quark", separation=12)
pa.sim.run(2000)

# Head-on collision
setup = create_collision("proton", "antiproton", speed=0.1, N=128)
setup.sim.run(10_000)
```

---

## What Happens Internally

When you call `sim.place_particle("proton", position=(32,32,24), velocity=(0,0,0.1))`:

1. **Catalog lookup**: finds Proton (mass_ratio=1836.15, phase=0, field_level=1)
2. **Auto-selects amp=3.0, sigma=5.0** (shallow well for motion)
3. **Relaxes eigenmode** at grid centre → converged psi_r, chi, ω
4. **Rolls** eigenmode from centre to (32,32,24) via `np.roll`
5. **Boosts** with k = γ·ω·0.1/c² phase gradient  
6. **Applies charge phase** (proton: θ=0 → no rotation)
7. **Superposes** onto existing Ψ buffers (supports multiple particles)
8. **Records** velocity boost for χ equilibration

When you call `sim.run(N)`:
1. **Auto-equilibrates**: Poisson-solves χ from combined |Ψ|², applies advective corrections
2. **Evolves**: leapfrog steps GOV-01 + GOV-02 on GPU

---

## Verified Results

### N=64 Proton-Antiproton Collision (Peak Tracking)

| Step | Time | Peak A (z) | Peak B (z) | Separation |
|------|------|-----------|-----------|------------|
| 0 | 0.0 | 23 | 41 | 18 |
| 2850 | 14.25 | 25 | 39 | 14 |
| 4750 | 23.75 | 27 | 37 | 10 |
| 6650 | 33.25 | 29 | 35 | 6 |
| 8550 | 42.75 | ~32 (merged) | — | 0 |

Particles approach at ~0.18c (faster than 0.1c boost due to mutual
gravitational attraction), merge at grid centre, and annihilate
(56.1% energy radiated as waves).

### Two-Soliton Elastic Scattering

Same-charge solitons: separation 12 → 5.05 (closest approach) → 13.2
(bounced apart). Energy conserved: 188.0 → 188.2 (0.1% drift).
