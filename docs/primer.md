# LFM in Five Minutes

A minimal introduction to the physics behind the `lfm-physics` library.

---

## One Sentence

> The universe is a cubic lattice where every point stores two numbers —
> a wave amplitude **Ψ** and a local stiffness **χ** — and they evolve
> by two coupled wave equations.

---

## The Two Fields

| Symbol | Name | Physical meaning |
|--------|------|------------------|
| **Ψ**  | Wave field | Energy / matter at each lattice point |
| **χ**  | Chi field  | Stiffness of the lattice at each point |

Empty space has χ = 19 everywhere.  Where energy concentrates, χ drops
below 19, forming a potential well — what we call *gravity*.

---

## The Two Equations

**GOV-01 — Wave Equation** (how Ψ evolves):

```
Ψⁿ⁺¹ = 2Ψⁿ − Ψⁿ⁻¹ + Δt²[c²∇²Ψⁿ − (χⁿ)²Ψⁿ]
```

Energy propagates through the lattice.  The `χ²Ψ` term means waves
oscillate faster where χ is high (empty space) and slower where χ is low
(near matter).  This is how gravity bends light.

**GOV-02 — Chi Equation** (how χ evolves):

```
χⁿ⁺¹ = 2χⁿ − χⁿ⁻¹ + Δt²[c²∇²χⁿ − κ(|Ψⁿ|² − E₀²)]
```

Energy density |Ψ|² pushes χ down — matter curves spacetime.  The
coupling κ = 1/63 is derived from the lattice geometry.

**That's it.**  These two update rules, applied at every lattice point
every timestep, produce gravity, waves, dark matter, expansion, and more.

---

## Why 19?

The number 19 comes from counting non-propagating modes on a 3D cubic
lattice:

| Mode type | Count | k-vectors |
|-----------|-------|-----------|
| Centre    | 1     | (0,0,0) |
| Faces     | 6     | (±1,0,0), (0,±1,0), (0,0,±1) |
| Edges     | 12    | (±1,±1,0), etc. |
| **Total** | **19** | **= χ₀** |

The remaining 8 corner modes (±1,±1,±1) are propagating — identifying
them with gluons gives N_gluons = 8.

From χ₀ = 19 alone, LFM derives 40+ physical constants to high accuracy.

---

## What Emerges

| Phenomenon | How it arises |
|------------|---------------|
| **Gravity** | Energy (|Ψ|²) dips χ → potential well → attraction |
| **Dark matter** | χ wells persist after matter moves away (memory) |
| **Electromagnetism** | Phase of complex Ψ = charge; interference = Coulomb |
| **Expansion** | Voids evacuate → χ rises → photons slow down |
| **Particles** | Standing waves trapped in self-consistent χ wells |
| **Atoms** | Nuclear χ well + bound electron eigenmodes |

---

## Quick Start

```python
import lfm

# Create a 64-cell cubic universe
config = lfm.SimulationConfig(grid_size=64)
sim = lfm.Simulation(config)

# Drop a soliton — a localized energy concentration
sim.place_soliton((32, 32, 32), amplitude=6.0)

# Evolve for 1000 timesteps
sim.run(steps=1000)

# Look at what happened
print(f"χ_min = {sim.chi.min():.2f}")  # Should be < 19 (gravity!)
```

See the [examples/](../examples/) directory for 14 runnable demos covering
gravity, electromagnetism, atoms, orbits, cosmology, and more.

---

## Going Deeper

| Topic | Resource |
|-------|----------|
| Full API reference | `help(lfm.Simulation)` |
| All 14 examples | [examples/](../examples/) |
| Visualization | `from lfm.viz import plot_slice` |
| Parameter sweeps | `from lfm import sweep` |
| Common errors | [troubleshooting.md](troubleshooting.md) |
