# Quick-start guide

This page walks you through the core workflow in five minutes.

## 1. The two governing equations

Every simulation runs exactly two coupled wave equations:

**GOV-01** (wave field):

$$\frac{\partial^2 \Psi}{\partial t^2} = c^2 \nabla^2 \Psi - \chi^2 \Psi$$

**GOV-02** (χ field):

$$\frac{\partial^2 \chi}{\partial t^2} = c^2 \nabla^2 \chi - \kappa\!\left(|\Psi|^2 - E_0^2\right)$$

- $\Psi$ — the wave field (real, complex, or 3-component colour)
- $\chi$ — local stiffness of the lattice; $\chi_0 = 19$ in vacuum
- $\kappa = 1/63$ — coupling constant (derived from 3D lattice geometry)

Everything else — gravity, dark matter, electromagnetism, structure
formation — emerges from these two equations alone.

## 2. Minimal example

```python
import lfm

# --- 1. Create a simulation ---
config = lfm.SimulationConfig(grid_size=48)
sim = lfm.Simulation(config)

# --- 2. Place a particle ---
sim.place_soliton((24, 24, 24), amplitude=5.0, sigma=4.0)

# --- 3. Let χ adjust (GOV-02 quasi-static limit) ---
sim.equilibrate()

# --- 4. Evolve ---
sim.run(steps=2000)

# --- 5. Inspect ---
m = sim.metrics()
print(f"χ_min  = {m['chi_min']:.2f}")   # below 19 = gravity well
print(f"wells  = {m['well_fraction']*100:.1f}%")
print(f"energy = {m['energy_total']:.4f}")
```

## 3. Field levels

Choose the right level for your physics:

| Level | `FieldLevel` | Physics |
|-------|-------------|---------|
| Real (Ψ ∈ ℝ) | `REAL` | Gravity, dark matter |
| Complex (Ψ ∈ ℂ) | `COMPLEX` | + Electromagnetism (charge = phase) |
| Colour (Ψ ∈ ℂ³) | `COLOR` | + Strong / weak force |

```python
config = lfm.SimulationConfig(
    grid_size=48,
    field_level=lfm.FieldLevel.COMPLEX,   # enable EM
)
```

## 4. Unit mapping

### Cosmological scales

```python
from lfm import CosmicScale

scale = CosmicScale(box_mpc=8461.0, grid_size=256)
print(scale.cell_to_mpc())        # Mpc per cell
print(scale.step_to_gyr(541_000)) # → 13.8 Gyr (present day)
```

### Planck scales

```python
from lfm import PlanckScale

# Default: 1 cell ≈ 33 Mpc (observable-universe calibration)
ps = PlanckScale(grid_size=256)

# Planck resolution: 1 cell = 1 Planck length (l_P ≈ 1.6 × 10⁻³⁵ m)
ps_p = PlanckScale.at_planck_resolution(grid_size=256)
print(ps_p.is_planck_resolution)   # True
print(ps_p.cell_size_m)            # 1.616e-35 m
```

## 5. Checkpointing

Save and resume long runs:

```python
from lfm.io import save_checkpoint, load_checkpoint

# --- run 10 000 steps, save ---
sim.run(steps=10_000)
save_checkpoint(sim, "checkpoints/step_10k.npz")

# --- later: restore and continue ---
sim2 = load_checkpoint("checkpoints/step_10k.npz")
sim2.run(steps=5_000)
```

## 6. Next steps

- Walk through the [examples](examples.md) for deeper physics demonstrations.
- Browse the [API reference](api/simulation.rst) for the full `Simulation` interface.
