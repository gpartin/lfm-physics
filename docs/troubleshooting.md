# Troubleshooting

Common problems and how to fix them.

---

## My simulation gives NaN

**Cause:** The wave amplitude is too large for the grid, causing the
leapfrog integrator to diverge.

**Fix:**
```python
# Lower the amplitude
sim.place_soliton((32, 32, 32), amplitude=4.0)   # instead of 12.0

# Or use a larger grid (more room to spread energy)
config = lfm.SimulationConfig(grid_size=128)      # instead of 32
```

**Rule of thumb:** Keep amplitude below ~8 for `grid_size=32`, below ~12
for `grid_size=64`.  When in doubt, start small and increase.

---

## Energy diverges over time

**Cause:** The timestep `dt` exceeds the CFL stability limit.

**Fix:** Use the default `dt=0.02` — it satisfies the CFL condition for
all standard setups.  If you've changed `dt` manually:

```python
# Safe: dt = 0.02 (default, well inside CFL)
config = lfm.SimulationConfig(dt=0.02)

# The CFL limit for the 19-point stencil with χ₀=19:
#   dt < 1/sqrt(16c²/(3·Δx²) + χ₀²) ≈ 0.104  (for Δx=c=1)
# Our default 0.02 gives ~5× safety margin.
```

---

## χ goes negative

**This is not a bug.** In extreme-gravity scenarios (very high amplitude
or dense clusters), χ can dip below zero.  This is the LFM equivalent of
a black hole interior.

**If you want to prevent it:** Enable the Mexican-hat self-interaction,
which creates a stable second vacuum at −χ₀:

```python
config = lfm.SimulationConfig(
    lambda_self=lfm.LAMBDA_H,  # 4/31 ≈ 0.129
)
```

**If you don't care about extreme gravity:** Use gravity-only mode
(default `lambda_self=0.0`) and lower the amplitude.

---

## Everything collapses into one blob

**Cause:** With periodic boundaries and enough energy, gravity always
wins — everything falls toward the centre of mass.  This is correct
physics for a closed universe.

**Fix:** Use frozen boundaries (the default) to simulate an open region
embedded in the vacuum:

```python
config = lfm.SimulationConfig(
    boundary_type=lfm.BoundaryType.FROZEN,  # default
)
```

Frozen boundaries hold χ = 19 at the edges, representing infinite empty
space beyond the simulation box.

---

## Simulation is very slow

**Check 1 — Are you using GPU?**
```python
print(lfm.gpu_available())  # True if CuPy + CUDA detected

# Force GPU
sim = lfm.Simulation(config, backend="gpu")
```

Install GPU support:
```bash
pip install "lfm-physics[gpu]"
```

**Check 2 — Is your grid too large for CPU?**

| Grid size | Cells     | CPU time/step | GPU time/step |
|-----------|-----------|---------------|---------------|
| 32³       | 32 K      | ~0.3 ms       | ~0.02 ms      |
| 64³       | 262 K     | ~3 ms         | ~0.05 ms      |
| 128³      | 2.1 M     | ~30 ms        | ~0.3 ms       |
| 256³      | 16.8 M    | ~300 ms       | ~7 ms         |

For grid_size ≥ 128 on CPU, expect minutes for long runs.  GPU gives
50–200× speedup.

---

## ImportError: matplotlib not found

The visualization module (`lfm.viz`) requires matplotlib:

```bash
pip install "lfm-physics[viz]"
```

Or install everything:
```bash
pip install "lfm-physics[all]"
```

---

## How do I save and resume a run?

```python
# Save
sim.save_checkpoint("my_run.npz")

# Resume later
sim2 = lfm.Simulation.load_checkpoint("my_run.npz")
sim2.run(steps=5000)  # continues where it left off
```

Checkpoints preserve the full simulation state: fields, config, step
count, and metric history.

---

## Which field level should I use?

| I want to simulate...             | Field level           |
|-----------------------------------|-----------------------|
| Gravity, dark matter, cosmology   | `FieldLevel.REAL`     |
| Electromagnetism, charged particles | `FieldLevel.COMPLEX` |
| Strong force, color confinement   | `FieldLevel.COLOR`    |

```python
config = lfm.SimulationConfig(
    field_level=lfm.FieldLevel.COMPLEX,  # for EM
)
```

Start with REAL (simplest, fastest).  Upgrade only when your physics
requires it.
