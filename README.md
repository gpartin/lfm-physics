# lfm-physics

**Lattice Field Medium** — a physics simulation library implementing the LFM framework.

Two governing equations. One integer (χ₀ = 19). All of physics.

## Install

```bash
pip install lfm-physics
```

For GPU acceleration (NVIDIA):
```bash
pip install lfm-physics[gpu]
```

## Quick Start

```python
import lfm

# Check fundamental constants
print(f"χ₀ = {lfm.CHI0}")       # 19.0
print(f"κ  = {lfm.KAPPA}")      # 0.015873...
print(f"α  = {lfm.ALPHA_EM}")   # 0.007299... ≈ 1/137.088

# Create a simulation
config = lfm.SimulationConfig(grid_size=64, dt=0.02)
sim = lfm.Simulation(config)

# Initialize with a Gaussian soliton
from lfm.fields import gaussian_soliton
sim.add_soliton(gaussian_soliton(center=(32, 32, 32), amplitude=6.0, sigma=3.0))

# Evolve
sim.run(steps=1000)

# Analyze
from lfm.analysis import compute_metrics
metrics = compute_metrics(sim.state)
print(f"χ_min = {metrics['chi_min']:.2f}")
```

## What is LFM?

The Lattice Field Medium framework derives all physics from two coupled wave equations
on a discrete 3D cubic lattice:

- **GOV-01** (Wave Equation): `∂²Ψ/∂t² = c²∇²Ψ − χ²Ψ`
- **GOV-02** (Field Equation): `∂²χ/∂t² = c²∇²χ − κ(|Ψ|² − E₀²) − 4λ_H·χ(χ² − χ₀²)`

From these two equations and χ₀ = 19 (derived from 3D lattice geometry: 1+6+12=19),
the framework predicts 41+ fundamental constants within 2% of measured values.

## Documentation

- [API Reference](https://github.com/gpartin/lfm-physics/wiki)
- [LFM Physics Papers](https://zenodo.org/communities/lfm-physics)
- [Constants & Predictions](https://github.com/gpartin/lfm-physics/blob/main/lfm/constants.py)

## License

MIT
