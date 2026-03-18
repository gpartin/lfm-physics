# lfm-physics

[![Tests](https://github.com/gpartin/lfm-physics/actions/workflows/test.yml/badge.svg)](https://github.com/gpartin/lfm-physics/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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

# Fundamental constants derived from χ₀ = 19
print(f"χ₀ = {lfm.CHI0}")       # 19.0
print(f"κ  = {lfm.KAPPA}")      # 0.015873...
print(f"α  = {lfm.ALPHA_EM}")   # 0.007299... ≈ 1/137.088

# Create and run a simulation
config = lfm.SimulationConfig(grid_size=64, report_interval=500)
sim = lfm.Simulation(config)
sim.place_soliton((32, 32, 32), amplitude=6.0, sigma=5.0)
sim.equilibrate()
sim.run(steps=1000)

# Analyze
m = sim.metrics()
print(f"χ_min = {m['chi_min']:.2f}")
print(f"Wells = {m['well_fraction']*100:.1f}%")

# 41+ predictions from one integer
catalog = lfm.predict_all()
for name, entry in list(catalog.items())[:5]:
    print(f"  {name}: predicted={entry['predicted']:.6f}, "
          f"measured={entry['measured']:.6f}, error={entry['error_pct']:.2f}%")
```

## Examples

See [`examples/`](examples/) for full working scripts:

- **[cosmic_structure_formation.py](examples/cosmic_structure_formation.py)** —
  256³ universe simulation showing spontaneous gravitational structure
  (wells + voids) from GOV-01 + GOV-02 alone. Includes cosmic time
  conversion, periodic snapshots, and milestone tracking.
  Set `GRID_SIZE = 64` for a quick CPU test.

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
- [Contributing](CONTRIBUTING.md)

## License

MIT
