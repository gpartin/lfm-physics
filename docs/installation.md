# Installation

## Requirements

- Python 3.10 or newer
- NumPy ≥ 1.24
- SciPy ≥ 1.10

## Install from PyPI

```bash
pip install lfm-physics
```

## Optional extras

**GPU acceleration** (requires a CUDA-capable GPU and CUDA 12):

```bash
pip install "lfm-physics[gpu]"
```

**Visualisation** (matplotlib):

```bash
pip install "lfm-physics[viz]"
```

**Everything**:

```bash
pip install "lfm-physics[all]"
```

**Documentation** (to build these docs locally):

```bash
pip install "lfm-physics[docs]"
```

## Install from source

```bash
git clone https://github.com/gpartin/lfm-physics.git
cd lfm-physics
pip install -e ".[all,dev]"
```

## Verify the install

```python
import lfm
print(lfm.__version__)
print(f"χ₀ = {lfm.CHI0}")  # should print 19
print(f"GPU available: {lfm.gpu_available()}")
```

## Backend detection

The library auto-detects GPU availability at import time:

- **GPU**: CuPy backend is used if `cupy-cuda12x` is installed and a CUDA device
  is present.  All arrays are kept on the GPU; no copies are made unless you
  access field properties directly.
- **CPU**: NumPy backend is used as a transparent fallback.  All simulation
  results are identical; the CPU path is simply slower for large grids.

You can force a specific backend:

```python
sim = lfm.Simulation(config, backend="cpu")   # always CPU
sim = lfm.Simulation(config, backend="gpu")   # raise if no GPU
sim = lfm.Simulation(config, backend="auto")  # default
```
