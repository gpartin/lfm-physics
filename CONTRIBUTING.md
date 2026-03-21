# Contributing to lfm-physics

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/gpartin/lfm-physics.git
cd lfm-physics
pip install -e ".[dev]"
pre-commit install
```

Run the full local quality gate before pushing:

```bash
pre-commit run
```

## Running Tests

```bash
pytest                         # full suite
pytest tests/test_config.py    # single module
pytest -x                      # stop on first failure
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
pre-commit run              # preferred: runs configured auto-fixes on staged files
pre-commit run --files path/to/file.py
ruff check lfm/ tests/
ruff format lfm/ tests/     # auto-format
```

- Line length: 100 characters
- Target: Python 3.10+
- All public functions need type hints and docstrings

## Pull Requests

1. Fork the repo and create a feature branch
2. Add tests for any new functionality
3. Ensure `pre-commit run` and `pytest` pass
4. Open a PR against `main`

## Physics Guidelines

- **GOV-01 and GOV-02 are the only governing equations** — never inject external physics
- All constants must derive from `CHI0 = 19` (see `lfm/constants.py`)
- New predictions go in `lfm/formulas/predictions.py` with measured values and error percentages
- Use the 19-point stencil for Laplacians (not 7-point) unless benchmarking

## Project Structure

```
lfm/
  constants.py       # Fundamental constants from chi_0 = 19
  config.py          # SimulationConfig dataclass
  simulation.py      # High-level Simulation facade
  core/
    backends/        # CPU (NumPy) and GPU (CuPy) implementations
    evolver.py       # Double-buffered evolution loop
    integrator.py    # Leapfrog GOV-01 + GOV-02
    stencils.py      # 19-point isotropic Laplacian
  fields/            # Soliton placement, equilibration, noise
  formulas/          # Analytic predictions and mass tables
  analysis/          # Energy, structure detection, metrics
examples/            # Runnable demo scripts
tests/               # pytest test suite
```

## Reporting Issues

Open a [GitHub issue](https://github.com/gpartin/lfm-physics/issues) with:
- What you expected vs what happened
- Minimal reproduction code
- Python version and OS
