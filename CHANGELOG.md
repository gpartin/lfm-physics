# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Initial package scaffolding
- `lfm.constants` — All LFM fundamental constants derived from χ₀ = 19
- `lfm.config` — `SimulationConfig` dataclass with validation
- `lfm.core.stencils` — 7-point and 19-point isotropic Laplacian operators
- `lfm.core.integrator` — Leapfrog time integration (GOV-01 + GOV-02)
- CI/CD via GitHub Actions (test on push, publish on tag)
