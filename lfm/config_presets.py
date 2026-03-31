"""
Config Presets
==============

Named presets that turn on the right physics terms for each field level.

Usage::

    from lfm.config_presets import gravity_only, gravity_em, full_physics

    config = full_physics(grid_size=64)
    sim = Simulation(config)
"""

from __future__ import annotations

from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.constants import (
    EPSILON_CC,
    KAPPA_C,
    KAPPA_STRING,
    KAPPA_TUBE,
    LAMBDA_H,
    SA_D,
    SA_GAMMA,
)


def gravity_only(grid_size: int = 64, **overrides: object) -> SimulationConfig:
    """Level 0: Real E, gravity only.

    For: cosmology, dark matter, cosmic web, neutral massive particles.
    """
    params: dict[str, object] = dict(
        grid_size=grid_size,
        field_level=FieldLevel.REAL,
        boundary_type=BoundaryType.FROZEN,
    )
    params.update(overrides)
    return SimulationConfig(**params)  # type: ignore[arg-type]


def gravity_em(grid_size: int = 64, **overrides: object) -> SimulationConfig:
    """Level 1: Complex Ψ, gravity + EM.

    For: charged leptons, photons, hydrogen atom, Coulomb force.
    Mexican hat enabled for vacuum stability.
    """
    params: dict[str, object] = dict(
        grid_size=grid_size,
        field_level=FieldLevel.COMPLEX,
        boundary_type=BoundaryType.FROZEN,
        lambda_self=LAMBDA_H,
    )
    params.update(overrides)
    return SimulationConfig(**params)  # type: ignore[arg-type]


def full_physics(grid_size: int = 64, **overrides: object) -> SimulationConfig:
    """Level 2: 3-color Ψₐ, all four forces.

    For: quarks, hadrons, confinement, strong interactions.
    Enables Mexican hat, color variance, cross-color coupling,
    CCV, SCV flux tubes, and S_a Helmholtz smoothing.
    """
    params: dict[str, object] = dict(
        grid_size=grid_size,
        field_level=FieldLevel.COLOR,
        boundary_type=BoundaryType.FROZEN,
        lambda_self=LAMBDA_H,
        kappa_c=KAPPA_C,
        epsilon_cc=EPSILON_CC,
        kappa_string=KAPPA_STRING,
        kappa_tube=KAPPA_TUBE,
        sa_gamma=SA_GAMMA,
        sa_d=SA_D,
    )
    params.update(overrides)
    return SimulationConfig(**params)  # type: ignore[arg-type]


def spinor_field(grid_size: int = 64, **overrides: object) -> SimulationConfig:
    """Spin-1/2 spinor: two-component complex field (FieldLevel.COLOR, n_colors=2).

    Array layout returned by Evolver.get_psi_real() / get_psi_imag()::

        psi_r[0], psi_i[0]  ->  Re(ψ_↑), Im(ψ_↑)  (spin-up)
        psi_r[1], psi_i[1]  ->  Re(ψ_↓), Im(ψ_↓)  (spin-down)

    GOV-02 coupling: χ sourced by |ψ_↑|² + |ψ_↓|² (spin-blind gravity).
    Use :mod:`lfm.fields.spinor` for initial conditions and
    :mod:`lfm.analysis.spinor` for measurements.

    The key prediction: 720° rotational periodicity (spinor sign flip at 360°)
    is measurable via interference — see
    :func:`lfm.analysis.spinor.spinor_interference_energy`.

    For: spin-1/2 dynamics, Stern-Gerlach, neutron interferometry analogs,
         spinor chi-well formation.
    """
    params: dict[str, object] = dict(
        grid_size=grid_size,
        field_level=FieldLevel.COLOR,
        n_colors=2,
        boundary_type=BoundaryType.FROZEN,
        lambda_self=LAMBDA_H,
    )
    params.update(overrides)
    return SimulationConfig(**params)  # type: ignore[arg-type]
