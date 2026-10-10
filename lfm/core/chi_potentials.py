"""Local chi-potential force laws for GOV-02 ablation experiments."""

from __future__ import annotations

import numpy as np

from lfm.config import ChiPotentialModel


def dimensionless_potential_derivative(
    y: np.ndarray,
    model: ChiPotentialModel,
    source_ratio: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return df/dy for V=lambda_h*chi0^4*f(y)."""

    y = np.asarray(y, dtype=np.float64)
    model = ChiPotentialModel(model)
    if model == ChiPotentialModel.CANONICAL_QUARTIC:
        return 2.0 * y
    if model in (
        ChiPotentialModel.FLAT_OCTIC,
        ChiPotentialModel.NONLINEAR_GRADIENT,
        ChiPotentialModel.VARIABLE_INERTIA,
    ):
        return 4.0 * y**3
    if model == ChiPotentialModel.FLAT_DODECIC:
        return 6.0 * y**5
    if model == ChiPotentialModel.FLAT_POWER_8:
        return 8.0 * y**7
    if model == ChiPotentialModel.FLAT_POWER_10:
        return 10.0 * y**9
    if model == ChiPotentialModel.FLAT_POWER_12:
        return 12.0 * y**11
    if model == ChiPotentialModel.SMOOTH_EXPONENTIAL:
        exp_term = np.exp(-(y**2))
        return 2.0 * y * (1.0 - exp_term + y**2 * exp_term)
    if model == ChiPotentialModel.RATIONAL_CROSSOVER:
        return 2.0 * y**3 * (2.0 + y**2) / (1.0 + y**2) ** 2
    if model == ChiPotentialModel.HYPERBOLIC_CROSSOVER:
        tanh_y = np.tanh(y)
        sech_sq = 1.0 - tanh_y**2
        return 2.0 * y * tanh_y * (tanh_y + y * sech_sq)
    if model == ChiPotentialModel.AMPLITUDE_STRENGTHENED:
        return 4.0 * y**3 + 6.0 * y**5
    if model == ChiPotentialModel.SOURCE_DEPENDENT:
        return 4.0 * y**3 + 2.0 * np.asarray(source_ratio) * y
    if model == ChiPotentialModel.RADICAL_CROSSOVER:
        return 4.0 * y**7 / np.sqrt(1.0 + y**8)
    raise ValueError(f"unsupported chi potential model: {model}")


def potential_force(
    chi: np.ndarray,
    model: ChiPotentialModel,
    *,
    chi0: float,
    lambda_h: float,
    source_density: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return -dV/dchi for a frozen local potential candidate."""

    chi_array = np.asarray(chi, dtype=np.float64)
    y = (chi_array**2 - chi0**2) / chi0**2
    source_ratio = np.asarray(source_density) / chi0**2
    derivative = dimensionless_potential_derivative(
        y,
        model,
        source_ratio,
    )
    return -2.0 * lambda_h * chi0**2 * chi_array * derivative


def variable_inertia(
    chi: np.ndarray,
    *,
    chi0: float,
) -> np.ndarray:
    """Return the K-family kinetic multiplier M(chi)=1+y^2."""

    chi_array = np.asarray(chi, dtype=np.float64)
    y = (chi_array**2 - chi0**2) / chi0**2
    return 1.0 + y**2
