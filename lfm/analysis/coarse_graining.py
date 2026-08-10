"""Local Fourier blocking diagnostics for LFM lattice propagators.

The routines in this module evaluate exact alias sums produced by averaging
finite blocks of existing lattice registers and then sampling one value per
block. They do not add a field, modify a governing equation, or solve a
Poisson equation.

A finite local block map cannot turn an analytic gapped propagator into a
massless pole. The functions expose that statement numerically for the LFM
19-point baseline and the labeled 27-point ablation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.core.stencils import eigenvalue_19pt, eigenvalue_27pt

if TYPE_CHECKING:
    from collections.abc import Callable

Array = np.ndarray


def block_window_magnitude_sq(
    wave_number: Array | float,
    block_factor: int,
) -> Array:
    """Return the squared Fourier response of a finite block average.

    The block contains ``block_factor`` consecutive fine sites. Directly
    summing the phase factors avoids removable ``0/0`` singularities at
    reciprocal-lattice wave numbers.
    """
    if block_factor < 1:
        raise ValueError("block_factor must be at least one")
    values = np.asarray(wave_number, dtype=np.float64)
    offsets = np.arange(block_factor, dtype=np.float64)
    phases = np.exp(1j * values[..., np.newaxis] * offsets)
    window = np.mean(phases, axis=-1)
    return np.asarray(np.abs(window) ** 2, dtype=np.float64)


def _stencil_eigenvalue(stencil: str) -> Callable[[Array, Array, Array], Array]:
    if stencil == "19":
        return eigenvalue_19pt
    if stencil == "27":
        return eigenvalue_27pt
    raise ValueError("stencil must be '19' or '27'")


def blocked_static_propagator(
    coarse_kx: Array | float,
    coarse_ky: Array | float,
    coarse_kz: Array | float,
    *,
    block_factor: int,
    mass_sq: float,
    stencil: str = "19",
) -> Array:
    """Return the exact propagator of a locally block-averaged register.

    ``coarse_k*`` are wave numbers on the decimated grid. Each coarse mode
    aliases ``block_factor**3`` fine-grid modes

    ``k_fine = (k_coarse + 2*pi*n) / block_factor``.

    The returned response is the finite positive weighted sum of microscopic
    propagators ``1 / (mass_sq + K_stencil)``. For ``mass_sq > 0`` it is
    analytic at zero momentum. For ``mass_sq == 0`` the uniform mode is
    singular and callers must provide nonzero coarse wave numbers.
    """
    if block_factor < 1:
        raise ValueError("block_factor must be at least one")
    if not np.isfinite(mass_sq) or mass_sq < 0.0:
        raise ValueError("mass_sq must be finite and nonnegative")
    eigenvalue = _stencil_eigenvalue(stencil)

    kx, ky, kz = np.broadcast_arrays(
        np.asarray(coarse_kx, dtype=np.float64),
        np.asarray(coarse_ky, dtype=np.float64),
        np.asarray(coarse_kz, dtype=np.float64),
    )
    response = np.zeros_like(kx, dtype=np.float64)
    weight_sum = np.zeros_like(kx, dtype=np.float64)
    for alias_x in range(block_factor):
        fine_kx = (kx + 2.0 * np.pi * alias_x) / block_factor
        weight_x = block_window_magnitude_sq(fine_kx, block_factor)
        for alias_y in range(block_factor):
            fine_ky = (ky + 2.0 * np.pi * alias_y) / block_factor
            weight_y = block_window_magnitude_sq(fine_ky, block_factor)
            for alias_z in range(block_factor):
                fine_kz = (kz + 2.0 * np.pi * alias_z) / block_factor
                weight_z = block_window_magnitude_sq(
                    fine_kz,
                    block_factor,
                )
                weight = weight_x * weight_y * weight_z
                stiffness = -np.asarray(
                    eigenvalue(fine_kx, fine_ky, fine_kz),
                    dtype=np.float64,
                )
                stiffness = np.maximum(stiffness, 0.0)
                denominator = mass_sq + stiffness
                if np.any(denominator <= 0.0):
                    raise ValueError("massless blocked propagator requires nonzero modes")
                response += weight / denominator
                weight_sum += weight

    if not np.allclose(weight_sum, 1.0, rtol=2.0e-13, atol=2.0e-13):
        raise RuntimeError("block-alias weights do not form a partition")
    return response


def response_log_slope(
    wave_number: Array,
    response: Array,
    *,
    count: int,
) -> float:
    """Fit the small-wave-number power of a positive response."""
    k_values = np.asarray(wave_number, dtype=np.float64)
    response_values = np.asarray(response, dtype=np.float64)
    if k_values.shape != response_values.shape:
        raise ValueError("wave_number and response shapes must match")
    if count < 3 or count > k_values.size:
        raise ValueError("count must select at least three available modes")
    selected_k = k_values[:count]
    selected_response = response_values[:count]
    if np.any(selected_k <= 0.0) or np.any(selected_response <= 0.0):
        raise ValueError("slope fit requires positive values")
    return float(
        np.polyfit(
            np.log(selected_k),
            np.log(selected_response),
            1,
        )[0]
    )


def inverse_response_intercept(
    wave_number: Array,
    response: Array,
    *,
    count: int,
) -> float:
    """Fit the zero-wave-number intercept of the inverse response."""
    k_values = np.asarray(wave_number, dtype=np.float64)
    response_values = np.asarray(response, dtype=np.float64)
    if k_values.shape != response_values.shape:
        raise ValueError("wave_number and response shapes must match")
    if count < 3 or count > k_values.size:
        raise ValueError("count must select at least three available modes")
    if np.any(response_values[:count] <= 0.0):
        raise ValueError("intercept fit requires positive response")
    coefficients = np.polyfit(
        k_values[:count] ** 2,
        1.0 / response_values[:count],
        1,
    )
    return float(coefficients[1])


__all__ = [
    "block_window_magnitude_sq",
    "blocked_static_propagator",
    "inverse_response_intercept",
    "response_log_slope",
]
