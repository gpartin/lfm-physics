"""Discrete-to-continuum observables for LFM emergence tests.

The helpers in this module do not add fields or forces. They derive local
observables from the exact 19-point spatial symbol, the leapfrog time branch,
and normalized multicomponent wave fields. They are intended to keep a clear
lineage between finite-site LFM data and proposed continuum descriptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.analysis.poincare import discrete_omega, group_velocity
from lfm.core.stencils import gradient_19pt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


def _log_log_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("log-log slope inputs must be positive")
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def axial_static_inverse_length(
    mass: float,
    *,
    spacing: float,
    c: float = 1.0,
) -> float:
    """Return the exact axial inverse correlation length of the 19-point grid.

    Along a lattice axis the 19-point symbol reduces to the centered
    second-difference symbol. Analytically continuing the static pole gives

    ``mu = 2*asinh(mass*spacing/(2*c))/spacing``

    The continuum limit is ``mu -> mass/c``.
    """
    if mass <= 0.0 or spacing <= 0.0 or c <= 0.0:
        raise ValueError("mass, spacing, and c must be positive")
    return float(2.0 * np.arcsinh(0.5 * mass * spacing / c) / spacing)


def relational_wave_scaling_scan(
    chi_values: Iterable[float],
    q_values: Iterable[float],
    spacings: Iterable[float],
    *,
    mass_factors: Iterable[float] = (1.0,),
    c: float = 1.0,
    courant: float = 0.2,
) -> dict[str, object]:
    """Measure clock, ruler, dispersion, and velocity scaling from GOV-01.

    For a homogeneous local value of ``chi``, each linear branch with mass
    ``m = factor*chi`` has the exact discrete dispersion measured by
    :func:`lfm.analysis.poincare.discrete_omega`. The dimensionless local
    variables are ``Omega=omega/m`` and ``q=c*k/m``. A local static wave scale
    supplies the ruler through the inverse correlation length ``mu``.

    The scan keeps physical ``chi`` and ``q`` fixed while reducing lattice
    spacing and timestep at fixed Courant number. It therefore tests a real
    continuum limit instead of merely increasing the number of sites in the
    same lattice-unit configuration.
    """
    chis = np.asarray(tuple(chi_values), dtype=np.float64)
    q_modes = np.asarray(tuple(q_values), dtype=np.float64)
    h_values = np.asarray(tuple(spacings), dtype=np.float64)
    factors = np.asarray(tuple(mass_factors), dtype=np.float64)
    if (
        chis.size == 0
        or q_modes.size == 0
        or h_values.size < 3
        or factors.size == 0
    ):
        raise ValueError("nonempty scans and at least three spacings are required")
    if (
        np.any(chis <= 0.0)
        or np.any(q_modes <= 0.0)
        or np.any(h_values <= 0.0)
        or np.any(factors <= 0.0)
        or c <= 0.0
        or courant <= 0.0
    ):
        raise ValueError("scan values, c, and courant must be positive")

    rows: list[dict[str, float]] = []
    summaries: list[dict[str, float]] = []
    for spacing in h_values:
        dt = courant * spacing / c
        dispersion_errors: list[float] = []
        velocity_errors: list[float] = []
        clock_ruler_errors: list[float] = []
        acceleration_universality: list[float] = []
        grouped: dict[tuple[float, float], list[float]] = {}
        for chi in chis:
            for factor in factors:
                mass = factor * chi
                rest_omega = float(
                    discrete_omega(
                        (0.0, 0.0, 0.0),
                        mass=mass,
                        c=c,
                        dt=dt,
                        spacing=spacing,
                        stencil="19",
                    )
                )
                inverse_length = axial_static_inverse_length(
                    mass,
                    spacing=spacing,
                    c=c,
                )
                clock_ruler = rest_omega / (c * inverse_length)
                clock_ruler_error = abs(clock_ruler - 1.0)
                clock_ruler_errors.append(clock_ruler_error)

                # If m_alpha(x)=factor*chi(x), the factor cancels from
                # -c^2*grad(log(m_alpha)). This records the branch coefficient
                # used by the universality audit without inventing a force.
                acceleration_coefficient = 1.0
                acceleration_universality.append(acceleration_coefficient)
                for q_value in q_modes:
                    k_value = q_value * mass / c
                    omega = float(
                        discrete_omega(
                            (k_value, 0.0, 0.0),
                            mass=mass,
                            c=c,
                            dt=dt,
                            spacing=spacing,
                            stencil="19",
                        )
                    )
                    velocity = float(
                        group_velocity(
                            (k_value, 0.0, 0.0),
                            mass=mass,
                            c=c,
                            dt=dt,
                            spacing=spacing,
                            stencil="19",
                        )[0]
                    )
                    omega_target = float(np.sqrt(1.0 + q_value * q_value))
                    velocity_target = float(
                        c * q_value / np.sqrt(1.0 + q_value * q_value)
                    )
                    dimensionless_omega = omega / mass
                    dispersion_error = abs(dimensionless_omega - omega_target)
                    velocity_error = abs(velocity - velocity_target) / c
                    dispersion_errors.append(dispersion_error)
                    velocity_errors.append(velocity_error)
                    grouped.setdefault((float(factor), float(q_value)), []).append(
                        dimensionless_omega
                    )
                    rows.append(
                        {
                            "spacing": float(spacing),
                            "dt": float(dt),
                            "chi": float(chi),
                            "mass_factor": float(factor),
                            "mass": float(mass),
                            "q": float(q_value),
                            "dimensionless_omega": dimensionless_omega,
                            "dimensionless_omega_target": omega_target,
                            "dispersion_error": dispersion_error,
                            "group_velocity_over_c": velocity / c,
                            "group_velocity_target_over_c": velocity_target / c,
                            "group_velocity_error_over_c": velocity_error,
                            "clock_ruler_product": clock_ruler,
                            "clock_ruler_error": clock_ruler_error,
                            "rest_acceleration_log_chi_coefficient": (
                                acceleration_coefficient
                            ),
                        }
                    )
        local_spread = max(
            max(values) - min(values) for values in grouped.values()
        )
        summaries.append(
            {
                "spacing": float(spacing),
                "max_dispersion_error": float(max(dispersion_errors)),
                "max_group_velocity_error_over_c": float(max(velocity_errors)),
                "max_clock_ruler_error": float(max(clock_ruler_errors)),
                "max_local_chi_dispersion_spread": float(local_spread),
                "rest_acceleration_branch_spread": float(
                    max(acceleration_universality)
                    - min(acceleration_universality)
                ),
            }
        )

    summary_arrays = {
        key: np.asarray([row[key] for row in summaries], dtype=np.float64)
        for key in (
            "max_dispersion_error",
            "max_group_velocity_error_over_c",
            "max_clock_ruler_error",
            "max_local_chi_dispersion_spread",
        )
    }
    convergence = {
        f"{key}_slope": _log_log_slope(h_values, values)
        for key, values in summary_arrays.items()
    }
    finest = summaries[int(np.argmin(h_values))]
    gates = {
        "second_order_dispersion": convergence[
            "max_dispersion_error_slope"
        ] > 1.8,
        "second_order_group_velocity": convergence[
            "max_group_velocity_error_over_c_slope"
        ] > 1.8,
        "second_order_clock_ruler": convergence[
            "max_clock_ruler_error_slope"
        ] > 1.8,
        "local_relational_collapse": float(
            finest["max_local_chi_dispersion_spread"]
        ) < 1.0e-3,
        "linear_branch_rest_acceleration_universal": float(
            finest["rest_acceleration_branch_spread"]
        ) < 1.0e-14,
    }
    return {
        "definition": {
            "local_clock": "exact k=0 leapfrog frequency",
            "local_ruler": "inverse axial static correlation length",
            "dimensionless_variables": "Omega=omega/m, q=c*k/m",
            "wkb_rest_acceleration": "a=-c^2*grad(log(chi))",
        },
        "rows": rows,
        "spacing_summaries": summaries,
        "convergence": convergence,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }


def normalize_internal_field(field: NDArray) -> NDArray[np.complex128]:
    """Normalize a component-first complex wave field at every lattice site."""
    values = np.asarray(field, dtype=np.complex128)
    if values.ndim != 4:
        raise ValueError("field must have shape (components, nx, ny, nz)")
    norm = np.sqrt(np.sum(np.abs(values) ** 2, axis=0))
    if np.any(norm <= 0.0):
        raise ValueError("internal field must be nonzero at every site")
    return np.asarray(values / norm[None, ...], dtype=np.complex128)


def composite_connection_19pt(
    normalized_field: NDArray,
    *,
    dx: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return A_i=Im(z_dagger*partial_i z) from a normalized wave field."""
    z = np.asarray(normalized_field, dtype=np.complex128)
    if z.ndim != 4:
        raise ValueError("normalized_field must be component-first and 4-D")
    norm_error = float(np.max(np.abs(np.sum(np.abs(z) ** 2, axis=0) - 1.0)))
    if norm_error > 1.0e-10:
        raise ValueError("normalized_field must have unit site norm")
    connection = [np.zeros(z.shape[1:], dtype=np.float64) for _ in range(3)]
    for component in z:
        gradient_real = gradient_19pt(component.real, dx=dx)
        gradient_imag = gradient_19pt(component.imag, dx=dx)
        for axis in range(3):
            connection[axis] += (
                component.real * gradient_imag[axis]
                - component.imag * gradient_real[axis]
            )
    return connection[0], connection[1], connection[2]


def composite_curvature_19pt(
    normalized_field: NDArray,
    *,
    dx: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the spatial curl of the composite internal connection.

    This is a derived D2 observable. It is not an independently evolved gauge
    field and this function makes no Maxwell-dynamics claim.
    """
    ax, ay, az = composite_connection_19pt(normalized_field, dx=dx)
    grad_ax = gradient_19pt(ax, dx=dx)
    grad_ay = gradient_19pt(ay, dx=dx)
    grad_az = gradient_19pt(az, dx=dx)
    f_xy = grad_ay[0] - grad_ax[1]
    f_yz = grad_az[1] - grad_ay[2]
    f_zx = grad_ax[2] - grad_az[0]
    return f_xy, f_yz, f_zx


def curvature_rms(curvature: tuple[NDArray, NDArray, NDArray]) -> float:
    """Return RMS magnitude of a three-component spatial curvature."""
    values = tuple(np.asarray(component, dtype=np.float64) for component in curvature)
    if len({value.shape for value in values}) != 1:
        raise ValueError("curvature components must have matching shapes")
    return float(np.sqrt(np.mean(sum(value * value for value in values))))
