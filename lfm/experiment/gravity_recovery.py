"""Local diagnostics for the experiment-only GOV-02 gravity recovery study.

This module contains observables and energy accounting. Evolution remains in
``Simulation``/``Evolver`` so experiments do not duplicate GOV-01/GOV-02
loops. No routine here performs an inverse solve or inserts a target profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit

from lfm.analysis.energy_current import stencil_links
from lfm.config import ChiPotentialModel
from lfm.constants import CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import gradient_19pt, laplacian_19pt


@dataclass(frozen=True)
class ChiCandidate:
    """Frozen metadata for one local stabilization candidate."""

    model: ChiPotentialModel
    family: str
    label: str
    conservative: bool
    gov01_compatible: bool
    modifies_gradient: bool = False
    modifies_inertia: bool = False
    source_dependent: bool = False


_CANDIDATES = (
    ChiCandidate(
        ChiPotentialModel.CANONICAL_QUARTIC,
        "A",
        "canonical_quartic_mexican_hat",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.FLAT_OCTIC,
        "B",
        "geometry_normalized_flat_octic",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.FLAT_DODECIC,
        "C",
        "geometry_normalized_flat_dodecic",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.FLAT_POWER_8,
        "D",
        "geometry_normalized_y_power_8",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.FLAT_POWER_10,
        "D",
        "geometry_normalized_y_power_10",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.FLAT_POWER_12,
        "D",
        "geometry_normalized_y_power_12",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.SMOOTH_EXPONENTIAL,
        "E",
        "smooth_exponential_crossover",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.RATIONAL_CROSSOVER,
        "F",
        "rational_flat_crossover",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.HYPERBOLIC_CROSSOVER,
        "G",
        "hyperbolic_tangent_crossover",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.NONLINEAR_GRADIENT,
        "H",
        "flat_octic_plus_nonlinear_link_gradient",
        True,
        True,
        modifies_gradient=True,
    ),
    ChiCandidate(
        ChiPotentialModel.AMPLITUDE_STRENGTHENED,
        "I",
        "amplitude_strengthened_flat_well",
        True,
        True,
    ),
    ChiCandidate(
        ChiPotentialModel.SOURCE_DEPENDENT,
        "J",
        "local_source_dependent_stabilization",
        True,
        False,
        source_dependent=True,
    ),
    ChiCandidate(
        ChiPotentialModel.VARIABLE_INERTIA,
        "K",
        "local_variable_inertia_stabilization",
        True,
        True,
        modifies_inertia=True,
    ),
    ChiCandidate(
        ChiPotentialModel.RADICAL_CROSSOVER,
        "L",
        "local_radical_crossover",
        True,
        True,
    ),
)


def positive_frequency_previous_layers(
    psi_real: np.ndarray,
    psi_imag: np.ndarray,
    chi: np.ndarray,
    *,
    dt: float,
    dx: float = 1.0,
    polynomial_degree: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Construct positive-frequency leapfrog Cauchy data locally.

    For ``A=-Delta_19+chi^2``, a leapfrog eigenmode obeys
    ``cos(theta)=1-dt^2*A/2``. The previous layer of a positive-frequency
    mode is therefore ``exp(+i theta) psi``. The sine factor is evaluated as
    a finite Chebyshev polynomial of ``A``. Each polynomial application is a
    composition of local 19-point stencil operations; no inverse or spectral
    force solver is used.
    """

    real = np.asarray(psi_real, dtype=np.float64)
    imag = np.asarray(psi_imag, dtype=np.float64)
    chi_array = np.asarray(chi, dtype=np.float64)
    if real.shape != imag.shape:
        raise ValueError("real and imaginary fields must have the same shape")
    if real.shape[-3:] != chi_array.shape:
        raise ValueError("field spatial shape must match chi")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive and finite")
    if polynomial_degree < 1:
        raise ValueError("polynomial_degree must be positive")

    spatial_shape = chi_array.shape
    leading_shape = real.shape[:-3]
    channel_count = int(np.prod(leading_shape)) if leading_shape else 1
    real_channels = real.reshape((channel_count,) + spatial_shape)
    imag_channels = imag.reshape((channel_count,) + spatial_shape)
    chi_sq = chi_array**2
    inverse_dx_sq = 1.0 / dx**2

    def apply_a(values: np.ndarray) -> np.ndarray:
        return -inverse_dx_sq * laplacian_19pt(values) + chi_sq * values

    minimum_eigenvalue = max(float(np.min(chi_sq)), 1.0e-12)
    maximum_eigenvalue = float(np.max(chi_sq)) + 8.0 * inverse_dx_sq
    if 0.25 * dt**2 * maximum_eigenvalue >= 1.0:
        raise ValueError("dt exceeds the positive-frequency CFL interval")
    midpoint = 0.5 * (minimum_eigenvalue + maximum_eigenvalue)
    half_width = 0.5 * (maximum_eigenvalue - minimum_eigenvalue)

    def sine_frequency(normalized: np.ndarray) -> np.ndarray:
        eigenvalue = midpoint + half_width * normalized
        return np.sqrt(eigenvalue * (1.0 - 0.25 * dt**2 * eigenvalue))

    coefficients = np.polynomial.chebyshev.chebinterpolate(  # type: ignore[type-var]
        sine_frequency,
        polynomial_degree,
    )

    def apply_x(values: np.ndarray) -> np.ndarray:
        return (apply_a(values) - midpoint * values) / half_width

    def apply_sine_frequency(values: np.ndarray) -> np.ndarray:
        next_term = np.zeros_like(values)
        next_next_term = np.zeros_like(values)
        for index in range(polynomial_degree, 0, -1):
            current = 2.0 * apply_x(next_term) - next_next_term + coefficients[index] * values
            next_next_term = next_term
            next_term = current
        return apply_x(next_term) - next_next_term + coefficients[0] * values

    previous_real = np.zeros_like(real_channels)
    previous_imag = np.zeros_like(imag_channels)
    for channel in range(channel_count):
        current_real = real_channels[channel]
        current_imag = imag_channels[channel]
        cosine_real = current_real - 0.5 * dt**2 * apply_a(current_real)
        cosine_imag = current_imag - 0.5 * dt**2 * apply_a(current_imag)
        sine_real = dt * apply_sine_frequency(current_real)
        sine_imag = dt * apply_sine_frequency(current_imag)
        previous_real[channel] = cosine_real - sine_imag
        previous_imag[channel] = cosine_imag + sine_real
    metadata: dict[str, float | int] = {
        "polynomial_degree": polynomial_degree,
        "minimum_operator_eigenvalue_bound": minimum_eigenvalue,
        "maximum_operator_eigenvalue_bound": maximum_eigenvalue,
        "maximum_chebyshev_coefficient": float(np.max(np.abs(coefficients))),
        "local_stencil_radius_upper_bound": polynomial_degree,
    }
    return (
        previous_real.reshape(real.shape),
        previous_imag.reshape(imag.shape),
        metadata,
    )


def gravity_recovery_candidates() -> tuple[ChiCandidate, ...]:
    """Return the frozen candidate catalog."""

    return _CANDIDATES


def candidate_manifest() -> list[dict[str, object]]:
    """Return JSON-serializable candidate declarations."""

    return [
        {
            **asdict(candidate),
            "model": int(candidate.model),
            "model_name": candidate.model.name,
        }
        for candidate in _CANDIDATES
    ]


def _dimensionless_y(
    chi: np.ndarray,
    chi0: float,
) -> np.ndarray:
    return (np.asarray(chi, dtype=np.float64) ** 2 - chi0**2) / chi0**2


def dimensionless_potential(
    y: np.ndarray,
    model: ChiPotentialModel,
    source_ratio: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return f(y) where V=Lambda_H*chi0^4*f(y)."""

    y = np.asarray(y, dtype=np.float64)
    model = ChiPotentialModel(model)
    if model == ChiPotentialModel.CANONICAL_QUARTIC:
        return y**2
    if model in (
        ChiPotentialModel.FLAT_OCTIC,
        ChiPotentialModel.NONLINEAR_GRADIENT,
        ChiPotentialModel.VARIABLE_INERTIA,
    ):
        return y**4
    if model == ChiPotentialModel.FLAT_DODECIC:
        return y**6
    if model == ChiPotentialModel.FLAT_POWER_8:
        return y**8
    if model == ChiPotentialModel.FLAT_POWER_10:
        return y**10
    if model == ChiPotentialModel.FLAT_POWER_12:
        return y**12
    if model == ChiPotentialModel.SMOOTH_EXPONENTIAL:
        return y**2 * (1.0 - np.exp(-(y**2)))
    if model == ChiPotentialModel.RATIONAL_CROSSOVER:
        return y**4 / (1.0 + y**2)
    if model == ChiPotentialModel.HYPERBOLIC_CROSSOVER:
        return (y * np.tanh(y)) ** 2
    if model == ChiPotentialModel.AMPLITUDE_STRENGTHENED:
        return y**4 * (1.0 + y**2)
    if model == ChiPotentialModel.SOURCE_DEPENDENT:
        return y**4 + np.asarray(source_ratio) * y**2
    if model == ChiPotentialModel.RADICAL_CROSSOVER:
        return np.sqrt(1.0 + y**8) - 1.0
    raise ValueError(f"unsupported chi potential model: {model}")


def dimensionless_potential_derivative(
    y: np.ndarray,
    model: ChiPotentialModel,
    source_ratio: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return df/dy for the frozen candidate family."""

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


def potential_density(
    chi: np.ndarray,
    model: ChiPotentialModel,
    *,
    chi0: float = CHI0,
    lambda_h: float = LAMBDA_H,
    source_density: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return the local self-potential density."""

    y = _dimensionless_y(chi, chi0)
    source_ratio = np.asarray(source_density) / chi0**2
    return lambda_h * chi0**4 * dimensionless_potential(y, model, source_ratio)


def potential_force(
    chi: np.ndarray,
    model: ChiPotentialModel,
    *,
    chi0: float = CHI0,
    lambda_h: float = LAMBDA_H,
    source_density: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return -dV/dchi, the local acceleration contribution."""

    chi_array = np.asarray(chi, dtype=np.float64)
    y = _dimensionless_y(chi_array, chi0)
    source_ratio = np.asarray(source_density) / chi0**2
    derivative = dimensionless_potential_derivative(
        y,
        model,
        source_ratio,
    )
    return -2.0 * lambda_h * chi0**2 * chi_array * derivative


def potential_second_derivative_at_vacuum(
    model: ChiPotentialModel,
    *,
    chi0: float = CHI0,
    lambda_h: float = LAMBDA_H,
) -> float:
    """Return V''(chi0) using a symmetric high-accuracy finite difference."""

    step = 1.0e-4 * chi0
    values = np.asarray([chi0 - step, chi0, chi0 + step])
    potential = potential_density(
        values,
        model,
        chi0=chi0,
        lambda_h=lambda_h,
    )
    return float((potential[2] - 2.0 * potential[1] + potential[0]) / step**2)


def variable_inertia(
    chi: np.ndarray,
    *,
    chi0: float = CHI0,
) -> np.ndarray:
    """Return the K-family local kinetic multiplier M(chi)=1+y^2."""

    y = _dimensionless_y(chi, chi0)
    return 1.0 + y**2


def chi_hamiltonian(
    chi: np.ndarray,
    chi_prev: np.ndarray,
    source_density: np.ndarray,
    model: ChiPotentialModel,
    *,
    dt: float,
    dx: float = 1.0,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
    e0_sq: float = 0.0,
) -> dict[str, float]:
    """Return the conservative chi-subsystem Hamiltonian component ledger."""

    chi = np.asarray(chi, dtype=np.float64)
    chi_prev = np.asarray(chi_prev, dtype=np.float64)
    source_density = np.asarray(source_density, dtype=np.float64)
    velocity = (chi - chi_prev) / dt
    inertia = (
        variable_inertia(chi, chi0=chi0) if model == ChiPotentialModel.VARIABLE_INERTIA else 1.0
    )
    kinetic = float(np.sum(0.5 * inertia * velocity**2))
    gradient = 0.0
    nonlinear_gradient = 0.0
    for offset, weight in stencil_links("19", oriented=False):
        neighbor = np.roll(
            chi,
            shift=tuple(-value for value in offset),
            axis=(0, 1, 2),
        )
        difference = (neighbor - chi) / dx
        gradient += float(np.sum(0.5 * weight * difference**2))
        if model == ChiPotentialModel.NONLINEAR_GRADIENT:
            nonlinear_gradient += float(np.sum(0.25 * weight * difference**4 / chi0**2))
    potential = float(
        np.sum(
            potential_density(
                chi,
                model,
                chi0=chi0,
                lambda_h=lambda_h,
                source_density=source_density - e0_sq,
            )
        )
    )
    source = float(np.sum(0.5 * (kappa / chi0) * (source_density - e0_sq) * chi**2))
    total = kinetic + gradient + nonlinear_gradient + potential + source
    return {
        "kinetic": kinetic,
        "gradient": gradient,
        "nonlinear_gradient": nonlinear_gradient,
        "potential": potential,
        "source": source,
        "total": total,
    }


def radial_shell_profile(
    values: np.ndarray,
    *,
    center: tuple[float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Return unit-width spherical-shell means, standard deviations, counts."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("values must be a 3D field")
    shape = values.shape
    if center is None:
        center = tuple((size - 1.0) / 2.0 for size in shape)
    coordinates = np.indices(shape, dtype=np.float64)
    radius = np.sqrt(sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3)))
    shell = np.floor(radius + 0.5).astype(np.int32)
    max_shell = int(shell.max())
    flat_shell = shell.ravel()
    flat_values = values.ravel()
    counts = np.bincount(flat_shell, minlength=max_shell + 1)
    sums = np.bincount(
        flat_shell,
        weights=flat_values,
        minlength=max_shell + 1,
    )
    sums_sq = np.bincount(
        flat_shell,
        weights=flat_values**2,
        minlength=max_shell + 1,
    )
    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    variances = (
        np.divide(
            sums_sq,
            counts,
            out=np.full_like(sums_sq, np.nan, dtype=np.float64),
            where=counts > 0,
        )
        - means**2
    )
    return {
        "radius": np.arange(max_shell + 1, dtype=np.float64),
        "mean": means,
        "std": np.sqrt(np.maximum(variances, 0.0)),
        "count": counts,
    }


def _linear_fit(
    design: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ coefficients
    residual = float(np.sum((values - predicted) ** 2))
    total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return coefficients, r_squared


def fit_inverse_r(
    radius: np.ndarray,
    profile: np.ndarray,
    *,
    r_min: float,
    r_max: float,
) -> dict[str, float]:
    """Fit profile=A/r+B on a predeclared radial window."""

    radius = np.asarray(radius, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    keep = (radius >= r_min) & (radius <= r_max) & np.isfinite(profile) & (radius > 0.0)
    if np.count_nonzero(keep) < 4:
        raise ValueError("inverse-r fit requires at least four shells")
    r = radius[keep]
    values = profile[keep]
    design = np.column_stack((1.0 / r, np.ones_like(r)))
    coefficients, r_squared = _linear_fit(design, values)
    return {
        "amplitude": float(coefficients[0]),
        "offset": float(coefficients[1]),
        "r_squared": r_squared,
        "point_count": int(r.size),
        "r_min": float(r_min),
        "r_max": float(r_max),
    }


def fit_power_law(
    radius: np.ndarray,
    magnitude: np.ndarray,
    *,
    r_min: float,
    r_max: float,
) -> dict[str, float]:
    """Fit magnitude=C*r^slope on a predeclared radial window."""

    radius = np.asarray(radius, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    keep = (radius >= r_min) & (radius <= r_max) & np.isfinite(magnitude) & (magnitude > 0.0)
    if np.count_nonzero(keep) < 4:
        raise ValueError("power-law fit requires at least four positive shells")
    log_r = np.log(radius[keep])
    log_magnitude = np.log(magnitude[keep])
    design = np.column_stack((log_r, np.ones_like(log_r)))
    coefficients, r_squared = _linear_fit(design, log_magnitude)
    return {
        "slope": float(coefficients[0]),
        "log_amplitude": float(coefficients[1]),
        "r_squared": r_squared,
        "point_count": int(log_r.size),
        "r_min": float(r_min),
        "r_max": float(r_max),
    }


def fit_yukawa(
    radius: np.ndarray,
    profile: np.ndarray,
    *,
    r_min: float,
    r_max: float,
) -> dict[str, float]:
    """Fit profile=A*exp(-r/L)/r+B without an inverse field solve."""

    radius = np.asarray(radius, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    keep = (radius >= r_min) & (radius <= r_max) & np.isfinite(profile) & (radius > 0.0)
    if np.count_nonzero(keep) < 5:
        raise ValueError("Yukawa fit requires at least five shells")
    r = radius[keep]
    values = profile[keep]

    def model(
        radius_value: np.ndarray,
        amplitude: float,
        length: float,
        offset: float,
    ) -> np.ndarray:
        return amplitude * np.exp(-radius_value / length) / radius_value + offset

    amplitude_guess = float((values[0] - values[-1]) * r[0])
    offset_guess = float(values[-1])
    parameters, _ = curve_fit(
        model,
        r,
        values,
        p0=(amplitude_guess, max(1.0, 0.25 * r_max), offset_guess),
        bounds=(
            (-np.inf, 0.05, -np.inf),
            (np.inf, 10.0 * r_max, np.inf),
        ),
        maxfev=50_000,
    )
    predicted = model(r, *parameters)
    residual = float(np.sum((values - predicted) ** 2))
    total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return {
        "amplitude": float(parameters[0]),
        "screening_length": float(parameters[1]),
        "offset": float(parameters[2]),
        "r_squared": r_squared,
        "point_count": int(r.size),
        "r_min": float(r_min),
        "r_max": float(r_max),
    }


def profile_observables(
    chi: np.ndarray,
    *,
    chi0: float = CHI0,
    dx: float = 1.0,
    center: tuple[float, float, float] | None = None,
    r_min: float,
    r_max: float,
) -> dict[str, object]:
    """Measure profile, acceleration proxy, flux, and angular anisotropy."""

    chi = np.asarray(chi, dtype=np.float64)
    delta = chi - chi0
    profile = radial_shell_profile(delta, center=center)
    profile["radius"] = profile["radius"] * dx
    grad_x, grad_y, grad_z = gradient_19pt(chi, dx=dx)
    magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    acceleration = radial_shell_profile(magnitude, center=center)
    acceleration["radius"] = acceleration["radius"] * dx
    inverse_r = fit_inverse_r(
        profile["radius"],
        profile["mean"],
        r_min=r_min,
        r_max=r_max,
    )
    power_law = fit_power_law(
        acceleration["radius"],
        acceleration["mean"],
        r_min=r_min,
        r_max=r_max,
    )
    yukawa = fit_yukawa(
        profile["radius"],
        profile["mean"],
        r_min=r_min,
        r_max=r_max,
    )
    keep = (
        (acceleration["radius"] >= r_min)
        & (acceleration["radius"] <= r_max)
        & np.isfinite(acceleration["mean"])
    )
    flux = acceleration["radius"][keep] ** 2 * acceleration["mean"][keep]
    flux_relative_spread = float(np.std(flux) / max(abs(float(np.mean(flux))), 1.0e-30))
    profile_anisotropy = float(
        np.nanmax(
            np.divide(
                profile["std"][keep],
                np.maximum(np.abs(profile["mean"][keep]), 1.0e-30),
            )
        )
    )
    return {
        "inverse_r_fit": inverse_r,
        "acceleration_power_fit": power_law,
        "yukawa_fit": yukawa,
        "shell_flux_relative_spread": flux_relative_spread,
        "profile_anisotropy_max": profile_anisotropy,
        "radial_profile": {key: np.asarray(value).tolist() for key, value in profile.items()},
        "acceleration_profile": {
            key: np.asarray(value).tolist() for key, value in acceleration.items()
        },
    }
