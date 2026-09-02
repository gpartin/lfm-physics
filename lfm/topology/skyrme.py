"""Unit-degree SU(2) hedgehogs and explicit FR quantization certificates.

This module implements an experimental Skyrme-type extension. It is not part
of the canonical LFM action, and no result from it is an electron observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import MatrixRankWarning, spsolve


@dataclass(frozen=True)
class SkyrmeHedgehogEnergy:
    """Dimensionless static energy and Derrick components."""

    total: float
    sigma: float
    skyrme: float


@dataclass(frozen=True)
class SkyrmeHedgehogSolution:
    """One source-free radial degree-one hedgehog solution."""

    radius: float
    dr: float
    r: np.ndarray
    profile: np.ndarray
    energy: SkyrmeHedgehogEnergy
    degree: float
    rms_topological_radius: float
    stationary_relative_residual: float
    unitarity_residual: float
    derrick_relative_first_derivative: float
    derrick_relative_second_derivative: float
    continuum_virial_mismatch: float
    optimizer_converged: bool
    optimizer_iterations: int
    newton_converged: bool
    newton_iterations: int
    message: str


@dataclass(frozen=True)
class FRQuantization:
    """Selected one-dimensional character of the Z2 universal-cover deck group.

    ``deck_character=-1`` is the fermionic Finkelstein-Rubinstein choice.
    Topology permits this choice for odd degree; canonical LFM does not
    currently derive it uniquely.
    """

    degree: int
    deck_character: int

    def __post_init__(self) -> None:
        if self.degree == 0:
            raise ValueError("degree must be nonzero")
        if self.deck_character not in (-1, 1):
            raise ValueError("deck_character must be -1 or +1")

    @property
    def rotation_loop_class(self) -> int:
        """Z2 class of one physical 2*pi rotation."""
        return abs(self.degree) % 2

    @property
    def exchange_loop_class(self) -> int:
        """Z2 class of one identical-soliton exchange."""
        return abs(self.degree) % 2

    def character(self, loop_class: int) -> int:
        """Evaluate the selected deck character on a Z2 loop class."""
        return self.deck_character ** (int(loop_class) % 2)

    def rotation_sign(self, full_turns: int = 1) -> int:
        """Wavefunction sign after an integer number of physical full turns."""
        loop_class = self.rotation_loop_class * int(full_turns)
        return self.character(loop_class)

    def exchange_sign(self, exchanges: int = 1) -> int:
        """Wavefunction sign after an integer number of identical exchanges."""
        loop_class = self.exchange_loop_class * int(exchanges)
        return self.character(loop_class)

    def wavefunction_on_sheet(
        self,
        base_amplitude: complex,
        sheet: int,
    ) -> complex:
        """Evaluate a universal-cover wavefunction on a chosen deck sheet."""
        return complex(base_amplitude) * self.character(sheet)

    def certificate(self) -> dict[str, object]:
        """Return an auditable topology/quantization statement."""
        return {
            "spatial_compactification": "S3",
            "target_manifold": "SU(2)~S3",
            "degree": self.degree,
            "configuration_space_pi1": "Z2",
            "rotation_loop_class": self.rotation_loop_class,
            "rotation_sign_2pi": self.rotation_sign(1),
            "rotation_sign_4pi": self.rotation_sign(2),
            "exchange_loop_class": self.exchange_loop_class,
            "exchange_sign_once": self.exchange_sign(1),
            "exchange_sign_twice": self.exchange_sign(2),
            "deck_character": self.deck_character,
            "deck_character_status": ("explicit_quantization_choice_not_uniquely_derived_from_LFM"),
            "primary_references": [
                "doi:10.1063/1.1664510",
                "arXiv:hep-th/9301101",
                "arXiv:hep-th/0509094",
            ],
        }


def _radial_nodes(radius: float, dr: float) -> np.ndarray:
    if radius <= 0.0 or dr <= 0.0:
        raise ValueError("radius and dr must be positive")
    segments_float = radius / dr
    segments = int(round(segments_float))
    if segments < 8 or not np.isclose(
        segments * dr,
        radius,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("radius/dr must be an integer of at least 8")
    return np.arange(segments + 1, dtype=np.float64) * dr


def make_hedgehog_profile(
    *,
    radius: float,
    dr: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Return the frozen smooth degree-one analytic initial profile."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    r = _radial_nodes(radius, dr)
    profile = np.empty_like(r)
    profile[0] = np.pi
    profile[-1] = 0.0
    if profile.size > 2:
        profile[1:-1] = 2.0 * np.arctan((scale / r[1:-1]) ** 2)
    return profile


def prolong_hedgehog_profile(
    *,
    source_r: np.ndarray,
    source_profile: np.ndarray,
    radius: float,
    new_dr: float,
) -> np.ndarray:
    """Cell-node prolongation with exact declared boundary values."""
    old_r = np.asarray(source_r, dtype=np.float64)
    old_profile = np.asarray(source_profile, dtype=np.float64)
    if old_r.ndim != 1 or old_r.size < 3 or old_profile.shape != old_r.shape:
        raise ValueError("source grid and profile must be matching vectors")
    if not np.all(np.diff(old_r) > 0.0):
        raise ValueError("source_r must be strictly increasing")
    new_r = _radial_nodes(radius, new_dr)
    profile = np.interp(
        new_r,
        old_r,
        old_profile,
        left=np.pi,
        right=0.0,
    )
    profile[0] = np.pi
    profile[-1] = 0.0
    return profile


def _validate_profile(
    profile: np.ndarray,
    *,
    radius: float,
    dr: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = _radial_nodes(radius, dr)
    values = np.asarray(profile, dtype=np.float64)
    if values.shape != r.shape:
        raise ValueError(f"profile must have shape {r.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile must be finite")
    if not np.isclose(values[0], np.pi, rtol=0.0, atol=1.0e-13):
        raise ValueError("profile origin boundary must be pi")
    if not np.isclose(values[-1], 0.0, rtol=0.0, atol=1.0e-13):
        raise ValueError("profile outer boundary must be zero")
    return r, values


def hedgehog_energy_gradient_hessian(
    profile: np.ndarray,
    *,
    radius: float,
    dr: float,
) -> tuple[SkyrmeHedgehogEnergy, np.ndarray, sp.csr_matrix, np.ndarray]:
    """Evaluate the frozen midpoint energy, gradient, Hessian, and force scale."""
    r, values = _validate_profile(profile, radius=radius, dr=dr)
    interiors = values.size - 2
    gradient_full = np.zeros_like(values)
    force_scale_full = np.zeros_like(values)
    main = np.zeros(interiors, dtype=np.float64)
    off = np.zeros(max(interiors - 1, 0), dtype=np.float64)
    sigma_sum = 0.0
    skyrme_sum = 0.0
    factor = 4.0 * np.pi * dr

    for index in range(values.size - 1):
        left = values[index]
        right = values[index + 1]
        midpoint = 0.5 * (left + right)
        derivative = (right - left) / dr
        radial_midpoint = 0.5 * (r[index] + r[index + 1])
        sin_midpoint = np.sin(midpoint)
        cos_midpoint = np.cos(midpoint)
        sin_squared = sin_midpoint * sin_midpoint
        radial_squared = radial_midpoint * radial_midpoint

        sigma_density = 0.5 * (radial_squared * derivative * derivative + 2.0 * sin_squared)
        skyrme_density = sin_squared * (
            derivative * derivative + 0.5 * sin_squared / radial_squared
        )
        sigma_sum += factor * sigma_density
        skyrme_sum += factor * skyrme_density

        partial_derivative = derivative * (radial_squared + 2.0 * sin_squared)
        partial_midpoint = (
            2.0
            * sin_midpoint
            * cos_midpoint
            * (1.0 + derivative * derivative + sin_squared / radial_squared)
        )
        local_left = factor * (-partial_derivative / dr + 0.5 * partial_midpoint)
        local_right = factor * (partial_derivative / dr + 0.5 * partial_midpoint)
        gradient_full[index] += local_left
        gradient_full[index + 1] += local_right
        force_scale_full[index] += abs(local_left)
        force_scale_full[index + 1] += abs(local_right)

        second_derivative = radial_squared + 2.0 * sin_squared
        mixed_derivative = 4.0 * derivative * sin_midpoint * cos_midpoint
        second_midpoint = (
            2.0
            * np.cos(2.0 * midpoint)
            * (1.0 + derivative * derivative + sin_squared / radial_squared)
            + np.sin(2.0 * midpoint) ** 2 / radial_squared
        )
        hessian_left = factor * (
            second_derivative / (dr * dr) - mixed_derivative / dr + 0.25 * second_midpoint
        )
        hessian_right = factor * (
            second_derivative / (dr * dr) + mixed_derivative / dr + 0.25 * second_midpoint
        )
        hessian_cross = factor * (-second_derivative / (dr * dr) + 0.25 * second_midpoint)

        if 0 < index < values.size - 1:
            main[index - 1] += hessian_left
        if 0 < index + 1 < values.size - 1:
            main[index] += hessian_right
        if 0 < index < values.size - 2:
            off[index - 1] += hessian_cross

    hessian = sp.diags(
        diagonals=(off, main, off),
        offsets=(-1, 0, 1),
        shape=(interiors, interiors),
        format="csr",
    )
    return (
        SkyrmeHedgehogEnergy(
            total=float(sigma_sum + skyrme_sum),
            sigma=float(sigma_sum),
            skyrme=float(skyrme_sum),
        ),
        gradient_full[1:-1],
        hessian,
        force_scale_full[1:-1],
    )


def hedgehog_degree(
    profile: np.ndarray,
    *,
    radius: float,
    dr: float,
) -> float:
    """Integrate the degree density using the same midpoint grid."""
    _, values = _validate_profile(profile, radius=radius, dr=dr)
    midpoint = 0.5 * (values[:-1] + values[1:])
    derivative = np.diff(values) / dr
    return float(-(2.0 / np.pi) * np.sum(dr * derivative * np.sin(midpoint) ** 2))


def hedgehog_unitarity_residual(profile: np.ndarray) -> float:
    """Return max |cos(F)^2+sin(F)^2-1| for the SU(2) parameterization."""
    values = np.asarray(profile, dtype=np.float64)
    return float(np.max(np.abs(np.cos(values) ** 2 + np.sin(values) ** 2 - 1.0)))


def _stationary_residual(
    profile: np.ndarray,
    *,
    radius: float,
    dr: float,
) -> float:
    _, gradient, _, force_scale = hedgehog_energy_gradient_hessian(
        profile,
        radius=radius,
        dr=dr,
    )
    denominator = max(float(np.linalg.norm(force_scale)), 1.0e-300)
    return float(np.linalg.norm(gradient)) / denominator


def _newton_polish(
    profile: np.ndarray,
    *,
    radius: float,
    dr: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, bool, int, str]:
    current = np.asarray(profile, dtype=np.float64).copy()
    for iteration in range(max_iterations + 1):
        energy, gradient, hessian, force_scale = hedgehog_energy_gradient_hessian(
            current,
            radius=radius,
            dr=dr,
        )
        residual = float(np.linalg.norm(gradient)) / max(
            float(np.linalg.norm(force_scale)),
            1.0e-300,
        )
        if residual <= tolerance:
            return current, True, iteration, "stationary residual converged"
        if iteration == max_iterations:
            break

        diagonal_scale = np.maximum(np.abs(hessian.diagonal()), 1.0)
        accepted = False
        for damping in (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
            system = (
                hessian
                if damping == 0.0
                else hessian + sp.diags(damping * diagonal_scale, format="csr")
            )
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    delta = spsolve(system, -gradient)
            except (MatrixRankWarning, RuntimeError, ValueError):
                continue
            if not np.all(np.isfinite(delta)):
                continue
            step = 1.0
            while step >= 1.0e-12:
                candidate = current.copy()
                candidate[1:-1] += step * delta
                candidate_energy, _, _, _ = hedgehog_energy_gradient_hessian(
                    candidate,
                    radius=radius,
                    dr=dr,
                )
                candidate_residual = _stationary_residual(
                    candidate,
                    radius=radius,
                    dr=dr,
                )
                if (
                    candidate_energy.total <= energy.total * (1.0 + 1.0e-14)
                    and candidate_residual < residual
                ):
                    current = candidate
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                break
        if not accepted:
            return current, False, iteration, "Newton line search failed"
    return current, False, max_iterations, "Newton iteration limit reached"


def _topological_rms_radius(
    profile: np.ndarray,
    *,
    radius: float,
    dr: float,
) -> float:
    r, values = _validate_profile(profile, radius=radius, dr=dr)
    radial_midpoint = 0.5 * (r[:-1] + r[1:])
    midpoint = 0.5 * (values[:-1] + values[1:])
    derivative = np.diff(values) / dr
    weights = -(2.0 / np.pi) * dr * derivative * np.sin(midpoint) ** 2
    degree = float(np.sum(weights))
    if degree <= 0.0:
        raise ValueError("topological density does not have positive degree")
    return float(np.sqrt(np.dot(weights, radial_midpoint**2) / degree))


def _derrick_scale_variations(
    profile: np.ndarray,
    *,
    radius: float,
    dr: float,
) -> tuple[float, float]:
    """Evaluate the discrete action along its infinitesimal scale mode."""
    r, values = _validate_profile(profile, radius=radius, dr=dr)
    energy, gradient, hessian, _ = hedgehog_energy_gradient_hessian(
        values,
        radius=radius,
        dr=dr,
    )
    derivative = np.gradient(values, dr, edge_order=2)
    scale_mode = -r[1:-1] * derivative[1:-1]
    first = abs(float(np.dot(gradient, scale_mode))) / max(
        energy.total,
        1.0e-300,
    )
    second = float(np.dot(scale_mode, hessian @ scale_mode)) / max(
        energy.total,
        1.0e-300,
    )
    return first, second


def solve_skyrme_hedgehog(
    *,
    radius: float,
    dr: float,
    initial_profile: np.ndarray | None = None,
    max_iterations: int = 4000,
    gradient_tolerance: float = 1.0e-10,
    newton_tolerance: float = 1.0e-12,
    newton_max_iterations: int = 80,
) -> SkyrmeHedgehogSolution:
    """Minimize the frozen degree-one hedgehog energy without clipping."""
    initial = (
        make_hedgehog_profile(radius=radius, dr=dr)
        if initial_profile is None
        else np.asarray(initial_profile, dtype=np.float64).copy()
    )
    _validate_profile(initial, radius=radius, dr=dr)

    def objective(interior: np.ndarray) -> tuple[float, np.ndarray]:
        profile = np.empty(interior.size + 2, dtype=np.float64)
        profile[0] = np.pi
        profile[-1] = 0.0
        profile[1:-1] = interior
        energy, gradient, _, _ = hedgehog_energy_gradient_hessian(
            profile,
            radius=radius,
            dr=dr,
        )
        return energy.total, gradient

    result = minimize(
        objective,
        initial[1:-1],
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1.0e-15,
            "maxls": 50,
            "maxcor": 30,
        },
    )
    optimized = np.empty_like(initial)
    optimized[0] = np.pi
    optimized[-1] = 0.0
    optimized[1:-1] = np.asarray(result.x, dtype=np.float64)
    (
        final_profile,
        newton_converged,
        newton_iterations,
        newton_message,
    ) = _newton_polish(
        optimized,
        radius=radius,
        dr=dr,
        tolerance=newton_tolerance,
        max_iterations=newton_max_iterations,
    )
    energy, _, _, _ = hedgehog_energy_gradient_hessian(
        final_profile,
        radius=radius,
        dr=dr,
    )
    total_scale = max(energy.total, 1.0e-300)
    derrick_first, derrick_second = _derrick_scale_variations(
        final_profile,
        radius=radius,
        dr=dr,
    )
    return SkyrmeHedgehogSolution(
        radius=float(radius),
        dr=float(dr),
        r=_radial_nodes(radius, dr),
        profile=final_profile,
        energy=energy,
        degree=hedgehog_degree(
            final_profile,
            radius=radius,
            dr=dr,
        ),
        rms_topological_radius=_topological_rms_radius(
            final_profile,
            radius=radius,
            dr=dr,
        ),
        stationary_relative_residual=_stationary_residual(
            final_profile,
            radius=radius,
            dr=dr,
        ),
        unitarity_residual=hedgehog_unitarity_residual(final_profile),
        derrick_relative_first_derivative=derrick_first,
        derrick_relative_second_derivative=derrick_second,
        continuum_virial_mismatch=float(abs(energy.sigma - energy.skyrme) / total_scale),
        optimizer_converged=bool(result.success),
        optimizer_iterations=int(result.nit),
        newton_converged=bool(newton_converged),
        newton_iterations=int(newton_iterations),
        message=(f"optimizer: {result.message}; Newton: {newton_message}"),
    )
