"""Fixed-Noether-charge radial solitons of the bare LFM action.

This module is deliberately particle-name agnostic. A converged result is a
scalar soliton candidate, not an electron. The radial solver is a continuum
discovery and refinement tool; positive candidates still require confirmation
with the canonical three-dimensional 19-point operator and live evolution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, root
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from lfm.constants import CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import laplacian_19pt

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class RadialNoetherEnergy:
    """Energy ledger for one radial fixed-charge configuration."""

    total: float
    temporal: float
    matter_gradient: float
    matter_mass: float
    chi_gradient: float
    chi_potential: float
    norm: float
    omega: float
    energy_per_charge: float


@dataclass(frozen=True)
class RadialNoetherSolution:
    """One optimized radial fixed-Noether-charge configuration."""

    radius: float
    dx: float
    r: np.ndarray
    phi: np.ndarray
    chi: np.ndarray
    target_charge: float
    charge: float
    energy: RadialNoetherEnergy
    rms_radius: float
    half_charge_radius: float
    stationary_relative_residual: float
    phi_relative_residual: float
    chi_relative_residual: float
    optimizer_converged: bool
    optimizer_status: int
    optimizer_iterations: int
    polisher_converged: bool
    polisher_iterations: int
    sparse_polisher_converged: bool
    sparse_polisher_iterations: int
    message: str


@dataclass(frozen=True)
class RadialNoetherSweepResult:
    """One labeled case from a radial solver sweep."""

    case_id: str
    solution: RadialNoetherSolution


@dataclass(frozen=True)
class CartesianNoetherState:
    """Complete two-layer leapfrog state for one lifted scalar candidate."""

    psi_real: np.ndarray
    psi_real_prev: np.ndarray
    psi_imag: np.ndarray
    psi_imag_prev: np.ndarray
    chi: np.ndarray
    chi_prev: np.ndarray
    center: tuple[float, float, float]
    velocity: tuple[float, float, float]
    omega: float
    dx: float
    dt: float


@dataclass(frozen=True)
class CartesianFixedChargeEnergy:
    """Energy ledger for one Cartesian fixed-charge configuration."""

    total: float
    temporal: float
    matter_gradient: float
    matter_mass: float
    chi_gradient: float
    chi_potential: float
    norm: float
    omega: float
    energy_per_charge: float


@dataclass(frozen=True)
class CartesianNoetherSolution:
    """One optimized Cartesian fixed-Noether-charge configuration."""

    phi: np.ndarray
    chi: np.ndarray
    target_charge: float
    charge: float
    dx: float
    energy: CartesianFixedChargeEnergy
    stationary_relative_residual: float
    phi_relative_residual: float
    chi_relative_residual: float
    optimizer_converged: bool
    optimizer_status: int
    optimizer_iterations: int
    function_evaluations: int
    message: str


def _case_float(case: dict[str, object], key: str, default: float | None = None) -> float:
    value = case[key] if default is None else case.get(key, default)
    return float(cast("float | int | str", value))


def _case_int(case: dict[str, object], key: str, default: int) -> int:
    return int(cast("float | int | str", case.get(key, default)))


def _tuple3_float(values: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    items = tuple(float(value) for value in values)
    if len(items) != 3:
        raise ValueError("expected a 3-vector")
    return cast("tuple[float, float, float]", items)


def radial_shell_geometry(radius: float, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell centers, shell volumes, and radial face areas."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    cells_float = radius / dx
    cells = int(round(cells_float))
    if cells < 8 or not np.isclose(cells * dx, radius, rtol=0.0, atol=1.0e-12):
        raise ValueError("radius/dx must be an integer of at least 8")
    faces = np.arange(cells + 1, dtype=np.float64) * dx
    centers = (np.arange(cells, dtype=np.float64) + 0.5) * dx
    volumes = (4.0 * np.pi / 3.0) * (faces[1:] ** 3 - faces[:-1] ** 3)
    face_areas = 4.0 * np.pi * faces**2
    return centers, volumes, face_areas


def _edge_energy_and_gradient(
    field: np.ndarray,
    face_areas: np.ndarray,
    dx: float,
    outer_boundary: float,
) -> tuple[float, np.ndarray]:
    gradient = np.zeros_like(field)
    energy = 0.0
    if field.size > 1:
        conductance = face_areas[1:-1] / dx
        differences = field[1:] - field[:-1]
        energy += 0.5 * float(np.sum(conductance * differences * differences))
        edge_force = conductance * differences
        gradient[:-1] -= edge_force
        gradient[1:] += edge_force

    outer_conductance = face_areas[-1] / (0.5 * dx)
    outer_difference = field[-1] - outer_boundary
    energy += 0.5 * outer_conductance * outer_difference * outer_difference
    gradient[-1] += outer_conductance * outer_difference
    return energy, gradient


def _edge_hessian(
    cells: int,
    face_areas: np.ndarray,
    dx: float,
) -> sp.csr_matrix:
    diagonal = np.zeros(cells, dtype=np.float64)
    off_diagonal = np.zeros(max(cells - 1, 0), dtype=np.float64)
    if cells > 1:
        conductance = face_areas[1:-1] / dx
        diagonal[:-1] += conductance
        diagonal[1:] += conductance
        off_diagonal[:] = -conductance
    diagonal[-1] += face_areas[-1] / (0.5 * dx)
    return sp.diags(
        (off_diagonal, diagonal, off_diagonal),
        offsets=(-1, 0, 1),
        shape=(cells, cells),
        format="csr",
    )


def radial_fixed_charge_energy_and_gradient(
    variables: np.ndarray,
    *,
    target_charge: float,
    radius: float,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
) -> tuple[RadialNoetherEnergy, np.ndarray]:
    """Evaluate the canonical fixed-charge energy and analytic gradient."""
    if target_charge <= 0.0:
        raise ValueError("target_charge must be positive")
    if chi0 <= 0.0 or kappa <= 0.0 or lambda_h <= 0.0:
        raise ValueError("canonical couplings must be positive")

    _, volumes, face_areas = radial_shell_geometry(radius, dx)
    cells = volumes.size
    values = np.asarray(variables, dtype=np.float64)
    if values.shape != (2 * cells,):
        raise ValueError(f"variables must have shape {(2 * cells,)}")
    phi = values[:cells]
    chi = values[cells:]
    norm = float(np.dot(volumes, phi * phi))
    if not np.isfinite(norm) or norm <= 1.0e-300:
        raise ValueError("matter norm must be finite and positive")

    b_value = chi0 / kappa
    omega = target_charge / norm
    temporal = target_charge * target_charge / (2.0 * norm)
    matter_gradient, grad_phi_edges = _edge_energy_and_gradient(
        phi,
        face_areas,
        dx,
        0.0,
    )
    chi_gradient_raw, grad_chi_edges = _edge_energy_and_gradient(
        chi,
        face_areas,
        dx,
        chi0,
    )
    matter_mass = 0.5 * float(np.dot(volumes, chi * chi * phi * phi))
    potential_density = (chi * chi - chi0 * chi0) ** 2
    chi_potential = b_value * lambda_h * float(np.dot(volumes, potential_density))
    chi_gradient = b_value * chi_gradient_raw
    total = temporal + matter_gradient + matter_mass + chi_gradient + chi_potential

    grad_phi = grad_phi_edges + volumes * chi * chi * phi - omega * omega * volumes * phi
    grad_chi = (
        b_value * grad_chi_edges
        + volumes * chi * phi * phi
        + 4.0 * b_value * lambda_h * volumes * chi * (chi * chi - chi0 * chi0)
    )
    gradient = np.concatenate((grad_phi, grad_chi))
    ledger = RadialNoetherEnergy(
        total=total,
        temporal=temporal,
        matter_gradient=matter_gradient,
        matter_mass=matter_mass,
        chi_gradient=chi_gradient,
        chi_potential=chi_potential,
        norm=norm,
        omega=omega,
        energy_per_charge=total / target_charge,
    )
    return ledger, gradient


def radial_fixed_charge_hessian(
    variables: np.ndarray,
    *,
    target_charge: float,
    radius: float,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
) -> sp.csr_matrix:
    """Return the analytic Hessian of the reduced fixed-charge energy."""
    _, volumes, face_areas = radial_shell_geometry(radius, dx)
    cells = volumes.size
    values = np.asarray(variables, dtype=np.float64)
    if values.shape != (2 * cells,):
        raise ValueError(f"variables must have shape {(2 * cells,)}")
    phi = values[:cells]
    chi = values[cells:]
    norm = float(np.dot(volumes, phi * phi))
    if not np.isfinite(norm) or norm <= 1.0e-300:
        raise ValueError("matter norm must be finite and positive")

    b_value = chi0 / kappa
    omega = target_charge / norm
    edge_hessian = _edge_hessian(cells, face_areas, dx)
    volume_phi = volumes * phi
    charge_rank_one = (4.0 * omega * omega / norm) * np.outer(
        volume_phi,
        volume_phi,
    )
    phi_block = (
        edge_hessian
        + sp.diags(volumes * (chi * chi - omega * omega), format="csr")
        + sp.csr_matrix(charge_rank_one)
    )
    cross_block = sp.diags(2.0 * volumes * chi * phi, format="csr")
    chi_diagonal = volumes * (
        phi * phi + 4.0 * b_value * lambda_h * (3.0 * chi * chi - chi0 * chi0)
    )
    chi_block = b_value * edge_hessian + sp.diags(
        chi_diagonal,
        format="csr",
    )
    return sp.bmat(
        (
            (phi_block, cross_block),
            (cross_block, chi_block),
        ),
        format="csr",
    )


def make_radial_bag_guess(
    *,
    target_charge: float,
    radius: float,
    dx: float,
    core_radius: float,
    omega_guess: float,
    chi_depth_fraction: float = 0.9,
    chi0: float = CHI0,
) -> np.ndarray:
    """Construct an unsupported smooth radial guess with the requested charge."""
    if not 0.0 < core_radius < radius:
        raise ValueError("core_radius must lie inside the domain")
    if not 0.0 < omega_guess < chi0:
        raise ValueError("omega_guess must lie between zero and chi0")
    if not 0.0 < chi_depth_fraction < 2.0:
        raise ValueError("chi_depth_fraction must lie in (0, 2)")
    r, volumes, _ = radial_shell_geometry(radius, dx)

    envelope = np.exp(-0.5 * (r / core_radius) ** 4)
    target_norm = target_charge / omega_guess
    amplitude = np.sqrt(target_norm / float(np.dot(volumes, envelope * envelope)))
    phi = amplitude * envelope
    chi = chi0 * (1.0 - chi_depth_fraction * np.exp(-0.5 * (r / core_radius) ** 4))
    return np.concatenate((phi, chi))


def prolong_radial_fields(
    *,
    radius: float,
    source_r: np.ndarray,
    phi: np.ndarray,
    chi: np.ndarray,
    new_dx: float,
    chi0: float = CHI0,
) -> np.ndarray:
    """Prolong cell-centered radial fields without clipping or resetting."""
    old_r = np.asarray(source_r, dtype=np.float64)
    old_phi = np.asarray(phi, dtype=np.float64)
    old_chi = np.asarray(chi, dtype=np.float64)
    if old_r.ndim != 1 or old_r.size < 2:
        raise ValueError("source_r must be a one-dimensional radial grid")
    if old_phi.shape != old_r.shape or old_chi.shape != old_r.shape:
        raise ValueError("source fields must match source_r")
    if not np.all(np.diff(old_r) > 0.0):
        raise ValueError("source_r must be strictly increasing")
    new_r, _, _ = radial_shell_geometry(radius, new_dx)
    prolonged_phi = np.interp(
        new_r,
        old_r,
        old_phi,
        left=float(old_phi[0]),
        right=0.0,
    )
    prolonged_chi = np.interp(
        new_r,
        old_r,
        old_chi,
        left=float(old_chi[0]),
        right=chi0,
    )
    return np.concatenate((prolonged_phi, prolonged_chi))


def prolong_radial_solution(
    solution: RadialNoetherSolution,
    *,
    new_dx: float,
    chi0: float = CHI0,
) -> np.ndarray:
    """Prolong a cell-centered radial solution without clipping or resetting."""
    return prolong_radial_fields(
        radius=solution.radius,
        source_r=solution.r,
        phi=solution.phi,
        chi=solution.chi,
        new_dx=new_dx,
        chi0=chi0,
    )


def _stationary_residuals(
    phi: np.ndarray,
    chi: np.ndarray,
    *,
    target_charge: float,
    radius: float,
    dx: float,
    chi0: float,
    kappa: float,
    lambda_h: float,
) -> tuple[float, float, float]:
    variables = np.concatenate((phi, chi))
    ledger, gradient = radial_fixed_charge_energy_and_gradient(
        variables,
        target_charge=target_charge,
        radius=radius,
        dx=dx,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
    )
    _, volumes, _ = radial_shell_geometry(radius, dx)
    cells = phi.size
    phi_equation = gradient[:cells] / volumes
    b_value = chi0 / kappa
    chi_equation = gradient[cells:] / (b_value * volumes)

    phi_scale = max(
        float(np.sqrt(np.dot(volumes, (ledger.omega * ledger.omega * phi) ** 2))),
        1.0e-300,
    )
    phi_residual = float(np.sqrt(np.dot(volumes, phi_equation * phi_equation))) / phi_scale
    chi_scale_field = (
        4.0 * lambda_h * chi * (chi * chi - chi0 * chi0) + (kappa / chi0) * chi * phi * phi
    )
    chi_scale = max(
        float(np.sqrt(np.dot(volumes, chi_scale_field * chi_scale_field))),
        4.0 * lambda_h * chi0**3 * np.sqrt(float(np.sum(volumes))) * 1.0e-12,
    )
    chi_residual = float(np.sqrt(np.dot(volumes, chi_equation * chi_equation))) / chi_scale
    return max(phi_residual, chi_residual), phi_residual, chi_residual


def sparse_newton_polish_radial(
    variables: np.ndarray,
    *,
    target_charge: float,
    radius: float,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
    residual_tolerance: float = 1.0e-9,
    max_iterations: int = 80,
) -> tuple[np.ndarray, bool, int, str]:
    """Damped sparse Newton polish of the same fixed-charge energy."""
    current = np.asarray(variables, dtype=np.float64).copy()
    cells = current.size // 2
    if current.shape != (2 * cells,) or cells < 8:
        raise ValueError("variables must contain two radial fields")

    for iteration in range(max_iterations + 1):
        residual, _, _ = _stationary_residuals(
            current[:cells],
            current[cells:],
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        if residual <= residual_tolerance:
            return current, True, iteration, "stationary residual converged"
        if iteration == max_iterations:
            break

        ledger, gradient = radial_fixed_charge_energy_and_gradient(
            current,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        hessian = radial_fixed_charge_hessian(
            current,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        diagonal_scale = np.maximum(np.abs(hessian.diagonal()), 1.0)
        accepted = False
        for damping in (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
            system = (
                hessian
                if damping == 0.0
                else hessian + sp.diags(damping * diagonal_scale, format="csr")
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    delta = spsolve(system, -gradient)
                except (MatrixRankWarning, RuntimeError, ValueError):
                    continue
            if not np.all(np.isfinite(delta)):
                continue

            step = 1.0
            while step >= 1.0e-10:
                candidate = current + step * delta
                try:
                    candidate_ledger, _ = radial_fixed_charge_energy_and_gradient(
                        candidate,
                        target_charge=target_charge,
                        radius=radius,
                        dx=dx,
                        chi0=chi0,
                        kappa=kappa,
                        lambda_h=lambda_h,
                    )
                    candidate_residual, _, _ = _stationary_residuals(
                        candidate[:cells],
                        candidate[cells:],
                        target_charge=target_charge,
                        radius=radius,
                        dx=dx,
                        chi0=chi0,
                        kappa=kappa,
                        lambda_h=lambda_h,
                    )
                except ValueError:
                    step *= 0.5
                    continue
                energy_ok = candidate_ledger.total <= ledger.total * (1.0 + 1.0e-13)
                if candidate_residual < residual and energy_ok:
                    current = candidate
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                break
        if not accepted:
            return (
                current,
                False,
                iteration,
                "damped sparse Newton line search failed",
            )
    return current, False, max_iterations, "sparse Newton iteration limit reached"


def solve_radial_noether_soliton(
    *,
    target_charge: float,
    radius: float,
    dx: float,
    core_radius: float,
    omega_guess: float,
    chi_depth_fraction: float = 0.9,
    initial_variables: np.ndarray | None = None,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
    max_iterations: int = 2000,
    gradient_tolerance: float = 1.0e-9,
    polish_tolerance: float = 1.0e-11,
    sparse_polish: bool = True,
    sparse_polish_tolerance: float = 1.0e-9,
    sparse_polish_max_iterations: int = 80,
) -> RadialNoetherSolution:
    """Minimize the canonical radial Hamiltonian at fixed Noether charge.

    No field is clipped or renormalized during optimization. The charge
    constraint is represented by the exact reduced term Q^2/(2*N), where
    N is the spatial matter norm.
    """
    r, volumes, _ = radial_shell_geometry(radius, dx)
    if initial_variables is None:
        initial = make_radial_bag_guess(
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            core_radius=core_radius,
            omega_guess=omega_guess,
            chi_depth_fraction=chi_depth_fraction,
            chi0=chi0,
        )
    else:
        initial = np.asarray(initial_variables, dtype=np.float64).copy()
        if initial.shape != (2 * r.size,):
            raise ValueError("initial_variables has the wrong shape")

    free_energy = target_charge * chi0
    cells = r.size
    phi_scale = max(float(np.max(np.abs(initial[:cells]))), 1.0)
    chi_scale = chi0
    variable_scale = np.concatenate(
        (
            np.full(cells, phi_scale, dtype=np.float64),
            np.full(cells, chi_scale, dtype=np.float64),
        )
    )
    scaled_initial = initial / variable_scale

    def objective(scaled_values: np.ndarray) -> tuple[float, np.ndarray]:
        values = scaled_values * variable_scale
        ledger, gradient = radial_fixed_charge_energy_and_gradient(
            values,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        return ledger.total / free_energy, gradient * variable_scale / free_energy

    result = minimize(
        objective,
        scaled_initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1.0e-14,
            "maxls": 50,
            "maxcor": 30,
        },
    )
    optimized = np.asarray(result.x, dtype=np.float64) * variable_scale
    optimized_ledger, _ = radial_fixed_charge_energy_and_gradient(
        optimized,
        target_charge=target_charge,
        radius=radius,
        dx=dx,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
    )
    optimized_residual, _, _ = _stationary_residuals(
        optimized[:cells],
        optimized[cells:],
        target_charge=target_charge,
        radius=radius,
        dx=dx,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
    )

    polish_scale = np.concatenate(
        (
            np.full(
                cells,
                max(float(np.max(np.abs(optimized[:cells]))), 1.0),
                dtype=np.float64,
            ),
            np.full(cells, chi0, dtype=np.float64),
        )
    )

    def stationarity(scaled_values: np.ndarray) -> np.ndarray:
        values = scaled_values * polish_scale
        _, gradient = radial_fixed_charge_energy_and_gradient(
            values,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        return gradient * polish_scale / free_energy

    polisher_success = False
    polisher_iterations = 0
    polisher_message = "not run"
    polished_values = optimized.copy()
    try:
        polished = root(
            stationarity,
            optimized / polish_scale,
            method="krylov",
            options={
                "fatol": float(polish_tolerance),
                "maxiter": min(int(max_iterations), 500),
                "disp": False,
            },
        )
        polished_values = np.asarray(polished.x, dtype=np.float64) * polish_scale
        polisher_success = bool(polished.success)
        polisher_iterations = int(getattr(polished, "nit", 0))
        polisher_message = str(polished.message)
    except (ArithmeticError, RuntimeError, ValueError) as error:
        polisher_message = f"{type(error).__name__}: {error}"

    use_polished = False
    if np.all(np.isfinite(polished_values)):
        try:
            polished_ledger, _ = radial_fixed_charge_energy_and_gradient(
                polished_values,
                target_charge=target_charge,
                radius=radius,
                dx=dx,
                chi0=chi0,
                kappa=kappa,
                lambda_h=lambda_h,
            )
            polished_residual, _, _ = _stationary_residuals(
                polished_values[:cells],
                polished_values[cells:],
                target_charge=target_charge,
                radius=radius,
                dx=dx,
                chi0=chi0,
                kappa=kappa,
                lambda_h=lambda_h,
            )
            use_polished = bool(
                polished_residual < optimized_residual
                and polished_ledger.total <= optimized_ledger.total * (1.0 + 1.0e-9)
            )
        except ValueError:
            use_polished = False

    final_values = polished_values if use_polished else optimized
    sparse_polisher_converged = False
    sparse_polisher_iterations = 0
    sparse_polisher_message = "not run"
    if sparse_polish:
        (
            sparse_values,
            sparse_polisher_converged,
            sparse_polisher_iterations,
            sparse_polisher_message,
        ) = sparse_newton_polish_radial(
            final_values,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
            residual_tolerance=sparse_polish_tolerance,
            max_iterations=sparse_polish_max_iterations,
        )
        sparse_residual, _, _ = _stationary_residuals(
            sparse_values[:cells],
            sparse_values[cells:],
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        current_residual, _, _ = _stationary_residuals(
            final_values[:cells],
            final_values[cells:],
            target_charge=target_charge,
            radius=radius,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        if sparse_residual < current_residual:
            final_values = sparse_values
    phi = final_values[:cells]
    chi = final_values[cells:]
    ledger, _ = radial_fixed_charge_energy_and_gradient(
        final_values,
        target_charge=target_charge,
        radius=radius,
        dx=dx,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
    )
    residual, phi_residual, chi_residual = _stationary_residuals(
        phi,
        chi,
        target_charge=target_charge,
        radius=radius,
        dx=dx,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
    )
    density_weight = volumes * phi * phi
    cumulative = np.cumsum(density_weight)
    half_index = int(np.searchsorted(cumulative, 0.5 * ledger.norm, side="left"))
    half_index = min(half_index, r.size - 1)
    rms_radius = np.sqrt(float(np.dot(density_weight, r * r)) / ledger.norm)
    charge = ledger.omega * ledger.norm
    return RadialNoetherSolution(
        radius=float(radius),
        dx=float(dx),
        r=r,
        phi=phi,
        chi=chi,
        target_charge=float(target_charge),
        charge=float(charge),
        energy=ledger,
        rms_radius=float(rms_radius),
        half_charge_radius=float(r[half_index]),
        stationary_relative_residual=float(residual),
        phi_relative_residual=float(phi_residual),
        chi_relative_residual=float(chi_residual),
        optimizer_converged=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_iterations=int(result.nit),
        polisher_converged=bool(polisher_success and use_polished),
        polisher_iterations=polisher_iterations,
        sparse_polisher_converged=bool(sparse_polisher_converged),
        sparse_polisher_iterations=int(sparse_polisher_iterations),
        message=(
            f"optimizer: {result.message}; polisher: {polisher_message}; "
            f"polished_result_used={use_polished}; "
            f"sparse_polisher: {sparse_polisher_message}"
        ),
    )


def sweep_radial_noether_solitons(
    cases: list[dict[str, object]],
) -> list[RadialNoetherSweepResult]:
    """Run a declared radial solver case list through one library entry point."""
    results: list[RadialNoetherSweepResult] = []
    required = {
        "case_id",
        "target_charge",
        "radius",
        "dx",
        "core_radius",
        "omega_guess",
    }
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"case is missing keys: {sorted(missing)}")
        initial_value = case.get("initial_variables")
        initial_variables = (
            None if initial_value is None else np.asarray(initial_value, dtype=np.float64)
        )
        solution = solve_radial_noether_soliton(
            target_charge=_case_float(case, "target_charge"),
            radius=_case_float(case, "radius"),
            dx=_case_float(case, "dx"),
            core_radius=_case_float(case, "core_radius"),
            omega_guess=_case_float(case, "omega_guess"),
            chi_depth_fraction=_case_float(case, "chi_depth_fraction", 0.9),
            initial_variables=initial_variables,
            max_iterations=_case_int(case, "max_iterations", 2000),
            gradient_tolerance=_case_float(case, "gradient_tolerance", 1.0e-9),
            polish_tolerance=_case_float(case, "polish_tolerance", 1.0e-11),
            sparse_polish=bool(case.get("sparse_polish", True)),
            sparse_polish_tolerance=_case_float(case, "sparse_polish_tolerance", 1.0e-9),
            sparse_polish_max_iterations=_case_int(case, "sparse_polish_max_iterations", 80),
        )
        results.append(
            RadialNoetherSweepResult(
                case_id=str(case["case_id"]),
                solution=solution,
            )
        )
    return results


def _cartesian_axis(grid_size: int, dx: float) -> np.ndarray:
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    return (np.arange(grid_size, dtype=np.float64) - 0.5 * (grid_size - 1)) * dx


def _interpolate_radial_profile(
    radius_grid: np.ndarray,
    source_r: np.ndarray,
    values: np.ndarray,
    outer_value: float,
) -> np.ndarray:
    return np.interp(
        radius_grid.ravel(),
        source_r,
        values,
        left=float(values[0]),
        right=float(outer_value),
    ).reshape(radius_grid.shape)


def lift_radial_noether_state(
    *,
    source_r: np.ndarray,
    phi: np.ndarray,
    chi: np.ndarray,
    omega: float,
    grid_size: int,
    dx: float,
    dt: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    perturbation_seed: int | None = None,
    perturbation_amplitude: float = 0.0,
    chi0: float = CHI0,
    dtype: np.dtype | type = np.float32,
) -> CartesianNoetherState:
    """Lift a radial relative equilibrium into full 3D leapfrog data.

    The only supported boost is along one coordinate axis. The current and
    previous layers are evaluated from the continuum Lorentz-coordinate
    transform of the same scalar profile. No clipping, projection, charge
    normalization, or post-processing is applied.
    """
    source_r = np.asarray(source_r, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    if source_r.ndim != 1 or source_r.size < 2:
        raise ValueError("source_r must be one-dimensional")
    if phi.shape != source_r.shape or chi.shape != source_r.shape:
        raise ValueError("radial fields must match source_r")
    if not np.all(np.diff(source_r) > 0.0):
        raise ValueError("source_r must be strictly increasing")
    if omega <= 0.0 or dt <= 0.0:
        raise ValueError("omega and dt must be positive")
    if perturbation_amplitude < 0.0:
        raise ValueError("perturbation_amplitude must be non-negative")

    velocity_array = np.asarray(velocity, dtype=np.float64)
    speed_sq = float(np.dot(velocity_array, velocity_array))
    if speed_sq >= 1.0:
        raise ValueError("boost speed must be below the LFM wave speed")
    nonzero_axes = np.flatnonzero(np.abs(velocity_array) > 1.0e-15)
    if nonzero_axes.size > 1:
        raise ValueError("only an axis-aligned boost is supported")

    axis = _cartesian_axis(grid_size, dx)
    x = axis[:, None, None] - float(center[0])
    y = axis[None, :, None] - float(center[1])
    z = axis[None, None, :] - float(center[2])
    gamma = 1.0 / np.sqrt(1.0 - speed_sq)

    current_components = [x, y, z]
    previous_components = [x, y, z]
    phase_current = np.zeros(
        (grid_size, grid_size, grid_size),
        dtype=np.float64,
    )
    phase_previous = np.full_like(phase_current, -omega * gamma * dt)

    if nonzero_axes.size == 1:
        boost_axis = int(nonzero_axes[0])
        speed = float(velocity_array[boost_axis])
        coordinate = current_components[boost_axis]
        current_components[boost_axis] = gamma * coordinate
        previous_components[boost_axis] = gamma * (coordinate + speed * dt)
        phase_current = -omega * gamma * speed * coordinate
        phase_previous = -omega * gamma * (dt + speed * coordinate)

    radius_current = np.sqrt(
        current_components[0] ** 2 + current_components[1] ** 2 + current_components[2] ** 2
    )
    radius_previous = np.sqrt(
        previous_components[0] ** 2 + previous_components[1] ** 2 + previous_components[2] ** 2
    )
    phi_current = _interpolate_radial_profile(
        radius_current,
        source_r,
        phi,
        0.0,
    )
    phi_previous = _interpolate_radial_profile(
        radius_previous,
        source_r,
        phi,
        0.0,
    )
    chi_current = _interpolate_radial_profile(
        radius_current,
        source_r,
        chi,
        chi0,
    )
    chi_previous = _interpolate_radial_profile(
        radius_previous,
        source_r,
        chi,
        chi0,
    )

    if perturbation_seed is not None and perturbation_amplitude > 0.0:
        if speed_sq > 0.0:
            raise ValueError("seeded perturbation and boost are separate S5 cases")
        rng = np.random.default_rng(perturbation_seed)
        coefficients = rng.normal(size=4)
        coefficients /= np.sum(np.abs(coefficients))
        half_peak_index = int(np.argmax(np.abs(phi) < 0.5 * np.max(np.abs(phi))))
        scale = max(float(source_r[half_peak_index]), dx)
        radius_sq = x * x + y * y + z * z
        mode = (
            coefficients[0] * np.tanh(x / scale)
            + coefficients[1] * np.tanh(y / scale)
            + coefficients[2] * np.tanh(z / scale)
            + coefficients[3] * (x * x - y * y) / (radius_sq + scale * scale)
        )
        matter_factor = 1.0 + perturbation_amplitude * mode
        chi_factor = 1.0 - perturbation_amplitude * mode
        phi_current *= matter_factor
        phi_previous *= matter_factor
        chi_current = chi0 + (chi_current - chi0) * chi_factor
        chi_previous = chi0 + (chi_previous - chi0) * chi_factor

    state_dtype = np.dtype(dtype)
    return CartesianNoetherState(
        psi_real=(phi_current * np.cos(phase_current)).astype(state_dtype),
        psi_real_prev=(phi_previous * np.cos(phase_previous)).astype(state_dtype),
        psi_imag=(phi_current * np.sin(phase_current)).astype(state_dtype),
        psi_imag_prev=(phi_previous * np.sin(phase_previous)).astype(state_dtype),
        chi=chi_current.astype(state_dtype),
        chi_prev=chi_previous.astype(state_dtype),
        center=_tuple3_float(center),
        velocity=_tuple3_float(velocity),
        omega=float(omega),
        dx=float(dx),
        dt=float(dt),
    )


def cartesian_fixed_charge_energy_and_gradient(
    variables: np.ndarray,
    *,
    target_charge: float,
    grid_size: int,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
) -> tuple[CartesianFixedChargeEnergy, np.ndarray]:
    """Evaluate the periodic 3D fixed-charge energy and analytic gradient."""
    if target_charge <= 0.0:
        raise ValueError("target_charge must be positive")
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    if chi0 <= 0.0 or kappa <= 0.0 or lambda_h <= 0.0:
        raise ValueError("canonical couplings must be positive")

    sites = grid_size**3
    values = np.asarray(variables, dtype=np.float64)
    if values.shape != (2 * sites,):
        raise ValueError(f"variables must have shape {(2 * sites,)}")
    phi = values[:sites].reshape((grid_size,) * 3)
    chi = values[sites:].reshape((grid_size,) * 3)
    volume = dx**3
    inv_dx2 = 1.0 / (dx * dx)
    b_value = chi0 / kappa

    norm = volume * float(np.sum(phi * phi))
    if not np.isfinite(norm) or norm <= 1.0e-300:
        raise ValueError("matter norm must be finite and positive")
    omega = target_charge / norm
    temporal = target_charge * target_charge / (2.0 * norm)

    lap_phi = laplacian_19pt(phi)
    lap_chi = laplacian_19pt(chi)
    matter_gradient = -0.5 * volume * inv_dx2 * float(np.sum(phi * lap_phi))
    matter_mass = 0.5 * volume * float(np.sum(chi * chi * phi * phi))
    chi_gradient = -0.5 * b_value * volume * inv_dx2 * float(np.sum(chi * lap_chi))
    chi_potential = b_value * lambda_h * volume * float(np.sum((chi * chi - chi0 * chi0) ** 2))
    total = temporal + matter_gradient + matter_mass + chi_gradient + chi_potential

    grad_phi = volume * (-inv_dx2 * lap_phi + (chi * chi - omega * omega) * phi)
    grad_chi = volume * (
        chi * phi * phi
        + b_value * (-inv_dx2 * lap_chi + 4.0 * lambda_h * chi * (chi * chi - chi0 * chi0))
    )
    gradient = np.concatenate((grad_phi.ravel(), grad_chi.ravel()))
    ledger = CartesianFixedChargeEnergy(
        total=total,
        temporal=temporal,
        matter_gradient=matter_gradient,
        matter_mass=matter_mass,
        chi_gradient=chi_gradient,
        chi_potential=chi_potential,
        norm=norm,
        omega=omega,
        energy_per_charge=total / target_charge,
    )
    return ledger, gradient


def solve_cartesian_noether_soliton(
    *,
    initial_phi: np.ndarray,
    initial_chi: np.ndarray,
    target_charge: float,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
    max_iterations: int = 1000,
    history: int = 10,
    gradient_tolerance: float = 1.0e-10,
    function_tolerance: float = np.finfo(np.float64).eps,
    progress_interval: int = 10,
    progress_callback: Callable[
        [int, CartesianFixedChargeEnergy, float],
        None,
    ]
    | None = None,
) -> CartesianNoetherSolution:
    """Minimize the exact periodic 3D Hamiltonian at fixed Noether charge.

    Optimization coordinates and objective units are rescaled only for
    numerical conditioning. No physical field is clipped, normalized,
    projected, or reset.
    """
    phi0 = np.asarray(initial_phi, dtype=np.float64)
    chi_initial = np.asarray(initial_chi, dtype=np.float64)
    if phi0.shape != chi_initial.shape or phi0.ndim != 3:
        raise ValueError("initial_phi and initial_chi must be matching 3D arrays")
    if len(set(phi0.shape)) != 1:
        raise ValueError("Cartesian fixed-charge solve requires a cubic grid")
    if not np.all(np.isfinite(phi0)) or not np.all(np.isfinite(chi_initial)):
        raise ValueError("initial fields must be finite")
    if max_iterations <= 0 or history <= 0 or progress_interval <= 0:
        raise ValueError("optimizer iteration and history limits must be positive")

    grid_size = phi0.shape[0]
    sites = grid_size**3
    matter_scale = max(float(np.max(np.abs(phi0))), 1.0)
    chi_scale = float(chi0)
    objective_scale = float(target_charge)
    scaled_initial = np.concatenate(
        (
            (phi0 / matter_scale).ravel(),
            (chi_initial / chi_scale).ravel(),
        )
    )

    def unpack(scaled: np.ndarray) -> np.ndarray:
        scaled64 = np.asarray(scaled, dtype=np.float64)
        return np.concatenate(
            (
                matter_scale * scaled64[:sites],
                chi_scale * scaled64[sites:],
            )
        )

    def objective(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        ledger, raw_gradient = cartesian_fixed_charge_energy_and_gradient(
            unpack(scaled),
            target_charge=target_charge,
            grid_size=grid_size,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        scaled_gradient = np.concatenate(
            (
                matter_scale * raw_gradient[:sites],
                chi_scale * raw_gradient[sites:],
            )
        )
        return (
            ledger.total / objective_scale,
            scaled_gradient / objective_scale,
        )

    callback_iterations = 0

    def callback(scaled: np.ndarray) -> None:
        nonlocal callback_iterations
        callback_iterations += 1
        if progress_callback is None or (
            callback_iterations != 1 and callback_iterations % progress_interval != 0
        ):
            return
        raw = unpack(scaled)
        ledger, _ = cartesian_fixed_charge_energy_and_gradient(
            raw,
            target_charge=target_charge,
            grid_size=grid_size,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )
        phi = raw[:sites].reshape((grid_size,) * 3)
        chi = raw[sites:].reshape((grid_size,) * 3)
        residual = cartesian_stationary_residual(
            phi,
            chi,
            omega=ledger.omega,
            dx=dx,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
        )[0]
        progress_callback(callback_iterations, ledger, residual)

    result = minimize(
        objective,
        scaled_initial,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": int(max_iterations),
            "maxcor": int(history),
            "gtol": float(gradient_tolerance),
            "ftol": float(function_tolerance),
            "maxls": 40,
        },
    )
    final_raw = unpack(np.asarray(result.x, dtype=np.float64))
    phi = final_raw[:sites].reshape((grid_size,) * 3)
    chi = final_raw[sites:].reshape((grid_size,) * 3)
    ledger, _ = cartesian_fixed_charge_energy_and_gradient(
        final_raw,
        target_charge=target_charge,
        grid_size=grid_size,
        dx=dx,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
    )
    residual, phi_residual, chi_residual = cartesian_stationary_residual(
        phi,
        chi,
        omega=ledger.omega,
        dx=dx,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
    )
    return CartesianNoetherSolution(
        phi=phi,
        chi=chi,
        target_charge=float(target_charge),
        charge=float(ledger.omega * ledger.norm),
        dx=float(dx),
        energy=ledger,
        stationary_relative_residual=float(residual),
        phi_relative_residual=float(phi_residual),
        chi_relative_residual=float(chi_residual),
        optimizer_converged=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        message=str(result.message),
    )


def cartesian_stationary_residual(
    phi: np.ndarray,
    chi: np.ndarray,
    *,
    omega: float,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
) -> tuple[float, float, float]:
    """Return exact 19-point residuals of a 3D scalar relative equilibrium."""
    phi64 = np.asarray(phi, dtype=np.float64)
    chi64 = np.asarray(chi, dtype=np.float64)
    if phi64.shape != chi64.shape or phi64.ndim != 3:
        raise ValueError("phi and chi must be matching 3D arrays")
    if omega <= 0.0 or dx <= 0.0:
        raise ValueError("omega and dx must be positive")
    inv_dx2 = 1.0 / (dx * dx)
    phi_equation = inv_dx2 * laplacian_19pt(phi64) + (omega * omega - chi64 * chi64) * phi64
    chi_equation = (
        inv_dx2 * laplacian_19pt(chi64)
        - 4.0 * lambda_h * chi64 * (chi64 * chi64 - chi0 * chi0)
        - (kappa / chi0) * chi64 * phi64 * phi64
    )
    volume = dx**3
    phi_scale_field = np.maximum(
        np.abs(omega * omega * phi64),
        np.abs(chi64 * chi64 * phi64),
    )
    chi_scale_field = np.abs(4.0 * lambda_h * chi64 * (chi64 * chi64 - chi0 * chi0)) + np.abs(
        (kappa / chi0) * chi64 * phi64 * phi64
    )
    phi_scale = max(
        float(np.sqrt(volume * np.sum(phi_scale_field * phi_scale_field))),
        1.0e-300,
    )
    chi_scale = max(
        float(np.sqrt(volume * np.sum(chi_scale_field * chi_scale_field))),
        1.0e-300,
    )
    phi_residual = float(np.sqrt(volume * np.sum(phi_equation * phi_equation))) / phi_scale
    chi_residual = float(np.sqrt(volume * np.sum(chi_equation * chi_equation))) / chi_scale
    return max(phi_residual, chi_residual), phi_residual, chi_residual


def cartesian_noether_charge(
    psi_real: np.ndarray,
    psi_real_prev: np.ndarray,
    psi_imag: np.ndarray,
    psi_imag_prev: np.ndarray,
    *,
    dt: float,
    dx: float,
) -> float:
    """Return the exactly conserved leapfrog U(1) bilinear charge."""
    pr = np.asarray(psi_real, dtype=np.float64)
    pr_prev = np.asarray(psi_real_prev, dtype=np.float64)
    pi = np.asarray(psi_imag, dtype=np.float64)
    pi_prev = np.asarray(psi_imag_prev, dtype=np.float64)
    if not (pr.shape == pr_prev.shape == pi.shape == pi_prev.shape):
        raise ValueError("all complex phase-space arrays must match")
    return float(np.sum(pr_prev * pi - pi_prev * pr)) * dx**3 / dt


def cartesian_noether_hamiltonian(
    psi_real: np.ndarray,
    psi_real_prev: np.ndarray,
    psi_imag: np.ndarray,
    psi_imag_prev: np.ndarray,
    chi: np.ndarray,
    chi_prev: np.ndarray,
    *,
    dt: float,
    dx: float,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
) -> dict[str, float]:
    """Evaluate the continuous bare-action Hamiltonian on leapfrog layers."""
    pr = np.asarray(psi_real, dtype=np.float64)
    pr_prev = np.asarray(psi_real_prev, dtype=np.float64)
    pi = np.asarray(psi_imag, dtype=np.float64)
    pi_prev = np.asarray(psi_imag_prev, dtype=np.float64)
    chi64 = np.asarray(chi, dtype=np.float64)
    chi_prev64 = np.asarray(chi_prev, dtype=np.float64)
    if not (
        pr.shape == pr_prev.shape == pi.shape == pi_prev.shape == chi64.shape == chi_prev64.shape
    ):
        raise ValueError("all phase-space arrays must match")
    if pr.ndim != 3:
        raise ValueError("Hamiltonian requires 3D fields")

    volume = dx**3
    inv_dx2 = 1.0 / (dx * dx)
    b_value = chi0 / kappa
    dpr = (pr - pr_prev) / dt
    dpi = (pi - pi_prev) / dt
    dchi = (chi64 - chi_prev64) / dt
    temporal = 0.5 * volume * float(np.sum(dpr * dpr + dpi * dpi))
    matter_gradient = (
        -0.5 * volume * inv_dx2 * float(np.sum(pr * laplacian_19pt(pr) + pi * laplacian_19pt(pi)))
    )
    matter_mass = 0.5 * volume * float(np.sum(chi64 * chi64 * (pr * pr + pi * pi)))
    chi_temporal = 0.5 * b_value * volume * float(np.sum(dchi * dchi))
    chi_gradient = -0.5 * b_value * volume * inv_dx2 * float(np.sum(chi64 * laplacian_19pt(chi64)))
    chi_potential = b_value * lambda_h * volume * float(np.sum((chi64 * chi64 - chi0 * chi0) ** 2))
    total = temporal + matter_gradient + matter_mass + chi_temporal + chi_gradient + chi_potential
    return {
        "total": total,
        "matter_temporal": temporal,
        "matter_gradient": matter_gradient,
        "matter_mass": matter_mass,
        "chi_temporal": chi_temporal,
        "chi_gradient": chi_gradient,
        "chi_potential": chi_potential,
    }


def cartesian_localization_metrics(
    psi_real: np.ndarray,
    psi_imag: np.ndarray,
    *,
    dx: float,
    core_radius: float | None = None,
    reference_density: np.ndarray | None = None,
    reference_center: tuple[float, float, float] | None = None,
) -> dict[str, float | list[float]]:
    """Measure center, radius, anisotropy, core fraction, and aligned profile."""
    pr = np.asarray(psi_real)
    pi = np.asarray(psi_imag)
    if pr.shape != pi.shape or pr.ndim != 3 or len(set(pr.shape)) != 1:
        raise ValueError("psi arrays must be matching cubic 3D arrays")
    rho = pr.astype(np.float64) ** 2 + pi.astype(np.float64) ** 2
    norm = float(np.sum(rho))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("density norm must be finite and positive")
    axis = _cartesian_axis(pr.shape[0], dx)
    wx = np.sum(rho, axis=(1, 2))
    wy = np.sum(rho, axis=(0, 2))
    wz = np.sum(rho, axis=(0, 1))
    center = np.array(
        [
            float(np.dot(axis, wx) / norm),
            float(np.dot(axis, wy) / norm),
            float(np.dot(axis, wz) / norm),
        ]
    )
    offsets = [axis - center[index] for index in range(3)]
    diagonal = np.array(
        [
            float(np.dot(offsets[0] ** 2, wx) / norm),
            float(np.dot(offsets[1] ** 2, wy) / norm),
            float(np.dot(offsets[2] ** 2, wz) / norm),
        ]
    )
    rho_xy = np.sum(rho, axis=2)
    rho_xz = np.sum(rho, axis=1)
    rho_yz = np.sum(rho, axis=0)
    xy = float(np.sum(rho_xy * offsets[0][:, None] * offsets[1][None, :]) / norm)
    xz = float(np.sum(rho_xz * offsets[0][:, None] * offsets[2][None, :]) / norm)
    yz = float(np.sum(rho_yz * offsets[1][:, None] * offsets[2][None, :]) / norm)
    covariance = np.array(
        [
            [diagonal[0], xy, xz],
            [xy, diagonal[1], yz],
            [xz, yz, diagonal[2]],
        ]
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    min_eigenvalue = max(float(eigenvalues[0]), 1.0e-300)
    anisotropy = float(eigenvalues[-1]) / min_eigenvalue
    rms_radius = float(np.sqrt(np.trace(covariance)))

    result: dict[str, float | list[float]] = {
        "density_norm": norm * dx**3,
        "peak_density": float(np.max(rho)),
        "center": center.tolist(),
        "rms_radius": rms_radius,
        "rms_radius_cells": rms_radius / dx,
        "second_moment_eigenvalues": eigenvalues.tolist(),
        "second_moment_anisotropy": anisotropy,
    }
    if core_radius is not None:
        distance_sq = (
            offsets[0][:, None, None] ** 2
            + offsets[1][None, :, None] ** 2
            + offsets[2][None, None, :] ** 2
        )
        result["moving_core_fraction"] = float(
            np.sum(rho[distance_sq <= core_radius * core_radius]) / norm
        )
    if reference_density is not None:
        reference = np.asarray(reference_density, dtype=np.float64)
        if reference.shape != rho.shape:
            raise ValueError("reference_density must match psi arrays")
        if reference_center is None:
            raise ValueError("reference_center is required with reference_density")
        shifts = tuple(
            int(round((reference_center[index] - center[index]) / dx)) for index in range(3)
        )
        aligned = np.roll(rho, shift=shifts, axis=(0, 1, 2))
        reference_norm = max(float(np.sum(reference)), 1.0e-300)
        result["recentered_density_l1"] = float(
            np.sum(np.abs(aligned - reference)) / reference_norm
        )
    return result
