"""Stationary positive-chi branches of the bare GOV-01/GOV-02 system."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.optimize import NoConvergence, newton_krylov
from scipy.sparse.linalg import eigsh, spsolve

from lfm.constants import CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import laplacian_19pt


@dataclass
class StationaryBranchPoint:
    """One normalized stationary solution or failed continuation point."""

    phi: np.ndarray
    chi: np.ndarray
    omega: float
    norm_target: float
    converged: bool
    cycles: int
    phi_residual: float
    chi_residual_rms: float
    chi_min: float
    effective_sites: float
    message: str


@dataclass
class SupportRemovalPoint:
    """One point in a fixed-source to self-source stationary homotopy."""

    phi: np.ndarray
    chi: np.ndarray
    source_density: np.ndarray
    omega: float
    norm_target: float
    dynamic_fraction: float
    converged: bool
    cycles: int
    phi_residual: float
    chi_residual_rms: float
    chi_min: float
    effective_sites: float
    message: str


def _interior_vector(field: np.ndarray) -> np.ndarray:
    return np.asarray(field[1:-1, 1:-1, 1:-1], dtype=np.float64).ravel()


def _embed_interior(vector: np.ndarray, grid_size: int, boundary: float) -> np.ndarray:
    field = np.full((grid_size, grid_size, grid_size), boundary, dtype=np.float64)
    field[1:-1, 1:-1, 1:-1] = np.asarray(vector, dtype=np.float64).reshape(
        grid_size - 2, grid_size - 2, grid_size - 2
    )
    return field


def _normalized_gaussian(grid_size: int, norm_target: float, sigma: float) -> np.ndarray:
    x = np.arange(grid_size, dtype=np.float64) - (grid_size - 1.0) / 2.0
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    phi = np.exp(-(xx * xx + yy * yy + zz * zz) / (2.0 * sigma * sigma))
    phi[[0, -1], :, :] = 0.0
    phi[:, [0, -1], :] = 0.0
    phi[:, :, [0, -1]] = 0.0
    phi *= np.sqrt(norm_target / max(float(np.sum(phi * phi)), 1.0e-300))
    return phi


def _lowest_mode(chi: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, float]:
    grid_size = chi.shape[0]
    laplacian, _ = _canonical_interior_laplacian(grid_size)
    operator = -laplacian + sp.diags(
        _interior_vector(chi * chi), format="csr"
    )
    values, vectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=_interior_vector(initial),
        tol=1.0e-9,
        maxiter=3000,
    )
    mode = _embed_interior(vectors[:, 0], grid_size, 0.0)
    if float(np.sum(mode * initial)) < 0.0:
        mode *= -1.0
    return mode, float(values[0])


def _solve_positive_chi_density(
    source_density: np.ndarray,
    initial_chi: np.ndarray,
    *,
    chi0: float,
    kappa: float,
    lambda_h: float,
    tolerance: float,
) -> tuple[np.ndarray, bool, str]:
    grid_size = source_density.shape[0]
    initial_positive = np.clip(initial_chi[1:-1, 1:-1, 1:-1], 1.0e-8, None)
    log_initial = np.log(initial_positive).ravel()

    def residual(log_vector: np.ndarray) -> np.ndarray:
        chi = _embed_interior(np.exp(log_vector), grid_size, chi0)
        value = (
            laplacian_19pt(chi)
            - (kappa / chi0) * chi * source_density
            - 4.0 * lambda_h * chi * (chi * chi - chi0 * chi0)
        )
        return _interior_vector(value)

    try:
        solved = newton_krylov(
            residual,
            log_initial,
            method="lgmres",
            f_tol=tolerance,
            maxiter=80,
            verbose=False,
        )
        return _embed_interior(np.exp(np.asarray(solved)), grid_size, chi0), True, "converged"
    except NoConvergence as error:
        candidate = np.asarray(error.args[0], dtype=np.float64)
        if candidate.shape != log_initial.shape or not np.all(np.isfinite(candidate)):
            return initial_chi.copy(), False, "positive-chi Newton solve did not converge"
        return (
            _embed_interior(np.exp(candidate), grid_size, chi0),
            False,
            "positive-chi Newton solve reached iteration limit",
        )
    except (ValueError, FloatingPointError, OverflowError) as error:
        return initial_chi.copy(), False, f"positive-chi solve failed: {type(error).__name__}"


def _solve_positive_chi(
    phi: np.ndarray,
    initial_chi: np.ndarray,
    *,
    chi0: float,
    kappa: float,
    lambda_h: float,
    tolerance: float,
) -> tuple[np.ndarray, bool, str]:
    return _solve_positive_chi_density(
        phi * phi,
        initial_chi,
        chi0=chi0,
        kappa=kappa,
        lambda_h=lambda_h,
        tolerance=tolerance,
    )


@lru_cache(maxsize=8)
def _canonical_interior_laplacian(
    grid_size: int,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return the canonical 19-point interior matrix and unit boundary sum."""
    points = [
        (x, y, z)
        for x in range(1, grid_size - 1)
        for y in range(1, grid_size - 1)
        for z in range(1, grid_size - 1)
    ]
    index = {point: i for i, point in enumerate(points)}
    face_offsets = (
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    )
    edge_offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if abs(dx) + abs(dy) + abs(dz) == 2
    )
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    boundary_sum = np.zeros(len(index), dtype=np.float64)
    for point, row in index.items():
        rows.append(row)
        cols.append(row)
        data.append(-4.0)
        for offsets, weight in ((face_offsets, 1.0 / 3.0), (edge_offsets, 1.0 / 6.0)):
            for dx, dy, dz in offsets:
                neighbor = (point[0] + dx, point[1] + dy, point[2] + dz)
                if neighbor in index:
                    rows.append(row)
                    cols.append(index[neighbor])
                    data.append(weight)
                else:
                    boundary_sum[row] += weight
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(index), len(index)))
    return matrix, boundary_sum


def _solve_positive_chi_sparse(
    source_density: np.ndarray,
    initial_chi: np.ndarray,
    *,
    chi0: float,
    kappa: float,
    lambda_h: float,
    tolerance: float,
    max_iterations: int = 60,
) -> tuple[np.ndarray, bool, str]:
    """Solve the positive stationary chi equation by sparse damped Newton."""
    grid_size = source_density.shape[0]
    laplacian, unit_boundary = _canonical_interior_laplacian(grid_size)
    boundary = chi0 * unit_boundary
    source_coeff = (kappa / chi0) * _interior_vector(source_density)
    u = _interior_vector(initial_chi).copy()

    def residual(vector: np.ndarray) -> np.ndarray:
        return (
            laplacian @ vector
            + boundary
            - source_coeff * vector
            - 4.0 * lambda_h * vector * (vector * vector - chi0 * chi0)
        )

    for _ in range(max_iterations):
        value = residual(u)
        if float(np.sqrt(np.mean(value * value))) < tolerance:
            return _embed_interior(u, grid_size, chi0), True, "converged"
        diagonal = source_coeff + 4.0 * lambda_h * (3.0 * u * u - chi0 * chi0)
        jacobian = laplacian - sp.diags(diagonal, format="csr")
        try:
            delta = spsolve(jacobian, -value)
        except (RuntimeError, ValueError):
            return initial_chi.copy(), False, "sparse positive-chi solve failed"
        if not np.all(np.isfinite(delta)):
            return initial_chi.copy(), False, "sparse positive-chi step was non-finite"
        old_norm = float(np.linalg.norm(value))
        step = 1.0
        while step > 1.0e-10:
            candidate = u + step * delta
            if float(np.min(candidate)) > 0.0 and float(np.linalg.norm(residual(candidate))) < old_norm:
                u = candidate
                break
            step *= 0.5
        else:
            return _embed_interior(u, grid_size, chi0), False, "positive-chi line search failed"
    return _embed_interior(u, grid_size, chi0), False, "positive-chi Newton solve reached iteration limit"


def solve_stationary_branch_point(
    grid_size: int,
    norm_target: float,
    *,
    previous: StationaryBranchPoint | None = None,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
    sigma: float = 3.5,
    max_cycles: int = 30,
    tolerance: float = 1.0e-7,
    mixing: float = 0.6,
) -> StationaryBranchPoint:
    """Solve the normalized stationary bare-LFM equations by continuation.

    The ansatz is ``Psi_a = u_a phi(x) exp(-i omega t)`` with a constant
    internal unit vector. Bare GOV-02 depends only on ``sum_a |Psi_a|^2``, so
    the scalar envelope ``phi`` contains the complete rank-one branch data.
    Positivity is enforced by solving for ``log(chi)``; no value is clipped
    after the nonlinear solve.
    """
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if norm_target <= 0.0:
        raise ValueError("norm_target must be positive")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must lie in (0, 1]")

    if previous is not None and previous.phi.shape == (grid_size,) * 3:
        phi = previous.phi.copy()
        phi *= np.sqrt(norm_target / max(float(np.sum(phi * phi)), 1.0e-300))
        chi = previous.chi.copy()
    else:
        phi = _normalized_gaussian(grid_size, norm_target, sigma)
        density_scale = phi * phi / max(float(np.max(phi * phi)), 1.0e-300)
        chi = chi0 - 0.05 * density_scale
        chi[[0, -1], :, :] = chi0
        chi[:, [0, -1], :] = chi0
        chi[:, :, [0, -1]] = chi0

    message = "maximum SCF cycles reached"
    omega_sq = chi0 * chi0
    phi_residual = float("inf")
    chi_residual_rms = float("inf")
    converged = False
    cycle = 0
    for cycle in range(1, max_cycles + 1):
        mode, omega_sq = _lowest_mode(chi, phi)
        mode *= np.sqrt(norm_target / max(float(np.sum(mode * mode)), 1.0e-300))
        phi = mixing * mode + (1.0 - mixing) * phi
        phi *= np.sqrt(norm_target / max(float(np.sum(phi * phi)), 1.0e-300))

        solved_chi, chi_ok, chi_message = _solve_positive_chi(
            phi,
            chi,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
            tolerance=tolerance,
        )
        chi = mixing * solved_chi + (1.0 - mixing) * chi
        chi[[0, -1], :, :] = chi0
        chi[:, [0, -1], :] = chi0
        chi[:, :, [0, -1]] = chi0

        h_phi = -laplacian_19pt(phi) + chi * chi * phi
        phi_equation = h_phi - omega_sq * phi
        phi_residual = float(np.linalg.norm(_interior_vector(phi_equation))) / max(
            float(np.linalg.norm(_interior_vector(omega_sq * phi))), 1.0
        )
        chi_equation = (
            laplacian_19pt(chi)
            - (kappa / chi0) * chi * phi * phi
            - 4.0 * lambda_h * chi * (chi * chi - chi0 * chi0)
        )
        chi_residual_rms = float(
            np.sqrt(np.mean(_interior_vector(chi_equation) ** 2))
        )
        if chi_ok and phi_residual < tolerance and chi_residual_rms < tolerance:
            converged = True
            message = "stationary equations converged"
            break
        message = chi_message if not chi_ok else "SCF residual above tolerance"

    density = phi * phi
    probability = density / max(float(np.sum(density)), 1.0e-300)
    effective_sites = 1.0 / max(float(np.sum(probability * probability)), 1.0e-300)
    return StationaryBranchPoint(
        phi=phi,
        chi=chi,
        omega=float(np.sqrt(max(omega_sq, 0.0))),
        norm_target=float(norm_target),
        converged=converged,
        cycles=cycle,
        phi_residual=phi_residual,
        chi_residual_rms=chi_residual_rms,
        chi_min=float(np.min(chi)),
        effective_sites=effective_sites,
        message=message,
    )


def continue_stationary_branch(
    grid_size: int,
    norm_targets: list[float],
    **kwargs,
) -> list[StationaryBranchPoint]:
    """Continue stationary solutions over an ordered norm sequence."""
    points: list[StationaryBranchPoint] = []
    previous: StationaryBranchPoint | None = None
    for target in norm_targets:
        point = solve_stationary_branch_point(
            grid_size,
            target,
            previous=previous,
            **kwargs,
        )
        points.append(point)
        if point.converged:
            previous = point
    return points


def solve_support_removal_point(
    fixed_source: np.ndarray,
    dynamic_fraction: float,
    *,
    previous: SupportRemovalPoint | None = None,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    lambda_h: float = LAMBDA_H,
    max_cycles: int = 60,
    tolerance: float = 1.0e-7,
    mixing: float = 0.5,
) -> SupportRemovalPoint:
    """Remove a prescribed source while following the stationary branch.

    ``dynamic_fraction=0`` uses the supplied source density in GOV-02.
    ``dynamic_fraction=1`` uses only ``phi**2`` and is therefore the bare
    rank-one stationary system. The total source norm is held fixed along the
    path; only its spatial support changes.
    """
    fixed_source = np.asarray(fixed_source, dtype=np.float64)
    if fixed_source.ndim != 3 or len(set(fixed_source.shape)) != 1:
        raise ValueError("fixed_source must be a cubic 3D array")
    if fixed_source.shape[0] < 8:
        raise ValueError("fixed_source grid must be at least 8")
    if not np.all(np.isfinite(fixed_source)) or np.any(fixed_source < 0.0):
        raise ValueError("fixed_source must be finite and nonnegative")
    if not 0.0 <= dynamic_fraction <= 1.0:
        raise ValueError("dynamic_fraction must lie in [0, 1]")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must lie in (0, 1]")

    grid_size = fixed_source.shape[0]
    norm_target = float(np.sum(fixed_source))
    if norm_target <= 0.0:
        raise ValueError("fixed_source must have positive total density")

    if previous is not None and previous.phi.shape == fixed_source.shape:
        phi = previous.phi.copy()
        phi *= np.sqrt(norm_target / max(float(np.sum(phi * phi)), 1.0e-300))
        chi = previous.chi.copy()
    else:
        sigma = max(grid_size / 8.0, 1.0)
        phi = _normalized_gaussian(grid_size, norm_target, sigma)
        source_scale = fixed_source / max(float(np.max(fixed_source)), 1.0e-300)
        chi = chi0 - (chi0 - 1.2) * source_scale
        chi[[0, -1], :, :] = chi0
        chi[:, [0, -1], :] = chi0
        chi[:, :, [0, -1]] = chi0

    message = "maximum SCF cycles reached"
    omega_sq = chi0 * chi0
    phi_residual = float("inf")
    chi_residual_rms = float("inf")
    converged = False
    source_density = fixed_source.copy()
    cycle = 0
    for cycle in range(1, max_cycles + 1):
        mode, omega_sq = _lowest_mode(chi, phi)
        mode *= np.sqrt(norm_target / max(float(np.sum(mode * mode)), 1.0e-300))
        phi = mixing * mode + (1.0 - mixing) * phi
        phi *= np.sqrt(norm_target / max(float(np.sum(phi * phi)), 1.0e-300))

        source_density = (
            (1.0 - dynamic_fraction) * fixed_source
            + dynamic_fraction * phi * phi
        )
        solved_chi, chi_ok, chi_message = _solve_positive_chi_sparse(
            source_density,
            chi,
            chi0=chi0,
            kappa=kappa,
            lambda_h=lambda_h,
            tolerance=tolerance,
        )
        chi = mixing * solved_chi + (1.0 - mixing) * chi
        chi[[0, -1], :, :] = chi0
        chi[:, [0, -1], :] = chi0
        chi[:, :, [0, -1]] = chi0

        h_phi = -laplacian_19pt(phi) + chi * chi * phi
        phi_equation = h_phi - omega_sq * phi
        phi_residual = float(np.linalg.norm(_interior_vector(phi_equation))) / max(
            float(np.linalg.norm(_interior_vector(omega_sq * phi))), 1.0
        )
        chi_equation = (
            laplacian_19pt(chi)
            - (kappa / chi0) * chi * source_density
            - 4.0 * lambda_h * chi * (chi * chi - chi0 * chi0)
        )
        chi_residual_rms = float(
            np.sqrt(np.mean(_interior_vector(chi_equation) ** 2))
        )
        if chi_ok and phi_residual < tolerance and chi_residual_rms < tolerance:
            converged = True
            message = "support-removal stationary equations converged"
            break
        message = chi_message if not chi_ok else "SCF residual above tolerance"

    density = phi * phi
    probability = density / max(float(np.sum(density)), 1.0e-300)
    effective_sites = 1.0 / max(float(np.sum(probability * probability)), 1.0e-300)
    return SupportRemovalPoint(
        phi=phi,
        chi=chi,
        source_density=source_density,
        omega=float(np.sqrt(max(omega_sq, 0.0))),
        norm_target=norm_target,
        dynamic_fraction=float(dynamic_fraction),
        converged=converged,
        cycles=cycle,
        phi_residual=phi_residual,
        chi_residual_rms=chi_residual_rms,
        chi_min=float(np.min(chi)),
        effective_sites=effective_sites,
        message=message,
    )


def continue_support_removal(
    fixed_source: np.ndarray,
    dynamic_fractions: list[float],
    **kwargs,
) -> list[SupportRemovalPoint]:
    """Continue from a prescribed source to the bare self-source endpoint."""
    points: list[SupportRemovalPoint] = []
    previous: SupportRemovalPoint | None = None
    for fraction in dynamic_fractions:
        point = solve_support_removal_point(
            fixed_source,
            fraction,
            previous=previous,
            **kwargs,
        )
        points.append(point)
        if point.converged:
            previous = point
    return points
