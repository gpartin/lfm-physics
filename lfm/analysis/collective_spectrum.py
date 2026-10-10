"""Collective-mode diagnostics for the bare full-R2 LFM equations.

This module does not add a metric, gauge field, constraint, or target force
law.  It linearizes the unchanged three-complex-component GOV-01 field and
the real GOV-02 field around homogeneous rotating backgrounds.  The spatial
operator is the exact canonical 19-point Fourier symbol.

The resulting quadratic eigenvalue problem is useful as a fail-closed screen:
Einstein or Maxwell closure needs the required propagating polarizations to
exist before a nonlinear live experiment can sensibly test their interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from lfm.constants import CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import eigenvalue_19pt

if TYPE_CHECKING:
    from numpy.typing import NDArray


RestoringModel = Literal["canonical_quartic", "flat_octic"]


@dataclass(frozen=True)
class RotatingBackground:
    """Equal-density full-R2 rotating background.

    ``q_vectors[a]`` is the carrier wave vector of component ``a``.  A zero
    matrix is the homogeneous condensate.  Equal-magnitude Cartesian rows are
    the preregistered isotropic Fourier triad.
    """

    model: RestoringModel
    total_density: float
    component_amplitude: float
    chi: float
    chi_mass_sq: float
    q_vectors: NDArray[np.float64]
    carrier_frequencies: NDArray[np.float64]
    spacing: float
    wave_speed: float


def _symbol_19(vector: NDArray[np.float64], spacing: float) -> float:
    argument = np.asarray(vector, dtype=float) * spacing
    value = eigenvalue_19pt(argument[0], argument[1], argument[2])
    return float(value) / spacing**2


def discrete_stiffness_19(vector: NDArray[np.float64], spacing: float = 1.0) -> float:
    """Return the nonnegative ``-L19`` eigenvalue at a physical wave vector."""

    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return -_symbol_19(np.asarray(vector, dtype=float), spacing)


def rotating_background(
    total_density: float,
    q_vectors: NDArray[np.float64] | None = None,
    *,
    model: RestoringModel = "canonical_quartic",
    spacing: float = 1.0,
    wave_speed: float = 1.0,
) -> RotatingBackground:
    """Construct an exact uniform-density background of bare GOV-01/GOV-02.

    The density is shared equally by the three complex GOV-01 components.
    The positive GOV-02 branch is used.  ``E0`` is zero, matching the current
    canonical vacuum experiments.
    """

    density = float(total_density)
    if density < 0.0:
        raise ValueError("total_density must be nonnegative")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    if wave_speed <= 0.0:
        raise ValueError("wave_speed must be positive")

    q = np.zeros((3, 3), dtype=float) if q_vectors is None else np.asarray(q_vectors, dtype=float)
    if q.shape != (3, 3):
        raise ValueError("q_vectors must have shape (3, 3)")

    source_coupling = KAPPA / CHI0
    if model == "canonical_quartic":
        chi_sq = CHI0**2 - source_coupling * density / (4.0 * LAMBDA_H)
        if chi_sq <= 0.0:
            raise ValueError("density lies beyond the positive quartic background branch")
        chi_mass_sq = 8.0 * LAMBDA_H * chi_sq
    elif model == "flat_octic":
        delta = -np.cbrt(source_coupling * density * CHI0**4 / (8.0 * LAMBDA_H))
        chi_sq = CHI0**2 + float(delta)
        if chi_sq <= 0.0:
            raise ValueError("density lies beyond the positive flat-octic background branch")
        chi_mass_sq = 48.0 * LAMBDA_H * chi_sq * float(delta) ** 2 / CHI0**4
    else:
        raise ValueError(f"unknown restoring model: {model}")

    chi = float(np.sqrt(chi_sq))
    frequencies = np.array(
        [np.sqrt(chi_sq + wave_speed**2 * discrete_stiffness_19(row, spacing)) for row in q],
        dtype=float,
    )
    return RotatingBackground(
        model=model,
        total_density=density,
        component_amplitude=float(np.sqrt(density / 3.0)),
        chi=chi,
        chi_mass_sq=float(chi_mass_sq),
        q_vectors=q.copy(),
        carrier_frequencies=frequencies,
        spacing=float(spacing),
        wave_speed=float(wave_speed),
    )


def gov02_background_residual(background: RotatingBackground) -> float:
    """Evaluate the algebraic GOV-02 residual of a rotating background."""

    chi = background.chi
    density = background.total_density
    source_coupling = KAPPA / CHI0
    delta = chi**2 - CHI0**2
    if background.model == "canonical_quartic":
        restoring = 4.0 * LAMBDA_H * chi * delta
    else:
        restoring = 8.0 * LAMBDA_H * chi * delta**3 / CHI0**4
    return float(source_coupling * chi * density + restoring)


def principal_symbol_audit() -> dict[str, object]:
    """Return the exact highest-derivative structure of the bare equations."""

    return {
        "real_field_count": 7,
        "principal_operator": "(partial_t^2 - c^2 L19) times identity_7",
        "background_dependent": False,
        "component_spin_under_spatial_rotations": "seven scalar amplitudes",
        "trivial_vacuum_gapless_spin_1_count": 0,
        "trivial_vacuum_gapless_spin_2_count": 0,
        "reason": (
            "GOV-01/GOV-02 couplings are algebraic. They change the mass matrix "
            "but not the shared 19-point principal symbol."
        ),
    }


def vacuum_spectrum_audit(model: RestoringModel) -> dict[str, object]:
    """Return the exact small-perturbation spectrum at the positive vacuum."""

    if model == "canonical_quartic":
        chi_mass_sq = 8.0 * LAMBDA_H * CHI0**2
    elif model == "flat_octic":
        chi_mass_sq = 0.0
    else:
        raise ValueError(f"unknown restoring model: {model}")
    return {
        "model": model,
        "vacuum": "Psi_a=0, chi=chi0",
        "gov01_real_mode_count": 6,
        "gov01_mass_sq": CHI0**2,
        "gov01_spatial_spin": 0,
        "chi_real_mode_count": 1,
        "chi_mass_sq": chi_mass_sq,
        "chi_spatial_spin": 0,
        "gapless_spin_0_count": 1 if chi_mass_sq == 0.0 else 0,
        "gapless_spin_1_count": 0,
        "gapless_spin_2_count": 0,
        "localized_background_implication": (
            "Any finite-energy localized state returns to this spectrum at spatial infinity."
        ),
    }


def zero_chi_branch_audit(
    total_density: float,
    model: RestoringModel,
) -> dict[str, object]:
    """Audit the exact ``chi=0`` constant full-R2 background.

    The canonical parameter remains ``CHI0=19``.  Because GOV-01 contains
    ``chi**2 Psi``, every component is massless on this field-value branch.
    The same square makes the linear matter/chi coupling vanish there.
    """

    density = float(total_density)
    if density < 0.0:
        raise ValueError("total_density must be nonnegative")
    source_coupling = KAPPA / CHI0
    if model == "canonical_quartic":
        threshold = 4.0 * LAMBDA_H * CHI0**2 / source_coupling
        chi_mass_sq = source_coupling * density - 4.0 * LAMBDA_H * CHI0**2
        threshold_leading_force = "-4 lambda_h chi^3"
    elif model == "flat_octic":
        threshold = 8.0 * LAMBDA_H * CHI0**2 / source_coupling
        chi_mass_sq = source_coupling * density - 8.0 * LAMBDA_H * CHI0**2
        threshold_leading_force = "-24 lambda_h chi^3"
    else:
        raise ValueError(f"unknown restoring model: {model}")

    tolerance = 1.0e-12 * max(1.0, threshold)
    if chi_mass_sq > tolerance:
        stability = "LINEARLY_STABLE_CHI_GAPPED"
    elif chi_mass_sq < -tolerance:
        stability = "TACHYONIC"
    else:
        stability = "CRITICAL_CHI_GAPLESS_NONLINEARLY_RESTORED"
    return {
        "model": model,
        "chi0_parameter": CHI0,
        "chi_field_value": 0.0,
        "total_density": density,
        "stability_threshold_density": threshold,
        "chi_linear_mass_sq": chi_mass_sq,
        "stability": stability,
        "threshold_leading_force": threshold_leading_force,
        "gov01_massless_real_scalar_count": 6,
        "gov01_spatial_spin": 0,
        "bare_local_gauge_redundancy": False,
        "bare_gauss_constraint": False,
        "longitudinal_modes_removed": 0,
        "linear_matter_to_chi_source_coefficient": 0.0,
        "linear_chi_to_matter_response_coefficient": 0.0,
        "coupling_reason": ("Both coefficients are proportional to d(chi^2)/dchi = 2 chi."),
        "simultaneous_newtonian_and_maxwell_carrier": False,
    }


def collective_qep_matrices(
    background: RotatingBackground,
    k_vector: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Build ``C, K`` for ``(-Omega^2 I + Omega C + K)x = 0``.

    The perturbation order is ``(u0,u1,u2,v0,v1,v2,xi)``.  ``u`` and ``v``
    are the carrier-frame amplitude and phase quadratures, and ``xi`` is the
    GOV-02 perturbation.  Even and odd carrier sidebands use the exact L19
    symbol, so the result includes lattice dispersion without a continuum
    replacement.
    """

    k = np.asarray(k_vector, dtype=float)
    if k.shape != (3,):
        raise ValueError("k_vector must have shape (3,)")

    c = background.wave_speed
    spacing = background.spacing
    amplitude = background.component_amplitude
    chi = background.chi
    source_coupling = KAPPA / CHI0
    c_matrix = np.zeros((7, 7), dtype=np.complex128)
    k_matrix = np.zeros((7, 7), dtype=np.complex128)

    for component in range(3):
        q = background.q_vectors[component]
        symbol_q = _symbol_19(q, spacing)
        symbol_plus = _symbol_19(q + k, spacing)
        symbol_minus = _symbol_19(q - k, spacing)
        even_stiffness = -(c**2) * (0.5 * (symbol_plus + symbol_minus) - symbol_q)
        odd_stiffness = -(c**2) * 0.5 * (symbol_plus - symbol_minus)
        u_index = component
        v_index = 3 + component
        mu = background.carrier_frequencies[component]

        c_matrix[u_index, v_index] = -2.0j * mu
        c_matrix[v_index, u_index] = 2.0j * mu
        k_matrix[u_index, u_index] = even_stiffness
        k_matrix[v_index, v_index] = even_stiffness
        k_matrix[u_index, v_index] = 1.0j * odd_stiffness
        k_matrix[v_index, u_index] = -1.0j * odd_stiffness
        k_matrix[u_index, 6] = 2.0 * chi * amplitude
        k_matrix[6, u_index] = 2.0 * source_coupling * chi * amplitude

    k_matrix[6, 6] = c**2 * discrete_stiffness_19(k, spacing) + background.chi_mass_sq
    return c_matrix, k_matrix


def collective_mode_eigenpairs(
    background: RotatingBackground,
    k_vector: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Solve the 14-dimensional companion eigenproblem for ``Omega``."""

    c_matrix, k_matrix = collective_qep_matrices(background, k_vector)
    zero = np.zeros_like(c_matrix)
    identity = np.eye(c_matrix.shape[0], dtype=np.complex128)
    companion = np.block([[zero, identity], [k_matrix, c_matrix]])
    values, vectors = np.linalg.eig(companion)
    order = np.lexsort((values.imag, values.real))
    return values[order], vectors[:, order]


def mode_polarization(
    position_eigenvector: NDArray[np.complex128],
    k_vector: NDArray[np.float64],
) -> dict[str, float]:
    """Measure the phase-vector longitudinal/transverse and TT content.

    For the Fourier triad the three phase quadratures can be identified with
    spatial axes.  The symmetric phase-displacement strain is a useful test of
    the proposed lattice-as-frame reading.  Its transverse-traceless part is
    computed rather than assumed.
    """

    state = np.asarray(position_eigenvector, dtype=np.complex128)
    if state.shape != (7,):
        raise ValueError("position_eigenvector must have shape (7,)")
    k = np.asarray(k_vector, dtype=float)
    norm_k = float(np.linalg.norm(k))
    phase = state[3:6]
    phase_norm_sq = float(np.vdot(phase, phase).real)
    if norm_k == 0.0 or phase_norm_sq == 0.0:
        return {
            "phase_fraction": 0.0,
            "phase_transverse_fraction": 0.0,
            "phase_longitudinal_fraction": 0.0,
            "phase_strain_tt_fraction": 0.0,
        }

    direction = k / norm_k
    longitudinal = complex(np.dot(direction, phase))
    longitudinal_fraction = float(abs(longitudinal) ** 2 / phase_norm_sq)
    transverse_fraction = float(max(0.0, 1.0 - longitudinal_fraction))

    strain = 1.0j * (np.outer(k, phase) + np.outer(phase, k))
    projector = np.eye(3) - np.outer(direction, direction)
    projected = projector @ strain @ projector
    tt = projected - 0.5 * projector * np.trace(projected)
    strain_norm_sq = float(np.vdot(strain, strain).real)
    tt_norm_sq = float(np.vdot(tt, tt).real)
    full_norm_sq = float(np.vdot(state, state).real)
    return {
        "phase_fraction": float(phase_norm_sq / full_norm_sq) if full_norm_sq else 0.0,
        "phase_transverse_fraction": transverse_fraction,
        "phase_longitudinal_fraction": longitudinal_fraction,
        "phase_strain_tt_fraction": tt_norm_sq / strain_norm_sq if strain_norm_sq else 0.0,
    }


def berry_action_derivative_audit() -> dict[str, object]:
    """Audit whether the bare action already contains Maxwell dynamics.

    For ``z = Psi/sqrt(Psi^dagger Psi)``, the Berry connection
    ``A_mu = Im(z^dagger partial_mu z)`` is a composite first-derivative
    readout.  Its curvature is quadratic in first derivatives.  A Maxwell
    ``F_mu_nu F^mu_nu`` term is fourth order in derivatives of ``z`` and is
    not present in the two-derivative bare GOV-01 action.
    """

    return {
        "composite_connection": "A_mu = Im(z_dagger partial_mu z)",
        "composite_curvature": ("F_mu_nu = 2 Im(partial_mu z_dagger partial_nu z)"),
        "bare_action_highest_derivative_order": 2,
        "composite_f_squared_derivative_order": 4,
        "bare_action_contains_independent_f_squared": False,
        "maxwell_requires_controlled_elimination_or_new_effective_term": True,
        "homogeneous_identity_is_kinematic": True,
        "inhomogeneous_maxwell_equation_is_bare_euler_lagrange_equation": False,
    }


__all__ = [
    "RestoringModel",
    "RotatingBackground",
    "berry_action_derivative_audit",
    "collective_mode_eigenpairs",
    "collective_qep_matrices",
    "discrete_stiffness_19",
    "gov02_background_residual",
    "mode_polarization",
    "principal_symbol_audit",
    "rotating_background",
    "vacuum_spectrum_audit",
    "zero_chi_branch_audit",
]
