"""Executable R3 link-frame action prototype for LFM.

R3 retains the canonical matter and radial-chi sectors and adds:

- a traceless symmetric four-direction frame-shape register;
- an oriented SO(4) frame-comparison link;
- an oriented U(1) phase link; and
- an oriented SU(3) color-frame link.

The link variables make neighbor comparison local-covariant and give loop
holonomy a positive local energy. The SO(4) curvature admits two chiral
three-component pieces, providing a bounded parity-sensitive location for
the existing epsilon_W parameter.

This module is a foundational candidate. It is not enabled in Simulation and
does not by itself establish live force recovery, confinement, weak chirality,
or a canonical change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from lfm.analysis.frame_completion import (
    FRAME_COMPONENT_COUNT,
    frame_projectors,
    rest_energy_source,
)
from lfm.constants import C_DEFAULT, CHI0, EPSILON_W, KAPPA, LAMBDA_H


R3_REGISTER_ID = "R3=(Psi_a,chi,S_AB,UFrame_ij,U1_ij,U3_ij)"
R3_ACTION_ID = "LFM-R3-LINK-FRAME-CANDIDATE-v1"


def _unitarity_error(matrix: np.ndarray) -> float:
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    return float(np.max(np.abs(matrix @ matrix.conj().T - identity)))


@dataclass(frozen=True)
class R3LinkFrameParameters:
    """Parameters of the single declared R3 candidate action."""

    chi0: float = CHI0
    kappa: float = KAPPA
    lambda_h: float = LAMBDA_H
    wave_speed: float = C_DEFAULT
    epsilon_w: float = EPSILON_W
    phase_link_stiffness: float = 1.0
    color_link_stiffness: float = 1.0
    frame_link_stiffness: float | None = None

    @property
    def frame_inertia(self) -> float:
        """Return the common scale/shape inertia hypothesis B=chi0/kappa."""

        return self.chi0 / self.kappa

    @property
    def radial_mass_sq(self) -> float:
        """Return the retained radial Mexican-hat curvature."""

        return 8.0 * self.lambda_h * self.chi0**2

    @property
    def effective_frame_link_stiffness(self) -> float:
        """Return the declared frame-link stiffness."""

        if self.frame_link_stiffness is None:
            return self.frame_inertia * self.wave_speed**2
        return self.frame_link_stiffness

    def __post_init__(self) -> None:
        positive = (
            self.chi0,
            self.kappa,
            self.lambda_h,
            self.wave_speed,
            self.phase_link_stiffness,
            self.color_link_stiffness,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("R3 positive parameters must be finite")
        if not np.isfinite(self.epsilon_w) or abs(self.epsilon_w) >= 1.0:
            raise ValueError("epsilon_w must satisfy abs(epsilon_w)<1")
        if (
            self.frame_link_stiffness is not None
            and (
                not np.isfinite(self.frame_link_stiffness)
                or self.frame_link_stiffness <= 0.0
            )
        ):
            raise ValueError("frame_link_stiffness must be positive")


@dataclass(frozen=True)
class R3ProductLink:
    """One oriented frame, phase, and color transport link."""

    frame: np.ndarray
    phase: complex
    color: np.ndarray

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame, dtype=np.float64)
        color = np.asarray(self.color, dtype=np.complex128)
        phase = complex(self.phase)
        if frame.shape != (4, 4):
            raise ValueError("frame link must have shape (4,4)")
        if color.shape != (3, 3):
            raise ValueError("color link must have shape (3,3)")
        if _unitarity_error(frame) > 1.0e-10:
            raise ValueError("frame link must be orthogonal")
        if float(np.linalg.det(frame)) <= 0.0:
            raise ValueError("frame link must be orientation preserving")
        if abs(abs(phase) - 1.0) > 1.0e-10:
            raise ValueError("phase link must have unit magnitude")
        if _unitarity_error(color) > 1.0e-10:
            raise ValueError("color link must be unitary")
        if abs(np.linalg.det(color) - 1.0) > 1.0e-10:
            raise ValueError("color link must have determinant one")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "color", color)

    @classmethod
    def identity(cls) -> R3ProductLink:
        """Return the identity transport."""

        return cls(
            frame=np.eye(4),
            phase=1.0 + 0.0j,
            color=np.eye(3, dtype=np.complex128),
        )

    def reverse(self) -> R3ProductLink:
        """Return the exactly constrained reverse-oriented link."""

        return R3ProductLink(
            frame=self.frame.T,
            phase=np.conj(self.phase),
            color=self.color.conj().T,
        )

    def compose(self, other: R3ProductLink) -> R3ProductLink:
        """Return the ordered product of two compatible transports."""

        return R3ProductLink(
            frame=self.frame @ other.frame,
            phase=self.phase * other.phase,
            color=self.color @ other.color,
        )


def transform_internal_matter(
    matter: np.ndarray,
    *,
    phase: complex,
    color: np.ndarray,
) -> np.ndarray:
    """Apply one local U(1) x SU(3) re-basing to a color triplet."""

    vector = np.asarray(matter, dtype=np.complex128)
    matrix = np.asarray(color, dtype=np.complex128)
    if vector.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("matter must be (3,) and color must be (3,3)")
    return complex(phase) * (matrix @ vector)


def transform_product_link(
    link: R3ProductLink,
    *,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    phase_i: complex,
    phase_j: complex,
    color_i: np.ndarray,
    color_j: np.ndarray,
) -> R3ProductLink:
    """Apply independent local endpoint changes of basis."""

    left_frame = np.asarray(frame_i, dtype=np.float64)
    right_frame = np.asarray(frame_j, dtype=np.float64)
    left_color = np.asarray(color_i, dtype=np.complex128)
    right_color = np.asarray(color_j, dtype=np.complex128)
    return R3ProductLink(
        frame=left_frame @ link.frame @ right_frame.T,
        phase=complex(phase_i) * link.phase * np.conj(complex(phase_j)),
        color=left_color @ link.color @ right_color.conj().T,
    )


def internal_covariant_difference(
    matter_i: np.ndarray,
    matter_j: np.ndarray,
    link_ij: R3ProductLink,
) -> np.ndarray:
    """Compare neighboring phase/color triplets in the local i basis."""

    left = np.asarray(matter_i, dtype=np.complex128)
    right = np.asarray(matter_j, dtype=np.complex128)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("matter values must have shape (3,)")
    transported = link_ij.phase * (link_ij.color @ right)
    return transported - left


def chiral_frame_curvature(
    frame_holonomy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split infinitesimal SO(4) curvature into two chiral 3-vectors."""

    matrix = np.asarray(frame_holonomy, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("frame_holonomy must have shape (4,4)")
    omega = 0.5 * (matrix - matrix.T)
    temporal = np.asarray(
        [omega[0, 1], omega[0, 2], omega[0, 3]]
    )
    spatial = np.asarray(
        [omega[2, 3], omega[3, 1], omega[1, 2]]
    )
    plus = (temporal + spatial) / np.sqrt(2.0)
    minus = (temporal - spatial) / np.sqrt(2.0)
    return plus, minus, omega


def product_plaquette_energy(
    holonomy: R3ProductLink,
    parameters: R3LinkFrameParameters = R3LinkFrameParameters(),
) -> dict[str, float]:
    """Return positive local loop energies for the declared R3 action."""

    identity4 = np.eye(4)
    symmetric_mismatch = 0.5 * (
        holonomy.frame + holonomy.frame.T
    ) - identity4
    plus, minus, _ = chiral_frame_curvature(holonomy.frame)
    frame_stiffness = parameters.effective_frame_link_stiffness
    frame_even = 0.5 * frame_stiffness * float(
        np.sum(symmetric_mismatch**2)
    )
    frame_chiral = 0.5 * frame_stiffness * (
        (1.0 + parameters.epsilon_w) * float(plus @ plus)
        + (1.0 - parameters.epsilon_w) * float(minus @ minus)
    )
    phase = parameters.phase_link_stiffness * (
        1.0 - float(np.real(holonomy.phase))
    )
    color = parameters.color_link_stiffness * (
        3.0 - float(np.real(np.trace(holonomy.color)))
    )
    total = frame_even + frame_chiral + phase + color
    return {
        "frame_even": frame_even,
        "frame_chiral": frame_chiral,
        "phase": phase,
        "color": color,
        "total": total,
    }


def frame_shape_acceleration(
    laplacian_shape: np.ndarray,
    bare_energy_density: np.ndarray,
    shape: np.ndarray,
    parameters: R3LinkFrameParameters = R3LinkFrameParameters(),
) -> np.ndarray:
    """Return the candidate sourced R3 frame-shape acceleration."""

    laplacian = np.asarray(laplacian_shape, dtype=np.float64)
    field = np.asarray(shape, dtype=np.float64)
    density = np.asarray(bare_energy_density, dtype=np.float64)
    if laplacian.shape != field.shape or field.shape[-1] != FRAME_COMPONENT_COUNT:
        raise ValueError("shape arrays must match with final dimension 10")
    if density.shape != field.shape[:-1]:
        raise ValueError("bare_energy_density must match spatial shape")
    _, shape_projector = frame_projectors()
    projected_laplacian = np.einsum(
        "ij,...j->...i",
        shape_projector,
        laplacian,
    )
    source = shape_projector @ rest_energy_source()
    clock_factor = np.exp(field[..., 0])
    return (
        parameters.wave_speed**2 * projected_laplacian
        - clock_factor[..., np.newaxis]
        * density[..., np.newaxis]
        * source
        / parameters.frame_inertia
    )


def r3_action_declaration(
    parameters: R3LinkFrameParameters = R3LinkFrameParameters(),
) -> dict[str, object]:
    """Return the machine-readable single-action declaration."""

    payload = asdict(parameters)
    payload["frame_inertia"] = parameters.frame_inertia
    payload["radial_mass_sq"] = parameters.radial_mass_sq
    payload["effective_frame_link_stiffness"] = (
        parameters.effective_frame_link_stiffness
    )
    return {
        "action_id": R3_ACTION_ID,
        "register_id": R3_REGISTER_ID,
        "canonical_status": "UNPROMOTED_FOUNDATIONAL_CANDIDATE",
        "retained_sectors": [
            "bare_GOV01_matter",
            "bare_GOV02_radial_chi",
            "full_mexican_hat",
        ],
        "added_site_registers": ["traceless_frame_shape_S_AB"],
        "added_link_registers": [
            "SO4_frame_transport",
            "U1_phase_transport",
            "SU3_color_transport",
        ],
        "neighbor_term": "norm(U1_ij*U3_ij*Psi_j-Psi_i)^2",
        "loop_terms": [
            "positive_SO4_chiral_frame_mismatch",
            "positive_U1_plaquette_mismatch",
            "positive_SU3_plaquette_mismatch",
        ],
        "weak_location": (
            "bounded parity weighting of the two SO4 chiral curvature pieces"
        ),
        "parameters": payload,
        "known_open_items": [
            "frame normalization derivation",
            "live link Hamilton equations",
            "weak matter representation and mediator mass",
            "strong confinement and running",
            "quantitative long-range force recovery",
            "integrated all-four evolution",
        ],
    }


def r3_action_fingerprint(
    parameters: R3LinkFrameParameters = R3LinkFrameParameters(),
) -> str:
    """Return a stable fingerprint of the declared candidate action."""

    encoded = json.dumps(
        r3_action_declaration(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
