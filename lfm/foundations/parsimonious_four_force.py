"""Experiment-only parsimonious four-force LFM action.

The action keeps the R6 local U(1), SU(2), and SU(3) connection machinery,
disables the independent SO(4) frame carrier, and uses the flat-octic chi
potential for the scalar gravity channel. Dormant frame arrays remain in the
shared experimental state container only for implementation compatibility.
They carry no action, momentum, source, or evolution in this action family.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from lfm.foundations.r3_link_frame_live import R3LiveParameters
from lfm.foundations.r6_unified_live import (
    R6Parameters,
    R6R4Parameters,
    R6Rates,
    R6State,
    step_r6,
)
from lfm.foundations.r6_unified_live import (
    kinetic_energy as r6_kinetic_energy,
)
from lfm.foundations.r6_unified_live import (
    potential_energy_and_rates as r6_potential_energy_and_rates,
)
from lfm.foundations.r6_unified_live import (
    total_hamiltonian as r6_total_hamiltonian,
)
from lfm.foundations.r6_unified_live import (
    vacuum_state as r6_vacuum_state,
)

P4F_ACTION_ID = "LFM-P4F-FLAT-CHI-LOCAL-GAUGE-EXPERIMENT-v1"
P4F_REGISTER_ID = (
    "P4F=(Psi3,Pi3,chi,pchi,U1,E1,SU2L,E2,H,pH,SU3,E3)"
)


def _p4f_r3_parameters() -> R3LiveParameters:
    return R3LiveParameters(
        stencil="19",
        frame_enabled=False,
        chi_potential="flat_octic",
    )


@dataclass(frozen=True)
class P4FParameters:
    """Frozen parameters for the parsimonious action candidate."""

    r6: R6Parameters = field(
        default_factory=lambda: R6Parameters(
            r4=R6R4Parameters(r3=_p4f_r3_parameters())
        )
    )

    def __post_init__(self) -> None:
        r3 = self.r6.r4.r3
        if r3.frame_enabled:
            raise ValueError("P4F prohibits the independent SO(4) frame")
        if r3.chi_potential != "flat_octic":
            raise ValueError("P4F requires the flat-octic chi potential")
        if r3.stencil != "19":
            raise ValueError("P4F v1 freezes the canonical 19-point graph")


def vacuum_state(
    size: int,
    parameters: P4FParameters = P4FParameters(),
) -> R6State:
    return r6_vacuum_state(size, parameters.r6)


def potential_energy_and_rates(
    state: R6State,
    parameters: P4FParameters = P4FParameters(),
) -> tuple[float, R6Rates, dict[str, float]]:
    return r6_potential_energy_and_rates(state, parameters.r6)


def kinetic_energy(
    state: R6State,
    parameters: P4FParameters = P4FParameters(),
) -> tuple[float, dict[str, float]]:
    return r6_kinetic_energy(state, parameters.r6)


def total_hamiltonian(
    state: R6State,
    parameters: P4FParameters = P4FParameters(),
) -> tuple[float, dict[str, float]]:
    return r6_total_hamiltonian(state, parameters.r6)


def inactive_frame_error(state: R6State) -> float:
    """Return the largest departure of the compatibility frame from vacuum."""

    base = state.r3
    identity = np.eye(4)
    return max(
        float(np.max(np.abs(base.shape))),
        float(np.max(np.abs(base.shape_momentum))),
        float(np.max(np.abs(base.frame_electric))),
        float(np.max(np.abs(base.frame_links - identity))),
    )


def step_p4f(
    state: R6State,
    dt: float,
    parameters: P4FParameters = P4FParameters(),
) -> None:
    if inactive_frame_error(state) != 0.0:
        raise ValueError("P4F compatibility frame must remain exact vacuum")
    step_r6(state, dt, parameters.r6)
    if inactive_frame_error(state) != 0.0:
        raise RuntimeError("inactive frame changed under P4F evolution")


def p4f_action_declaration(
    parameters: P4FParameters = P4FParameters(),
) -> dict[str, object]:
    r3 = parameters.r6.r4.r3
    return {
        "action_id": P4F_ACTION_ID,
        "register_id": P4F_REGISTER_ID,
        "canonical_status": "EXPERIMENT_ONLY_UNPROMOTED",
        "parameters": asdict(parameters),
        "site_terms": [
            "covariant_GOV01_matter",
            "flat_octic_GOV02_chi",
            "chi_squared_universal_matter_coupling",
            "chi_weighted_SU2_orientation_alignment",
        ],
        "link_terms": [
            "compact_U1_electric_and_loop_energy",
            "compact_SU2L_electric_and_loop_energy",
            "compact_SU3_electric_and_loop_energy",
            "local_positive_chi_color_dielectric",
        ],
        "reductions": {
            "gravity_only": (
                "identity gauge links, zero link electric fields, zero weak "
                "matter, fixed weak orientation"
            ),
            "bare_flat_octic": (
                "all connection sectors at exact identity vacuum"
            ),
        },
        "inactive_compatibility_registers": [
            "SO4 frame shape",
            "SO4 frame links",
            "SO4 frame electric momenta",
        ],
        "derivation_status": {
            "flat_octic_range": "numerically_supported_not_unique",
            "link_transformation_laws": "derived_from_local_covariance",
            "link_dynamics": "leading_local_positive_Hamiltonian",
            "group_selection": "motivated_not_unique",
            "all_force_closure": "pending_strict_live_gate",
        },
        "constants": {
            "chi0": r3.chi0,
            "kappa": r3.kappa,
            "lambda_h": r3.lambda_h,
            "epsilon_w": r3.epsilon_w,
        },
        "forbidden_mechanisms_used": [],
        "paper_45_update_authorized": False,
    }


def p4f_action_fingerprint(
    parameters: P4FParameters = P4FParameters(),
) -> str:
    encoded = json.dumps(
        p4f_action_declaration(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def flat_octic_minimality_ledger() -> dict[str, object]:
    """Audit the minimal gapless bounded monomial well in ``chi**2``.

    The audit is deliberately limited to analytic one-monomial potentials
    whose leading departure from either vacuum is a power of
    ``z = chi**2 - chi0**2``. It does not assert uniqueness among all smooth
    local potentials.
    """

    rows = []
    for z_power in range(2, 7):
        nonnegative = (z_power % 2) == 0
        gapless = z_power > 2
        rows.append(
            {
                "z_power": z_power,
                "field_degree": 2 * z_power,
                "nonnegative_for_both_signs_of_z": nonnegative,
                "vacuum_hessian_vanishes": gapless,
                "admissible": nonnegative and gapless,
            }
        )
    admissible = [row for row in rows if row["admissible"]]
    minimum = min(int(row["z_power"]) for row in admissible)
    return {
        "assumptions": [
            "local analytic potential",
            "Z2 symmetry through z=chi**2-chi0**2",
            "vacua at plus_or_minus_chi0",
            "bounded below on both sides of the vacuum",
            "vanishing vacuum Hessian for an unscreened linear response",
            "one leading monomial in z",
        ],
        "rows": rows,
        "minimal_admissible_z_power": minimum,
        "minimal_admissible_field_degree": 2 * minimum,
        "normalization_identity": (
            "lambda_h*chi0**4*(z/chi0**2)**4"
            "=lambda_h*z**4/chi0**4"
        ),
        "uniqueness_boundary": (
            "minimal only within the declared analytic monomial class"
        ),
    }
