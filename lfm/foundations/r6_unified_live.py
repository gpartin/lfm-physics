"""Experimental R6 action with causal weak kinetic normalization.

R6 is the R5 action with no new register and no new potential term. The
weak electric inertia is set equal to the already-derived weak magnetic
stiffness, 1/epsilon_W. This makes the weak characteristic speed the same
unit speed as every other live carrier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from lfm.foundations.r4_unified_live import (
    R4Parameters,
    R4Rates,
    R4State,
    group_constraint_errors,
    reverse_momenta,
    state_distance,
)
from lfm.foundations.r5_unified_live import (
    R5Parameters,
    kinetic_energy as r5_kinetic_energy,
    potential_energy_and_rates as r5_potential_energy_and_rates,
    step_r5,
    total_hamiltonian as r5_total_hamiltonian,
)


R6_ACTION_ID = "LFM-R6-CAUSAL-WEAK-NORMALIZATION-EXPERIMENT-v1"
R6_REGISTER_ID = "R6=R5=R4(no_new_registers)"
R6State = R4State
R6Rates = R4Rates


@dataclass(frozen=True)
class R6R4Parameters(R4Parameters):
    """R4 parameter set with Lorentz-matched weak link inertia."""

    @property
    def weak_inertia(self) -> float:
        return self.weak_stiffness


@dataclass(frozen=True)
class R6Parameters:
    """Complete R6 parameter set."""

    r4: R6R4Parameters = field(default_factory=R6R4Parameters)

    @property
    def r5(self) -> R5Parameters:
        return R5Parameters(r4=self.r4)

    @property
    def stencil(self) -> str:
        return self.r4.stencil

    @property
    def square_coefficient(self) -> float:
        return self.r5.square_coefficient


def vacuum_state(
    size: int,
    parameters: R6Parameters = R6Parameters(),
) -> R6State:
    return R4State.vacuum(size, parameters.r4)


def potential_energy_and_rates(
    state: R6State,
    parameters: R6Parameters = R6Parameters(),
) -> tuple[float, R6Rates, dict[str, float]]:
    return r5_potential_energy_and_rates(state, parameters.r5)


def kinetic_energy(
    state: R6State,
    parameters: R6Parameters = R6Parameters(),
) -> tuple[float, dict[str, float]]:
    return r5_kinetic_energy(state, parameters.r5)


def total_hamiltonian(
    state: R6State,
    parameters: R6Parameters = R6Parameters(),
) -> tuple[float, dict[str, float]]:
    return r5_total_hamiltonian(state, parameters.r5)


def step_r6(
    state: R6State,
    dt: float,
    parameters: R6Parameters = R6Parameters(),
) -> None:
    step_r5(state, dt, parameters.r5)


def r6_action_declaration(
    parameters: R6Parameters = R6Parameters(),
) -> dict[str, object]:
    return {
        "action_id": R6_ACTION_ID,
        "register_id": R6_REGISTER_ID,
        "canonical_status": "EXPERIMENT_ONLY_UNPROMOTED",
        "parameters": asdict(parameters),
        "derived_parameters": {
            "face_square_coefficient": parameters.square_coefficient,
            "weak_stiffness": parameters.r4.weak_stiffness,
            "weak_inertia": parameters.r4.weak_inertia,
            "weak_speed_squared": (
                parameters.r4.weak_stiffness
                / parameters.r4.weak_inertia
            ),
        },
        "retained_terms": ["complete_R5_action"],
        "changed_terms": [
            "weak_electric_inertia_equals_weak_magnetic_stiffness"
        ],
        "new_registers": [],
        "new_potential_terms": [],
        "forbidden_mechanisms_used": [],
        "paper_45_update_authorized": False,
    }


def r6_action_fingerprint(
    parameters: R6Parameters = R6Parameters(),
) -> str:
    encoded = json.dumps(
        r6_action_declaration(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
