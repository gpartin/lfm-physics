"""Algebraic screening records for candidate LFM cube-frame carriers.

The screen does not add a degree of freedom to GOV-01/GOV-02. It makes the
requirements for a candidate explicit and rejects familiar false-positive
routes before nonlinear evolution or visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CandidateVerdict(str, Enum):
    """Outcome of the algebraic candidate screen."""

    SURVIVES = "SURVIVES"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class FrameCandidate:
    """Declared properties of one proposed long-range carrier."""

    candidate_id: str
    degrees_of_freedom: str
    source_observable: str
    gapless: bool | None
    positive_energy: bool | None
    source_derived_from_lfm: bool | None
    attractive_for_positive_energy: bool | None
    local_action_written: bool | None
    net_source_compatible: bool | None
    nonlinear_closure_written: bool | None
    static_response_power: float | None
    notes: str = ""


@dataclass(frozen=True)
class CandidateAssessment:
    """Assessment of a frame candidate against frozen prerequisites."""

    candidate: FrameCandidate
    verdict: CandidateVerdict
    reasons: tuple[str, ...]


def assess_frame_candidate(candidate: FrameCandidate) -> CandidateAssessment:
    """Screen a candidate without assuming target gravitational equations.

    ``static_response_power`` is the measured or derived small-k exponent of
    the sourced response. A long-range static Green response in three spatial
    dimensions requires a ``k^-2`` pole, represented here by ``-2``.
    """
    requirements = {
        "gapless carrier": candidate.gapless,
        "positive Hamiltonian": candidate.positive_energy,
        "source derived from LFM": candidate.source_derived_from_lfm,
        "attraction for positive energy": candidate.attractive_for_positive_energy,
        "explicit local action": candidate.local_action_written,
        "net-source/zero-mode consistency": candidate.net_source_compatible,
        "nonlinear closure": candidate.nonlinear_closure_written,
    }
    unknown = tuple(name for name, value in requirements.items() if value is None)
    failed = tuple(name for name, value in requirements.items() if value is False)
    if candidate.static_response_power is None:
        unknown += ("measured static small-k response",)
    elif abs(candidate.static_response_power + 2.0) > 0.15:
        failed += (
            "static response lacks the required small-k k^-2 pole "
            f"(measured power {candidate.static_response_power:.3f})",
        )

    if failed:
        return CandidateAssessment(
            candidate=candidate,
            verdict=CandidateVerdict.REJECTED,
            reasons=failed + tuple(f"unknown: {name}" for name in unknown),
        )
    if unknown:
        return CandidateAssessment(
            candidate=candidate,
            verdict=CandidateVerdict.BLOCKED,
            reasons=tuple(f"unknown: {name}" for name in unknown),
        )
    return CandidateAssessment(
        candidate=candidate,
        verdict=CandidateVerdict.SURVIVES,
        reasons=("all algebraic prerequisites satisfied",),
    )


def current_frame_candidate_ledger() -> tuple[CandidateAssessment, ...]:
    """Return the frozen ledger for routes examined as of 2026-07-24."""
    candidates = (
        FrameCandidate(
            candidate_id="canonical-radial-chi",
            degrees_of_freedom="one real site scalar chi",
            source_observable="bare local wave norm/energy coupling",
            gapless=False,
            positive_energy=True,
            source_derived_from_lfm=True,
            attractive_for_positive_energy=None,
            local_action_written=True,
            net_source_compatible=True,
            nonlinear_closure_written=True,
            static_response_power=0.0,
            notes="Mexican-hat curvature gives the radial chi mode a mass gap.",
        ),
        FrameCandidate(
            candidate_id="positive-sync-one-form",
            degrees_of_freedom="oriented synchronization link",
            source_observable="exact bare LFM energy current",
            gapless=True,
            positive_energy=True,
            source_derived_from_lfm=True,
            attractive_for_positive_energy=False,
            local_action_written=True,
            net_source_compatible=False,
            nonlinear_closure_written=False,
            static_response_power=-2.0,
            notes="Positive normalization gives repulsion for positive sources.",
        ),
        FrameCandidate(
            candidate_id="negative-sync-one-form",
            degrees_of_freedom="oriented synchronization link",
            source_observable="exact bare LFM energy current",
            gapless=True,
            positive_energy=False,
            source_derived_from_lfm=True,
            attractive_for_positive_energy=True,
            local_action_written=True,
            net_source_compatible=False,
            nonlinear_closure_written=False,
            static_response_power=-2.0,
            notes="Attraction requires a negative-energy normalization.",
        ),
        FrameCandidate(
            candidate_id="ordinary-displacement-strain",
            degrees_of_freedom="cube displacement vector and symmetric strain",
            source_observable="derivative strain coupling",
            gapless=True,
            positive_energy=True,
            source_derived_from_lfm=True,
            attractive_for_positive_energy=None,
            local_action_written=True,
            net_source_compatible=True,
            nonlinear_closure_written=False,
            static_response_power=0.0,
            notes="Derivative source/backreaction cancels the elastic 1/k^2 pole.",
        ),
        FrameCandidate(
            candidate_id="independent-affine-cube-frame",
            degrees_of_freedom="undetermined local cube-frame variables",
            source_observable="exact bare LFM energy-current candidate",
            gapless=None,
            positive_energy=None,
            source_derived_from_lfm=True,
            attractive_for_positive_energy=None,
            local_action_written=None,
            net_source_compatible=None,
            nonlinear_closure_written=None,
            static_response_power=None,
            notes="This is the open construction problem, not an implemented mode.",
        ),
    )
    return tuple(assess_frame_candidate(candidate) for candidate in candidates)
