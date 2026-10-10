"""Candidate carrier screening tests."""

from __future__ import annotations

from dataclasses import replace

from lfm.analysis.frame_candidates import (
    CandidateVerdict,
    FrameCandidate,
    assess_frame_candidate,
    current_frame_candidate_ledger,
)


def _complete_candidate() -> FrameCandidate:
    return FrameCandidate(
        candidate_id="complete",
        degrees_of_freedom="test",
        source_observable="test",
        gapless=True,
        positive_energy=True,
        source_derived_from_lfm=True,
        attractive_for_positive_energy=True,
        local_action_written=True,
        net_source_compatible=True,
        nonlinear_closure_written=True,
        static_response_power=-2.0,
    )


def test_complete_candidate_survives_algebraic_screen() -> None:
    assessment = assess_frame_candidate(_complete_candidate())
    assert assessment.verdict is CandidateVerdict.SURVIVES


def test_ghost_candidate_is_rejected() -> None:
    candidate = replace(_complete_candidate(), positive_energy=False)
    assessment = assess_frame_candidate(candidate)
    assert assessment.verdict is CandidateVerdict.REJECTED
    assert "positive Hamiltonian" in assessment.reasons


def test_unknown_action_blocks_candidate() -> None:
    candidate = replace(_complete_candidate(), local_action_written=None)
    assessment = assess_frame_candidate(candidate)
    assert assessment.verdict is CandidateVerdict.BLOCKED


def test_current_ledger_has_no_survivor() -> None:
    ledger = current_frame_candidate_ledger()
    verdicts = {row.candidate.candidate_id: row.verdict for row in ledger}
    assert verdicts["canonical-radial-chi"] is CandidateVerdict.REJECTED
    assert verdicts["positive-sync-one-form"] is CandidateVerdict.REJECTED
    assert verdicts["negative-sync-one-form"] is CandidateVerdict.REJECTED
    assert verdicts["ordinary-displacement-strain"] is CandidateVerdict.REJECTED
    assert verdicts["independent-affine-cube-frame"] is CandidateVerdict.BLOCKED
