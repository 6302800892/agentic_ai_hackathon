"""Output guardrail (AC-01, AC-03, AC-06, AC-10) - the last node before END.

  1. Diagnosis-language detector -> rewrite to a clinician-referral message.
  2. Citation check -> schedule/refer drafts must cite a chunk that was actually retrieved; else escalate.
  3. High-risk gating -> urgent / clinical / coverage-denial / uncited outputs force requires_clinician or
     human review, never auto-handled.
  4. PHI scan + masking of every outbound string.
  5. Schema validation (FinalResponse).
"""
from __future__ import annotations

import re
from functools import lru_cache

from src.config import get_settings
from src.guardrails.phi import mask_text
from src.state import (CoverageResult, FinalResponse, IntakeClassification, NextStepDraft, RiskTier)

SAFE_CLINICIAN_MESSAGE = ("I'm not able to assess symptoms or say what a condition might be. I've passed your "
                          "request to a clinician who will review it and contact you. If your symptoms are severe "
                          "or getting worse, please call your local emergency number.")


@lru_cache(maxsize=1)
def _dx_patterns() -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in get_settings().guardrails.get("diagnosis_output_patterns", [])]


def contains_diagnosis(text: str) -> bool:
    return any(p.search(text or "") for p in _dx_patterns())


def classify_output_risk(draft: NextStepDraft, intake: IntakeClassification | None,
                         coverage: CoverageResult | None) -> RiskTier:
    """low: admin/informational or clarify/decline. medium: schedule/refer (staff confirms).
    high: urgent, red flag, clinical question, coverage denial, escalation, low confidence, or uncited."""
    if draft.requires_clinician or draft.action == "escalate":
        return "high"
    if intake and (intake.urgency == "urgent" or intake.red_flags or intake.intent == "clinical_question"):
        return "high"
    if coverage and not coverage.eligible and draft.action in ("schedule", "refer") and \
            any(g.rule_id != "COV-R4" for g in coverage.gaps):
        return "high"
    if draft.action in ("schedule", "refer"):
        if not draft.citations:
            return "high"
        if intake and intake.intent == "admin":
            return "low"
        return "medium"
    return "low"


def apply_output_guard(*, request_id: str, draft: NextStepDraft, intake: IntakeClassification | None,
                       coverage: CoverageResult | None, retrieved_chunk_ids: set[str], pathway_id: str | None,
                       recalled: list[str], run_id: str | None) -> tuple[FinalResponse, list[str]]:
    notes: list[str] = []
    draft = draft.model_copy(deep=True)

    if contains_diagnosis(draft.patient_message) or contains_diagnosis(draft.staff_note):
        notes.append("diagnosis_language_rewritten")
        draft.patient_message = SAFE_CLINICIAN_MESSAGE
        draft.action, draft.requires_clinician = "escalate", True

    if draft.action in ("schedule", "refer"):
        valid = [c for c in draft.citations if c.chunk_id in retrieved_chunk_ids]
        if len(valid) != len(draft.citations):
            notes.append("unresolvable_citation_removed")
        draft.citations = valid
        if not valid:
            notes.append("uncited_action_downgraded_to_escalate")
            draft.action = "escalate"
            draft.staff_note = (draft.staff_note + " | No supporting policy citation; needs human review.").strip(" |")

    if intake and (intake.urgency == "urgent" or intake.red_flags or intake.intent == "clinical_question"):
        if not draft.requires_clinician or draft.action != "escalate":
            notes.append("high_risk_forced_to_clinician")
        draft.action, draft.requires_clinician = "escalate", True

    tier = classify_output_risk(draft, intake, coverage)
    masked_msg, masked_note = mask_text(draft.patient_message, dates=False), mask_text(draft.staff_note, dates=False)
    if masked_msg != draft.patient_message or masked_note != draft.staff_note:
        notes.append("phi_masked_in_output")

    final = FinalResponse(
        request_id=request_id, action=draft.action, risk_tier=tier, requires_clinician=draft.requires_clinician,
        patient_message=masked_msg, staff_note=masked_note, citations=draft.citations,
        coverage_gaps=coverage.gaps if coverage else [], intent=intake.intent if intake else None,
        pathway_id=pathway_id, proposed_slot=draft.proposed_slot, guard_notes=notes,
        recalled_memories=recalled, run_id=run_id)
    return FinalResponse.model_validate(final.model_dump()), notes
