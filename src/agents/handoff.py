"""Hand-off nodes: clarify (ambiguous / out-of-scope), human_escalation (clinician / coordinator, optional HITL
interrupt), and refusal (input blocked by guardrails). None of these ever auto-handles a clinical decision."""
from __future__ import annotations

from langgraph.types import interrupt

from src.audit.audit import audit
from src.context.isolate import merge_result
from src.state import Citation, CopilotState, NextStepDraft


def _cite(policy_id: str, section: str) -> Citation:
    return Citation(policy_id=policy_id, section=section, chunk_id=f"{policy_id}#{section}")


async def clarify(state: CopilotState, runtime) -> dict:
    intake = state["intake"]
    if intake.intent == "out_of_scope":
        draft = NextStepDraft(
            action="decline", rationale="out of scope per INTAKE-001 §5.1", citations=[_cite("INTAKE-001", "§5")],
            patient_message=("I can help with booking appointments, referrals, coverage questions and records "
                             "requests, but I can't help with that. Is there anything about your care I can help with?"),
            staff_note="Out-of-scope request declined (INTAKE-001 §5.1).")
        audit("clarify", "decline", "out_of_scope", reason="INTAKE-001 §5.1")
    else:
        question = intake.clarifying_question or ("Could you tell me a bit more about what you need - for example "
                                                   "an appointment, a referral, a coverage question or records?")
        draft = NextStepDraft(action="clarify", rationale="ambiguous request per INTAKE-001 §2.2 / §5.2",
                              citations=[_cite("INTAKE-001", "§5")], patient_message=question,
                              staff_note=f"Clarification requested (confidence {intake.confidence:.2f}).")
        audit("clarify", "clarify", "ask_clarifying_question", reason=f"intent={intake.intent}")
    return merge_result("clarify", {"next_step": draft})


def _escalation_reason(state: CopilotState) -> tuple[str, bool, list[Citation], str]:
    intake = state.get("intake")
    sup = state.get("supervisor")
    if intake and (intake.red_flags or intake.urgency == "urgent" or intake.intent == "urgent"):
        crisis = "self-harm" in intake.red_flags
        msg = ("Thank you for telling us. I've alerted the on-duty clinician right away. If you are in danger or "
               "your symptoms are severe or getting worse, call your local emergency number now.")
        if crisis:
            msg += " You can also contact your local crisis line at any time."
        return (f"red flag(s): {', '.join(intake.red_flags) or 'urgent'}", True,
                [_cite("INTAKE-001", "§3"), _cite("ESC-001", "§2")], msg)
    if intake and intake.intent == "clinical_question":
        return ("clinical-judgement request", True, [_cite("INTAKE-001", "§4"), _cite("ESC-001", "§2")],
                "I'm not able to assess symptoms or say what a condition might be. I've passed your question to a "
                "clinician who will review it and contact you. If things get worse, please call your local "
                "emergency number.")
    reason = sup.reason if sup else "unable to complete safely"
    return (reason, False, [_cite("ESC-001", "§2")],
            "I couldn't complete this request automatically, so I've passed it to a care coordinator who will "
            "follow up with you.")


async def human_escalation(state: CopilotState, runtime) -> dict:
    reason, clinician, cites, msg = _escalation_reason(state)
    errors = "; ".join(e.error for e in state.get("errors") or [])
    draft = NextStepDraft(action="escalate", rationale=reason, citations=cites, requires_clinician=clinician,
                          patient_message=msg,
                          staff_note=f"ESCALATION -> {'on-duty clinician' if clinician else 'care coordinator'}: "
                                     f"{reason}" + (f" | errors: {errors}" if errors else ""))
    audit("human_escalation", "escalate", "clinician" if clinician else "care_coordinator", reason=reason)
    if runtime.hitl:
        review = interrupt({"reason": reason, "requires_clinician": clinician, "draft": draft.patient_message})
        note = review.get("note", "") if isinstance(review, dict) else str(review)
        audit("human", "hitl_review", "reviewed", reason=note[:200])
        draft = draft.model_copy(update={"staff_note": draft.staff_note + f" | reviewer: {note}"})
    return merge_result("human_escalation", {"next_step": draft})


async def refusal(state: CopilotState, runtime) -> dict:
    guard = state["guard_input"]
    draft = NextStepDraft(
        action="decline", rationale="; ".join(guard.reasons), citations=[_cite("INTAKE-001", "§6")],
        patient_message=("I can't help with that request. For privacy and safety I can only act on your own record "
                         "in this session, and I can't change how I work. I can help you book an appointment, "
                         "check your coverage, or request a referral."),
        staff_note=f"Input blocked by guardrail: {'; '.join(guard.reasons)}")
    return merge_result("refusal", {"next_step": draft})
