"""Coordinator: drafts the coordinated next step (schedule / refer / escalate).

The *decision* is deterministic from the structured facts (intent, coverage gaps, pathway) so it is auditable
and testable; Gemini only writes the wording (DraftText). Slot choice honours preferences stated in this session
or recalled from long-term memory (AC-05). Clinical decisions are never made here.
"""
from __future__ import annotations

import re

from src.audit.audit import audit
from src.context.isolate import merge_result
from src.context.select import context_for_coordinator
from src.state import CopilotState, DraftText, ErrorRecord, NextStepDraft

AGENT = "coordinator"


def decide_action(state: CopilotState) -> dict:
    intake, cov = state["intake"], state.get("coverage")
    gaps = [g.rule_id for g in cov.gaps] if cov else []
    blocking = [g for g in gaps if g != "COV-R4"]
    if intake.intent == "admin":
        return {"action": "refer", "target": "front-desk administration team", "slot_pathway": None,
                "reason": "administrative request routed per CP-ADMIN-001"}
    if blocking:
        return {"action": "escalate", "target": "coverage desk", "slot_pathway": None,
                "reason": f"coverage gap(s) {', '.join(blocking)} - human coverage review"}
    if "COV-R4" in gaps:
        return {"action": "refer", "target": "primary care (referral needed first)", "slot_pathway": "GENERAL",
                "reason": "COV-R4: referral required, none on file - primary-care visit first"}
    if intake.intent == "referral":
        return {"action": "refer", "target": "specialist referral queue", "slot_pathway": state["pathway"].pathway_id,
                "reason": "referral on file - draft specialist referral request"}
    return {"action": "schedule", "target": "care pathway clinic", "slot_pathway": state["pathway"].pathway_id,
            "reason": f"eligible - schedule per {state['pathway'].pathway_id}"}


def preferred_period(state: CopilotState) -> str | None:
    texts = list(state["intake"].stated_preferences) + [m.text for m in state.get("recalled_memories") or []]
    joined = " ".join(texts).lower()
    for period in ("morning", "afternoon"):
        if re.search(period, joined):
            return period
    return None


def _template(decided: dict, state: CopilotState, slot: dict | None, period: str | None) -> DraftText:
    pathway = state["pathway"].pathway_id
    cov = state.get("coverage")
    when = f"{slot['clinic']} on {slot['start'].replace('T', ' at ')}" if slot else "the next available slot"
    pref = f" (matching your preference for {period}s)" if period and slot and slot.get("period") == period else ""
    a = decided["action"]
    if a == "escalate":
        msg = ("I wasn't able to confirm coverage for this service, so I've passed your request to our coverage "
               "team. They'll contact you about the options - no booking has been made yet.")
    elif decided["target"].startswith("primary care"):
        msg = (f"This service needs a referral from primary care first, so the next step is a primary-care "
               f"appointment: {when}{pref}. The clinician there will decide on the referral.")
    elif decided["target"].startswith("front-desk"):
        msg = ("I've passed your request to the front-desk administration team. They'll process it and reply "
               "using the verified contact details on your record.")
    elif a == "refer":
        msg = f"I've drafted a referral request; the specialist team will review it. Earliest slot: {when}{pref}."
    else:
        msg = (f"I can offer an appointment at {when}{pref}. A member of the care team will confirm the booking "
               f"with you.")
    gaps = "; ".join(f"{g.rule_id}: {g.description}" for g in cov.gaps) if cov and cov.gaps else "none"
    cites = ", ".join(f"{c.policy_id} {c.section}" for c in state["pathway"].citations)
    note = (f"intent={state['intake'].intent}; pathway={pathway}; action={a} -> {decided['target']}; "
            f"coverage gaps: {gaps}; citations: {cites}")
    return DraftText(patient_message=msg, staff_note=note, rationale=decided["reason"])


async def run(state: CopilotState, runtime) -> dict:
    intake = state["intake"]
    decided = decide_action(state)
    errors = list(state.get("errors") or [])
    period, slot = preferred_period(state), None
    if decided["slot_pathway"] and decided["action"] in ("schedule", "refer"):
        urgency = intake.urgency if intake.urgency in ("routine", "soon") else "routine"
        res = await runtime.tools.call("list_available_slots", {"pathway_id": decided["slot_pathway"],
                                                                "urgency": urgency}, agent=AGENT)
        if res.ok and res.data.get("slots"):
            slots = res.data["slots"]
            slot = next((s for s in slots if period and s["period"] == period), slots[0])
        else:
            errors.append(ErrorRecord(node=AGENT, error=f"list_available_slots {res.status}: {res.error}"))

    template = _template(decided, state, slot, period)
    text = await runtime.structured(DraftText, context_for_coordinator(
        state, {"decided_action": decided["action"], "route_to": decided["target"], "proposed_slot": slot,
                "preference_applied": period, "template_patient_message": template.patient_message}),
        agent=AGENT)
    text = text or template

    draft = NextStepDraft(action=decided["action"], rationale=text.rationale, citations=state["pathway"].citations,
                          requires_clinician=False, patient_message=text.patient_message,
                          staff_note=text.staff_note or template.staff_note, proposed_slot=slot)
    audit(AGENT, "draft_next_step", draft.action, reason=decided["reason"],
          details={"pathway": state["pathway"].pathway_id, "slot": slot["slot_id"] if slot else None,
                   "preference_applied": period})
    return merge_result(AGENT, {"next_step": draft, "errors": errors})
