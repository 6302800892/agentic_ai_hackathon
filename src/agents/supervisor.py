"""Supervisor: routes each request to the right worker.

Routing is a *pure function* of the typed state (`decide`), which makes the conditional edges unit-testable
(tests/test_routing.py). Model judgement enters only through the structured outputs the workers wrote
(intake classification, coverage, pathway). Safety rules take precedence over everything else:
  urgent / red flag / clinical question -> human_escalation (clinician), never auto-handled.
The node also enforces the loop guard: after `max_worker_calls` decisions the run is escalated to a human.
"""
from __future__ import annotations

from typing import Callable

from src.audit.audit import audit
from src.config import get_settings
from src.state import CopilotState, SupervisorDecision

NEEDS_COVERAGE = {"schedule", "referral", "coverage_question"}
NEEDS_PATHWAY = {"schedule", "referral", "coverage_question", "admin"}


def decide(state: CopilotState) -> tuple[str, str]:
    if any(e.fatal for e in state.get("errors") or []):
        return "human_escalation", "fatal worker/tool error - cannot complete safely"
    intake = state.get("intake")
    if intake is None:
        return "intake_classifier", "no intake classification yet"
    if intake.urgency == "urgent" or intake.red_flags or intake.intent == "urgent":
        return "human_escalation", f"red flag / urgent: {', '.join(intake.red_flags) or intake.intent}"
    if intake.intent == "clinical_question":
        return "human_escalation", "clinical-judgement request - clinician decides"
    threshold = get_settings().guardrails.get("confidence_clarify_below", 0.45)
    if intake.intent in ("ambiguous", "out_of_scope") or intake.needs_clarification or intake.confidence < threshold:
        return "clarify", f"intent={intake.intent} confidence={intake.confidence:.2f}"
    if intake.intent in NEEDS_COVERAGE and state.get("coverage") is None:
        return "coverage_checker", f"intent={intake.intent} requires eligibility check"
    if intake.intent in NEEDS_PATHWAY and state.get("pathway") is None:
        return "care_pathway", "care-pathway policy not yet retrieved"
    pathway = state.get("pathway")
    if pathway is not None and pathway.pathway_id == "NONE":
        return "human_escalation", "no applicable care-pathway policy found"
    return "coordinator", "all facts gathered - draft next step"


def route_from_supervisor(state: CopilotState) -> str:
    return decide(state)[0]


def supervisor_edge(state: CopilotState) -> str:
    """Conditional-edge function: reads the decision the supervisor node just wrote."""
    return state["supervisor"].next


def make_supervisor_node(router: Callable[[CopilotState], str] | None = None, max_worker_calls: int | None = None):
    limit = max_worker_calls if max_worker_calls is not None else get_settings().limits.get("max_worker_calls", 8)

    async def supervisor(state: CopilotState) -> dict:
        step = state.get("step_count", 0)
        if router is None:
            nxt, reason = decide(state)
        else:
            nxt, reason = router(state), "custom router"
        if step >= limit:
            nxt, reason = "human_escalation", f"loop_guard: max_worker_calls={limit} reached"
        audit("supervisor", "route", nxt, reason=reason, details={"step": step + 1})
        return {"supervisor": SupervisorDecision(next=nxt, reason=reason),
                "route_history": list(state.get("route_history") or []) + [nxt], "step_count": step + 1}

    return supervisor
