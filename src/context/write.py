"""Write: workers persist *structured* results into state fields (not free text into the chat history).

`scratchpad()` is the coordinator's compact, trusted fact sheet assembled from those structured writes.
"""
from __future__ import annotations

from src.state import CopilotState


def scratchpad(state: CopilotState) -> dict:
    intake, cov, path = state.get("intake"), state.get("coverage"), state.get("pathway")
    return {
        "intent": intake.intent if intake else None,
        "category": intake.reason_for_visit_category if intake else None,
        "service_code": intake.service_code if intake else None,
        "urgency": intake.urgency if intake else None,
        "stated_preferences": intake.stated_preferences if intake else [],
        "coverage": {
            "eligible": cov.eligible, "plan_status": cov.plan_status,
            "gaps": [g.model_dump() for g in cov.gaps], "referral_required": cov.referral_required,
        } if cov else None,
        "pathway_id": path.pathway_id if path else None,
        "pathway_recommendation": path.recommended_action if path else None,
        "citations": [f"{c.policy_id} {c.section}" for c in path.citations] if path else [],
        "prior_context": [m.text for m in state.get("recalled_memories", [])],
    }
