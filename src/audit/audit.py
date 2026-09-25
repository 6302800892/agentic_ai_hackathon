"""Audit middleware (AC-10): one JSONL record per consequential agent action -> logs/agent_actions.jsonl.

Consequential actions: routing decisions, guardrail blocks/sanitisations, tool calls, access denials,
escalations, the drafted next step, and memory writes. Every record is PHI-masked before it is written.
"""
from __future__ import annotations

from typing import Any

from src.config import LOGS
from src.guardrails.phi import mask_obj
from src.observability.run_context import (current_span_id, jsonl_append, now_iso, patient_ref_var,
                                           request_id_var, run_id_var)

AUDIT_LOG = LOGS / "agent_actions.jsonl"

ACTIONS = {"route", "tool_call", "guardrail_block", "guardrail_sanitize", "guardrail_allow", "access_denied",
           "escalate", "clarify", "decline", "draft_next_step", "memory_write", "memory_recall",
           "output_rewrite", "degraded", "hitl_review"}


def audit(actor: str, action: str, decision: str, *, tool: str | None = None, reason: str = "",
          details: dict[str, Any] | None = None) -> dict:
    if action not in ACTIONS:
        raise ValueError(f"unknown audit action {action!r}")
    record = {
        "timestamp": now_iso(),
        "run_id": run_id_var.get(),
        "span_id": current_span_id(),
        "request_id": request_id_var.get(),
        "actor": actor,
        "action": action,
        "tool": tool,
        "decision": decision,
        "reason": reason,
        "patient_ref": patient_ref_var.get(),
        "details": details or {},
    }
    record = mask_obj(record)
    jsonl_append(AUDIT_LOG, record)
    return record
