"""Isolate: each worker gets its own minimal scope and may only write the state keys it owns."""
from __future__ import annotations

from typing import Any

OWNERSHIP: dict[str, set[str]] = {
    "input_guard": {"guard_input", "quarantined_input", "messages"},
    "refusal": {"next_step"},
    "context_prep": {"recalled_memories", "summary", "messages"},
    "supervisor": {"supervisor", "route_history", "step_count"},
    "intake_classifier": {"intake", "errors"},
    "coverage_checker": {"coverage", "errors"},
    "care_pathway": {"pathway", "retrieved_chunks", "errors"},
    "clarify": {"next_step"},
    "human_escalation": {"next_step"},
    "coordinator": {"next_step", "errors"},
    "memory_write": set(),
    "output_guard": {"final_response", "messages"},
}


def merge_result(agent: str, update: dict[str, Any]) -> dict[str, Any]:
    """Return the update only if the agent writes nothing outside its scope."""
    foreign = set(update) - OWNERSHIP[agent]
    if foreign:
        raise PermissionError(f"{agent} attempted to write foreign state keys: {sorted(foreign)}")
    return update
