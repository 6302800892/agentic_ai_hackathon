"""LangGraph multi-agent copilot.

START -> input_guard --block--> refusal -----------------------------------------------> output_guard -> END
              | allow/sanitize                                                               ^
              v                                                                              |
         context_prep (recall long-term memory, summarization middleware)                   |
              v                                                                              |
         supervisor <----------------------------------+                                    |
   (conditional edges)                                  |                                    |
     |-> intake_classifier ---------------------------->|                                    |
     |-> coverage_checker  (MCP tools) ---------------->|                                    |
     |-> care_pathway      (agentic RAG) ------------->|                                    |
     |-> clarify ------------------------------------------> memory_write ------------------>|
     |-> human_escalation (clinician / HITL interrupt) -----> memory_write                   |
     '-> coordinator (draft next step, MCP slots) ---------> memory_write                    |

Typed state: src/state.CopilotState. Structured (Pydantic) output at every node boundary.
Checkpointer: AsyncSqliteSaver (short-term). Store: AsyncSqliteStore (long-term). Limits: config/limits.yaml.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agents import care_pathway, coordinator, coverage_checker, handoff, intake_classifier
from src.agents.supervisor import make_supervisor_node, supervisor_edge
from src.audit.audit import audit
from src.context.isolate import merge_result
from src.context.quarantine import quarantine
from src.context.summarization import summarize_if_needed
from src.guardrails.input_guard import evaluate_input
from src.guardrails.output_guard import apply_output_guard
from src.memory.long_term import namespace
from src.observability.run_context import run_id_var
from src.runtime import Runtime
from src.state import CopilotState, NextStepDraft

WORKERS = ["intake_classifier", "coverage_checker", "care_pathway", "coordinator", "clarify", "human_escalation"]
AGENT_NODES = ["input_guard", "refusal", "context_prep", "supervisor", *WORKERS, "memory_write", "output_guard"]


def build_graph(runtime: Runtime, checkpointer=None, store=None,
                router: Callable[[CopilotState], str] | None = None, max_worker_calls: int | None = None):

    # ---------------------------------------------------------------- I/O guard nodes
    async def input_guard(state: CopilotState) -> dict:
        q = quarantine(state["input_text"], already_masked=True)
        guard = evaluate_input(q, state.get("ingress_flags") or [])
        action = {"block": "guardrail_block", "sanitize": "guardrail_sanitize", "allow": "guardrail_allow"}[guard.decision]
        audit("input_guard", action, guard.decision, reason="; ".join(guard.reasons),
              details={"flags": guard.flags, "injection_score": q.injection_score})
        if any(f.startswith("cross_patient") for f in guard.flags):
            audit("input_guard", "access_denied", "refused", reason="request referenced another patient")
        return merge_result("input_guard", {"quarantined_input": q, "guard_input": guard,
                                            "messages": [HumanMessage(q.wrapped())]})

    def route_after_guard(state: CopilotState) -> str:
        return "refusal" if state["guard_input"].decision == "block" else "context_prep"

    async def output_guard(state: CopilotState) -> dict:
        draft: NextStepDraft = state["next_step"]
        pathway = state.get("pathway")
        final, notes = apply_output_guard(
            request_id=state["request_id"], draft=draft, intake=state.get("intake"), coverage=state.get("coverage"),
            retrieved_chunk_ids={c["chunk_id"] for c in state.get("retrieved_chunks") or []},
            pathway_id=pathway.pathway_id if pathway else None,
            recalled=[m.text for m in state.get("recalled_memories") or []], run_id=run_id_var.get())
        if notes:
            audit("output_guard", "output_rewrite", final.action, reason="; ".join(notes))
        return merge_result("output_guard", {"final_response": final,
                                             "messages": [AIMessage(final.patient_message, name="copilot")]})

    # ---------------------------------------------------------------- context + memory nodes
    async def context_prep(state: CopilotState) -> dict:
        update = await summarize_if_needed(state, runtime)
        recalled = await runtime.memory.recall(state["patient_ref"], state["quarantined_input"].masked_text)
        if recalled:
            audit("context_prep", "memory_recall", f"{len(recalled)}_items",
                  details={"kinds": sorted({m.kind for m in recalled})})
        return merge_result("context_prep", {**update, "recalled_memories": recalled})

    async def memory_write(state: CopilotState) -> dict:
        intake, draft, cov, path = state.get("intake"), state.get("next_step"), state.get("coverage"), state.get("pathway")
        facts = await runtime.memory.extract(state["quarantined_input"].masked_text)
        if intake and draft:
            open_items = [f"{g.rule_id} for {cov.service_code}" for g in cov.gaps] if cov and cov.gaps else []
            facts.append({"kind": "visit_outcome", "key": f"visit-{state['request_id']}", "text": (
                f"Previous visit {datetime.now(timezone.utc):%Y-%m-%d}: {intake.intent} request "
                f"({intake.reason_for_visit_category}) -> {draft.action}"
                f"{' via ' + path.pathway_id if path else ''}"
                f"{'; open items: ' + ', '.join(open_items) if open_items else ''}.")})
        keys = await runtime.memory.remember(state["patient_ref"], facts)
        audit("memory_write", "memory_write", f"{len(keys)}_items", details={
            "namespace": "/".join(namespace("<patient_ref>")), "kinds": sorted({f.get("kind", "fact") for f in facts})})
        return {}

    # ---------------------------------------------------------------- worker wrappers
    async def intake_node(state): return await intake_classifier.run(state, runtime)
    async def coverage_node(state): return await coverage_checker.run(state, runtime)
    async def pathway_node(state): return await care_pathway.run(state, runtime)
    async def coordinator_node(state): return await coordinator.run(state, runtime)
    async def clarify_node(state): return await handoff.clarify(state, runtime)
    async def escalation_node(state): return await handoff.human_escalation(state, runtime)
    async def refusal_node(state): return await handoff.refusal(state, runtime)

    g = StateGraph(CopilotState)
    g.add_node("input_guard", input_guard)
    g.add_node("refusal", refusal_node)
    g.add_node("context_prep", context_prep)
    g.add_node("supervisor", make_supervisor_node(router, max_worker_calls))
    g.add_node("intake_classifier", intake_node)
    g.add_node("coverage_checker", coverage_node)
    g.add_node("care_pathway", pathway_node)
    g.add_node("coordinator", coordinator_node)
    g.add_node("clarify", clarify_node)
    g.add_node("human_escalation", escalation_node)
    g.add_node("memory_write", memory_write)
    g.add_node("output_guard", output_guard)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_guard, {"refusal": "refusal", "context_prep": "context_prep"})
    g.add_edge("refusal", "output_guard")
    g.add_edge("context_prep", "supervisor")
    g.add_conditional_edges("supervisor", supervisor_edge, {w: w for w in WORKERS})
    for w in ("intake_classifier", "coverage_checker", "care_pathway"):
        g.add_edge(w, "supervisor")
    for w in ("coordinator", "clarify", "human_escalation"):
        g.add_edge(w, "memory_write")
    g.add_edge("memory_write", "output_guard")
    g.add_edge("output_guard", END)
    return g.compile(checkpointer=checkpointer, store=store)
