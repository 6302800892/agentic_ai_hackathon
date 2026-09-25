"""Copilot service: owns the runtime resources and drives one request through the graph.

Per request:
  ingress guard (cross-patient detection + PHI masking on the raw text)  ->  root span `copilot.request`
  (run_id = its trace id)  ->  graph.ainvoke with thread_id + recursion_limit  ->  FinalResponse.
Any GraphRecursionError / unexpected exception degrades to a safe human escalation (never a crash).
"""
from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from src.audit.audit import audit
from src.config import get_settings
from src.graph import build_graph
from src.guardrails.input_guard import ingress
from src.guardrails.phi import patient_ref, session_token
from src.memory.long_term import MEMORY_DB, LongTermMemory, open_store
from src.memory.short_term import CHECKPOINT_DB, open_checkpointer
from src.observability.run_context import current_trace_id, patient_ref_var, request_id_var, run_id_var
from src.observability.tracing import get_tracer
from src.runtime import Runtime, make_gemini
from src.state import FinalResponse, fresh_turn_state
from src.tools.logging_middleware import ToolExecutor
from src.tools.mcp_client import MCP_TOOL_NAMES, MCPToolProvider, local_resource, local_tools
from src.tools.rag_tool import build_rag_tool, get_index

log = logging.getLogger(__name__)

SAFE_FALLBACK = ("I couldn't complete this request automatically, so I've passed it to a care coordinator who "
                 "will follow up with you.")


class Copilot:
    def __init__(self, transport: str = "mcp", hitl: bool = False, use_llm: bool = True,
                 checkpoint_path: Path | str = CHECKPOINT_DB, memory_path: Path | str = MEMORY_DB,
                 rag_backend: str | None = None, tools_override: dict | None = None, **graph_kw):
        self.transport, self.hitl, self.use_llm = transport, hitl, use_llm
        self.checkpoint_path, self.memory_path = checkpoint_path, memory_path
        self.rag_backend, self.tools_override, self.graph_kw = rag_backend, tools_override, graph_kw
        self._stack = AsyncExitStack()
        self.transport_used = None

    async def __aenter__(self) -> "Copilot":
        s = get_settings()
        self.checkpointer = await self._stack.enter_async_context(open_checkpointer(self.checkpoint_path))
        self.store = await self._stack.enter_async_context(open_store(self.memory_path))

        tools, mcp_names, intake_rules = {}, set(), ""
        if self.tools_override is not None:
            tools, self.transport_used = dict(self.tools_override), "override"
        elif self.transport == "mcp":
            try:
                provider = await self._stack.enter_async_context(MCPToolProvider())
                tools, mcp_names = dict(provider.tools), set(MCP_TOOL_NAMES)
                intake_rules = await provider.read_resource("policy://intake/rules")
                self.transport_used = "mcp"
            except Exception as e:  # graceful degradation: same contracts, in-process
                log.warning("MCP server unavailable (%s); falling back to in-process tools.", e)
                audit("system", "degraded", "mcp_unavailable_local_tools", reason=str(e)[:200])
        if not tools:
            tools, self.transport_used = local_tools(), "local"
        if not intake_rules:
            intake_rules = local_resource("policy://intake/rules")

        rag = build_rag_tool(get_index(self.rag_backend or s.rag_backend))
        tools[rag.name] = rag
        llm = make_gemini() if self.use_llm else None
        llm_light = make_gemini(light=True) if self.use_llm else None
        self.runtime = Runtime(tools=ToolExecutor(tools, mcp_tool_names=mcp_names),
                               memory=LongTermMemory(self.store, llm_light), llm=llm, llm_light=llm_light,
                               intake_rules=intake_rules, hitl=self.hitl, rag_tool_name=rag.name)
        self.graph = build_graph(self.runtime, checkpointer=self.checkpointer, store=self.store, **self.graph_kw)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    @property
    def mode(self) -> str:
        return "gemini" if self.runtime.llm is not None else "rules-only (no GOOGLE_API_KEY)"

    async def handle(self, request_id: str, session_id: str, patient_id: str, text: str,
                     thread_id: str | None = None, recursion_limit: int | None = None) -> FinalResponse | dict:
        limits = get_settings().limits
        ref = patient_ref(patient_id)
        ing = ingress(text, patient_id)          # PHI never goes further than this in plaintext
        thread = thread_id or session_id
        config = {"configurable": {"thread_id": thread},
                  "recursion_limit": recursion_limit or limits.get("recursion_limit", 25)}
        inp = fresh_turn_state(request_id=request_id, session_id=session_id, thread_id=thread, patient_ref=ref,
                               session_token=session_token(session_id, ref), input_text=ing.masked_text,
                               ingress_flags=ing.flags)
        return await self._run(inp, config, request_id, ref, ing.masked_text)

    async def resume(self, request_id: str, patient_id: str, thread_id: str, review: dict) -> FinalResponse | dict:
        """Resume a run paused by the HITL interrupt in human_escalation."""
        config = {"configurable": {"thread_id": thread_id}}
        return await self._run(Command(resume=review), config, request_id, patient_ref(patient_id), "[resume]")

    async def _run(self, inp, config: dict, request_id: str, ref: str, masked_text: str):
        tracer = get_tracer()
        with tracer.start_as_current_span("copilot.request", attributes={
                "openinference.span.kind": "CHAIN", "span.category": "acting", "request_id": request_id,
                "session.id": config["configurable"]["thread_id"], "patient_ref": ref,
                "input.value": masked_text}) as span:
            run_id = current_trace_id()
            tokens = [run_id_var.set(run_id), request_id_var.set(request_id), patient_ref_var.set(ref)]
            span.set_attribute("run_id", run_id or "")
            try:
                result = await self.graph.ainvoke(inp, config)
                if result.get("__interrupt__"):
                    span.set_attribute("output.value", "interrupted_for_human_review")
                    return {"interrupt": [i.value for i in result["__interrupt__"]], "run_id": run_id}
                final: FinalResponse = result["final_response"]
            except GraphRecursionError as e:
                audit("system", "degraded", "recursion_limit_escalation", reason=str(e)[:200])
                final = self._safe(request_id, run_id, "recursion_limit")
            except Exception as e:
                log.exception("run failed")
                audit("system", "degraded", "exception_escalation", reason=f"{type(e).__name__}: {str(e)[:200]}")
                final = self._safe(request_id, run_id, f"exception:{type(e).__name__}")
            finally:
                for var, tok in zip((run_id_var, request_id_var, patient_ref_var), tokens):
                    var.reset(tok)
            span.set_attribute("output.value", json.dumps(final.model_dump(), default=str))
            span.set_attribute("copilot.action", final.action)
            span.set_attribute("copilot.risk_tier", final.risk_tier)
            return final

    @staticmethod
    def _safe(request_id: str, run_id: str | None, note: str) -> FinalResponse:
        return FinalResponse(request_id=request_id, action="escalate", risk_tier="high", requires_clinician=False,
                             patient_message=SAFE_FALLBACK, staff_note=f"Automatic handling stopped: {note}",
                             guard_notes=[note], run_id=run_id)

    async def forget(self, patient_id: str) -> int:
        n = await self.runtime.memory.forget(patient_ref(patient_id))
        audit("system", "memory_write", f"forgot_{n}_items", reason="patient erasure request (DPDP)")
        return n
