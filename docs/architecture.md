# Architecture

```
            ┌──────────── src/service.py (Copilot) ────────────┐
 request →  │ ingress guard: cross-patient detection + PHI mask │  (raw text stops here)
            │ root span copilot.request  (run_id = trace id)   │
            └──────────────────────┬───────────────────────────┘
                                   ▼
 ┌──────────────────────────── src/graph.py (LangGraph StateGraph[CopilotState]) ─────────────────────────────┐
 │ START → input_guard ──block──▶ refusal ─────────────────────────────────────────────▶ output_guard → END │
 │            │ allow/sanitize                                                               ▲             │
 │            ▼                                                                              │             │
 │      context_prep  (long-term recall + summarization middleware)                        │             │
 │            ▼                                                                              │             │
 │       supervisor ◀──────────────┬────────────────┬──────────────┐                        │             │
 │   (pure routing fn `decide`)    │                │              │                        │             │
 │    ├─▶ intake_classifier ───────┘                │              │                        │             │
 │    ├─▶ coverage_checker ── MCP get_patient_record / check_coverage                        │             │
 │    ├─▶ care_pathway ─────── agentic RAG loop (search_care_policy ×≤3 + grade + rewrite)   │             │
 │    ├─▶ clarify ─────────────────────┐                                                      │             │
 │    ├─▶ human_escalation (HITL) ─────┼──▶ memory_write (LangMem + rules → SqliteStore) ─────┘             │
 │    └─▶ coordinator (MCP list_available_slots) ─┘                                                        │
 │ checkpointer: AsyncSqliteSaver (short-term)   store: AsyncSqliteStore (long-term)                       │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        tools ─▶ src/tools/logging_middleware.py (timeout, retry, span, logs/tool_calls.jsonl, audit)
        MCP   ─▶ src/tools/mcp_client.py ══stdio══ mcp_server/server.py (3 tools + 2 resources)
        spans ─▶ src/observability/tracing.py → Phoenix (OTLP) + traces/otel_spans_live.jsonl
```

## Components

| Concern | Where | Notes |
|---|---|---|
| Typed state + structured outputs | `src/state.py` | `CopilotState` TypedDict; Pydantic models at every node boundary |
| Supervisor + conditional edges | `src/agents/supervisor.py` | `decide()` is pure, so it is unit-tested; safety rules come first; loop guard |
| Workers | `src/agents/intake_classifier.py`, `src/agents/coverage_checker.py`, `src/agents/care_pathway.py` | Each writes only its own state keys (`src/context/isolate.py`) |
| Next step | `src/agents/coordinator.py` | Decision is deterministic; Gemini writes only the wording |
| Hand-offs | `src/agents/handoff.py` | clarify, human_escalation (`interrupt()`), refusal |
| MCP server | `mcp_server/server.py`, `mcp_server/store.py` | Tools: `get_patient_record`, `check_coverage`, `list_available_slots`. Resources: `policy://intake/rules`, `patients://schema`. HMAC session scoping. |
| MCP client | `src/tools/mcp_client.py` | One persistent stdio session through `langchain-mcp-adapters` |
| Agentic RAG | `src/tools/rag_tool.py`, `src/agents/care_pathway.py` | Chroma + MiniLM; retrieve → grade → rewrite (≤3 rounds) + targeted rule retrieval |
| Context engineering | `src/context/` | write, select, compress, isolate, summarization, quarantine |
| Memory | `src/memory/short_term.py`, `src/memory/long_term.py` | Thread checkpoints; per-patient semantic memories (LangMem + rules) |
| Guardrails | `src/guardrails/` | ingress + input guard, output guard + risk tiers, PHI |
| Observability | `src/observability/tracing.py`, `src/observability/spans.py` | OpenInference LangChain instrumentor, masking exporter, span categories |
| Audit | `src/audit/audit.py` | `logs/agent_actions.jsonl` |
| Interfaces | `src/cli.py`, `src/api/app.py` | CLI (run / chat / forget); FastAPI SSE streaming (bonus) |

## Context-engineering map

| Strategy | Implementation |
|---|---|
| **Write** | Workers write Pydantic results to state fields; `src/context/write.py::scratchpad` assembles the coordinator's fact sheet |
| **Select** | `src/context/select.py` builds a per-agent minimal prompt. The classifier gets quarantined text, the summary and recent turns, plus §3 of the MCP resource `policy://intake/rules`. |
| **Compress** | `src/context/compress.py`: history trimming to a token budget and per-chunk truncation |
| **Isolate** | `src/context/isolate.py::merge_result` rejects writes outside an agent's own keys |
| **Summarize** | `src/context/summarization.py`: above `summary_trigger_tokens`, old turns fold into `summary` (`RemoveMessage`) |
| **Quarantine** | `src/context/quarantine.py`: mask → score injection → neutralise tags and role labels → wrap in `<untrusted_patient_input>` inside a HumanMessage |
