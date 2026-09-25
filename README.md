# Patient Intake & Care-Coordination Copilot (BC-AAIE-HACK-10)

A LangGraph multi-agent copilot for front-desk and care-coordination staff. For each synthetic patient request
it does four things:
1. Classifies the reason for visit.
2. Verifies coverage through a custom MCP server.
3. Retrieves and cites the care-pathway policy (agentic RAG).
4. Drafts a next step: schedule, refer or escalate.

It **never diagnoses**, and urgent or clinical-judgement cases always go to a clinician. The system is
instrumented with Arize Phoenix, cost-governed, guarded (input/output guardrails, audit trail, PHI masking),
documented for governance, and evaluated (DeepEval with a Gemini judge, plus agent tests).

- **Model provider:** Google Gemini only.
- **Data:** synthetic only.
- **Runtime:** pip + Python. No Docker and no external database.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt             # or: pip install -e ".[dev,api]"  (pyproject.toml)
python -m spacy download en_core_web_sm      # Presidio NER (optional; regex masking always applies)
cp .env.example .env                         # then set GOOGLE_API_KEY (Windows: copy .env.example .env)
```

The committed sample inputs are in `data/`. They are regenerable with `python scripts/generate_synthetic_data.py`
and the output is deterministic.

## 1. Run the copilot (single command)

```bash
python -m src.cli run --input data/samples/intake_requests.jsonl
```

- This runs 14 requests (including a two-turn session) through the graph, with the MCP server over stdio and
  Phoenix tracing. The UI runs at http://localhost:6006; add `--keep-ui` to keep it open.
- `data/samples/session_return_visit.jsonl` holds the cross-session return visits.
- Other commands:
  - `python -m src.cli chat --patient SYN-P-00004 --session demo1 [--hitl]` for interactive use. `--hitl` pauses
    escalations for a reviewer note.
  - `python -m src.cli forget --patient SYN-P-00004` erases long-term memory (DPDP erasure).
  - `uvicorn src.api.app:app` is the bonus FastAPI SSE streaming endpoint (`POST /intake`).

## 2. Regenerate traces, evaluation and all evidence (single command)

```bash
python scripts/regenerate_all.py            # add --keep-ui to screenshot Phoenix at the end
```

Everything runs in one process with one tracer, so all artifacts share `run_id`s. The steps:
1. Reset the evidence logs and local state. `traces/failures/` is kept.
2. Build the index.
3. Run the demo.
4. Run the golden-set eval (DeepEval judge = Gemini).
5. Run the red team.
6. Export the Phoenix spans.
7. Compute the golden signals.
8. Build the dashboard data and chart.
9. Run `pytest`.
10. Run the citation, secret and PHI gate, then write the manifest (`reports/manifest.json`).

Individual steps: `scripts/run_demo.py`, `scripts/run_eval.py`, `scripts/run_redteam.py`,
`scripts/export_traces.py`, `scripts/golden_signals.py`, `scripts/dashboard.py`, `scripts/verify_citations.py`.

**Dashboard screenshot.** `reports/dashboard.png` must be a screenshot of the Phoenix UI:
1. Run `python scripts/regenerate_all.py --keep-ui` (or `python -m src.cli run --keep-ui`).
2. Open the `patient-intake-copilot` project at localhost:6006 and save the screenshot as `reports/dashboard.png`.
   `python scripts/capture_dashboard.py` does this automatically if Playwright is installed.

`reports/dashboard_data.csv` is the span data the screenshot shows, and `reports/dashboard_chart.png` is a
chart drawn from that CSV.

## 3. Tests (offline: no API key, no network)

```bash
pytest -q
```

| Test file | What it checks |
|---|---|
| `tests/test_routing.py` | Conditional edges route to the right worker; safety routes take precedence |
| `tests/test_loops.py` | max-steps guard, recursion limit, RAG round cap, tool-failure cascade, timeouts |
| `tests/test_tool_contracts.py` | Every tool's input and output schema plus error paths, and the real MCP server over stdio |
| `tests/test_memory_persistence.py` | Cross-session and cross-restart recall; writes `logs/memory_test.log` |
| `tests/test_guardrails.py` | Injection, cross-patient access, PHI, diagnosis rewrite, citation gating, no PHI in committed evidence |

## Required artifacts → where they are

| Area | Path |
|---|---|
| LangGraph graph (typed state, supervisor + workers, conditional edges, checkpointer, structured output) | `src/graph.py`, `src/state.py`, `src/agents/` |
| MCP server (3 tools, 2 resources) + transcript | `mcp_server/`, `src/tools/mcp_client.py`, `logs/mcp_transcript.jsonl` |
| Context engineering | `src/context/` |
| Tiered memory | `src/memory/`, `tests/test_memory_persistence.py`, `logs/memory_test.log` |
| Agentic-RAG tool + corpus | `src/tools/rag_tool.py`, `src/agents/care_pathway.py`, `data/policy_corpus/` |
| Phoenix instrumentation | `src/observability/tracing.py` (called from `src/cli.py` and `scripts/regenerate_all.py`) |
| Trace export | `traces/phoenix_spans.jsonl` (+ `.parquet` when available), `traces/export_manifest.json` |
| Tool-invocation log | `logs/tool_calls.jsonl` (written by `src/tools/logging_middleware.py`) |
| Failure-mode analysis | `docs/failure-analysis.md`, evidence in `traces/failures/` |
| Golden signals | `reports/golden_signals.json` (`scripts/golden_signals.py`) |
| Cost/latency dashboard | `reports/dashboard.png` (Phoenix screenshot), `reports/dashboard_data.csv`, `reports/dashboard_chart.png` |
| Guardrails | `src/guardrails/` (wired as graph nodes `input_guard` and `output_guard`) |
| Audit trail | `logs/agent_actions.jsonl` (written by `src/audit/audit.py`) |
| Secrets hygiene | `.env.example`, `.gitignore`, secret scan in `scripts/verify_citations.py` |
| Governance pack | `docs/risk-register.md`, `docs/model-card.md`, `docs/compliance.md`, `docs/output-risk.md` |
| Evaluation | `reports/eval_report.json` (`scripts/run_eval.py`), `data/golden/golden_set.jsonl` |
| Agent tests | `tests/test_routing.py`, `tests/test_loops.py`, `tests/test_tool_contracts.py` |
| Bonus | `src/api/app.py` (FastAPI SSE), `reports/redteam_results.json`, PHI before/after in `docs/output-risk.md` |

Architecture and the context-engineering map are in `docs/architecture.md`. The full spec is `SPEC.md`.

## Configuration

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini key. Without it, the copilot runs in **rules-only degraded mode** and the LLM-as-judge is skipped. |
| `GEMINI_MODEL` / `GEMINI_MODEL_LIGHT` | Default `gemini-3.5-flash` / `gemini-3.5-flash-lite` (the 2.5 models are no longer available to new API keys) |
| `PHI_HMAC_SALT` | Salt for `PT-xxxxxxxx` pseudonyms and MCP session tokens |
| `PHOENIX_COLLECTOR_ENDPOINT` | Leave empty to launch Phoenix in-process, or point at `phoenix serve` or Phoenix Cloud (`https://app.phoenix.arize.com/s/<space>`) |
| `PHOENIX_API_KEY` | Only for Phoenix Cloud or an auth-enabled server. It is sent as a bearer token on OTLP export and on the REST span export. |
| `COPILOT_RAG_BACKEND` | `chroma` (default) or `keyword` (offline fallback) |

Limits, guardrail patterns and prices are in `config/limits.yaml`, `config/guardrails.yaml` and
`config/pricing.yaml`.

## Troubleshooting

- **Windows "An Application Control policy has blocked this file" (Smart App Control).** This can block native
  DLLs in `sqlalchemy`, `pyarrow.parquet`, `scikit-learn` and `spacy`. The copilot degrades gracefully:
  - no Phoenix UI: spans still go to `traces/otel_spans_live.jsonl`, and the export falls back to it;
  - no parquet: the export uses `.jsonl`;
  - embeddings use ONNX MiniLM through Chroma;
  - PHI masking falls back to regex.

  For full Phoenix evidence, run on a machine without the policy, or allow the venv in Windows Security.
- **deepeval 2.x** is incompatible with langchain 1.x (`No module named 'langchain.schema'`). Use deepeval ≥ 3.
