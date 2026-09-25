# Failure-Mode Analysis

These are real failures observed in our own runs of the copilot. Evidence is preserved with
`scripts/capture_failure.py`, which snapshots the failing run's spans and log records into `traces/failures/`
before the fix is applied. Regenerating the evidence (`scripts/regenerate_all.py`) never deletes those snapshots,
so every citation below keeps resolving. `scripts/verify_citations.py` checks that it does.

**How to read the evidence**
- `run_id` = OpenTelemetry trace id of the request's root span `copilot.request`. It matches the Phoenix
  `context.trace_id` column.
- `span_id` = Phoenix `context.span_id`.
- The snapshot index is `traces/failures/index.json`.

| ID | Failure | Layer | Severity | Status |
|---|---|---|---|---|
| F-01 | Turn 2 of a conversation lost the MSK context from turn 1, so it was routed to the wrong pathway | Context / intake | High (mis-routing) | Fixed |
| F-02 | Proposed appointment date masked as `[DATE]` in the patient's message | Guardrails / PHI | Medium (unusable answer) | Fixed |
| F-03 | Tool names in the MCP transcript were redacted as `[REDACTED]`, which broke AC-07 reconciliation | Observability / audit | Medium (evidence integrity) | Fixed |
| F-04 | Coordinator's Gemini call failed after all retries (503 high demand), so the template wording was used | Model resilience | Low (safe fallback, slower) | Fixed (longer backoff) |
| F-05 | `gemini-3.5-flash` quota exhausted mid-run (429), so every call burned its full retry budget and the regeneration stalled for 40+ minutes | Model resilience / cost | High (pipeline stall) | Fixed (circuit breaker + model switch) |
| F-06 | Free-tier limit of 15 requests/minute hit on `gemini-3.5-flash-lite` (429); calls weren't paced and the provider's `retryDelay` was ignored | Cost & quota governance | High (degraded answers, stalled eval) | Fixed (rate limiter + honour `retryDelay`) |

F-01 to F-03 were captured in rules-only runs, so their traces contain agent and tool spans but no LLM spans.
F-04 to F-06 were captured in Gemini-mode runs, and their snapshots contain the failing `ChatGoogleGenerativeAI`
LLM spans.

---

### F-01 — Multi-turn context lost: follow-up request routed to general practice instead of MSK

- **Evidence.** In run_id `c46622b3734b29546b914da562ae0e36` (request R-013b, session S-013):
  - span_id `c8e33fc004c1df2e` (`intake_classifier`) outputs `reason_for_visit_category: GENERAL`,
    `service_code: PRIMARY_CARE`.
  - span_id `1d53e239bcc23fe5` (`care_pathway`) then retrieves `CP-GEN-001` instead of `CP-MSK-002`.
  - Snapshot files: `traces/failures/F-01_spans.jsonl`, plus the tool-log record in
    `traces/failures/F-01_logs.jsonl` line 3. That record shows the RAG query
    `"general primary care check-up ..."` with `pathway_hint: GENERAL`.
- **Symptom.** Turn 1 of the session was *"I twisted my ankle playing football on Saturday"*. Turn 2 was
  *"Can I get an appointment for it? I prefer afternoons."* The patient was offered a primary-care slot, not the
  musculoskeletal (physiotherapy) pathway.
- **Impact.** Mis-routing, which is the core problem this copilot exists to fix. It violates AC-05 (use facts stated
  earlier in the interaction).
- **Root cause.** The intake rules (`src/agents/heuristics.py::classify`) only used earlier turns when the
  current turn's category was `UNKNOWN`. The word *"appointment"* matched the generic GENERAL rule, so the
  specific category stated in turn 1 (ankle → MSK) was never consulted. The short-term memory was fine: the
  earlier turn was in the checkpointed `messages`. The classifier just didn't give it enough weight.
- **Fix.** `src/agents/heuristics.py::classify` now falls back to the earlier-turn category when the current turn
  is `UNKNOWN` *or* only generic (`GENERAL`). The LLM path already gets the earlier turns through
  `src/context/select.py::context_for_classifier`.
- **Verification.**
  - In the regenerated demo, request R-013b in `reports/demo_outputs.jsonl` has `pathway_id: CP-MSK-002`.
  - `tests/test_memory_persistence.py` asserts the same thing across a full restart
    (`t2.pathway_id == "CP-MSK-002"`).

### F-02 — Output masking destroyed the appointment date

- **Evidence.** In run_id `e05ab2b2f1d08fc6710b54a13b1becf2` (request R-012), span_id `d50b72511d7fe242`
  (`output_guard`) outputs the `patient_message` *"I can offer an appointment at Childrens Clinic on [DATE] at
  09:30 ..."*. The coordinator's audit record, `traces/failures/F-02_logs.jsonl` line 14, shows the slot was chosen
  correctly, so the date was lost in the guard, not in the agent. Snapshot: `traces/failures/F-02_spans.jsonl`.
- **Symptom.** Every schedule or refer answer showed `[DATE]` instead of the slot date.
- **Impact.** The next step can't be acted on (AC-03), and the patient has to call back.
- **Root cause.** `src/guardrails/phi.py::mask_text` applied the date-of-birth recogniser (ISO dates) to *all*
  text. The output guard runs it over system-generated text, where ISO dates are appointment times, not DOBs.
- **Fix.** `mask_text(..., dates=...)` makes date masking explicit:
  - Patient-supplied text at ingress keeps `dates=True`, because a date there may be a DOB.
  - `src/guardrails/output_guard.py` and visit-outcome memories use `dates=False`.
  - Identifier, name, phone and email masking still applies everywhere.
- **Verification.**
  - Request R-012 in `reports/demo_outputs.jsonl` now shows the real slot date.
  - `tests/test_guardrails.py::test_phi_is_masked_at_ingress` still proves that a DOB in patient text is masked.

### F-03 — MCP transcript tool names redacted by the PHI masker

- **Evidence.** In run_id `a0fbb7dcf1962b5ea6a070301693d4b8` (request R-001, MCP transport), the client-side
  transcript record `traces/failures/F-03_logs.jsonl` line 16 has `"name": "[REDACTED]"` for the call made in
  span_id `c38297b7bffd6134` (`tool.get_patient_record`). The span itself names the tool correctly, so the
  problem is limited to the log writer. Snapshot: `traces/failures/F-03_spans.jsonl`.
- **Symptom.** Tool names in `logs/mcp_transcript.jsonl` couldn't be reconciled with the MCP server's tools.
- **Impact.** It breaks AC-07 (the tool log must reconcile with the code) and weakens the audit trail.
- **Root cause.** `src/guardrails/phi.py::mask_obj` redacts any key in `SENSITIVE_KEYS`, which includes `name`
  (a patient's name). The transcript used `name` as the field for the *tool* name, so masking designed for
  patient records hit an operational field.
- **Fix.** The transcript field is now `tool_name` everywhere (`mcp_server/server.py`,
  `src/tools/logging_middleware.py`, `src/tools/mcp_client.py`), which matches `logs/tool_calls.jsonl`. The
  masker stays strict for patient data.
- **Verification.**
  - Every record in `logs/mcp_transcript.jsonl` now carries `tool_name` (for example `"tool_name": "get_patient_record"`).
  - `tests/test_tool_contracts.py::test_committed_tool_log_names_reconcile_with_code` checks reconciliation on
    every run.

### F-04 — Coordinator LLM unavailable (503), fallback wording used

- **Evidence.** In run_id `34f6d3f5b4fdb1da5cdafbb1b078f006` (request R-001, Gemini mode), the LLM span
  span_id `f449a80322aaec88` (`ChatGoogleGenerativeAI`, status ERROR) shows `503 UNAVAILABLE: This model is
  currently experiencing high demand`. It sits under the `coordinator` node span span_id `74f4d42c50297ee9`.
  The audit record `traces/failures/F-04_logs.jsonl` line 14 shows `action: degraded`,
  `decision: llm_unavailable_fallback`. Snapshot: `traces/failures/F-04_spans.jsonl`.
- **Symptom.** The patient still got a correct, cited `schedule` draft, but with template wording, and the request
  took far longer because every attempt waited out the 503.
- **Impact.** Low for safety: the decision is deterministic and the output guard still ran. The cost is latency,
  which is visible in the acting p95 in `reports/golden_signals.json`.
- **Root cause.** A provider-side overload. The retry policy in `src/runtime.py` used a short exponential backoff
  (max 4 s), so all attempts landed inside the same overload spike.
- **Fix.** `src/runtime.py::Runtime._with_retries` now backs off 2–20 s between attempts, and the budget is
  `llm_retries: 3` in `config/limits.yaml`. The deterministic template stays the safe fallback.
- **Verification.** `tests/test_loops.py::test_tool_failure_cascades_to_escalation_not_retry_loop` covers bounded
  retries. Degraded calls are counted as `degraded` actions in `logs/agent_actions.jsonl`.

### F-05 — Model quota exhausted (429): the regeneration stalled

- **Evidence.** In run_id `756c80f3c730ac6cc38c41e7e68254e9` (request R-010, Gemini mode), the LLM span
  span_id `70423eecc81f8cec` (`ChatGoogleGenerativeAI`, status ERROR) shows `429 RESOURCE_EXHAUSTED: You exceeded
  your current quota`. The coordinator node span is span_id `9f9b93cb8ba43387`, and the audit record is
  `traces/failures/F-05_logs.jsonl` line 19. Snapshot: `traces/failures/F-05_spans.jsonl`. Across that run, 18 calls
  degraded and the DeepEval judge (same model) completed no case in 40 minutes.
- **Symptom.** The evidence regeneration never reached its report steps.
- **Impact.** High for delivery: no eval report and no golden signals. There was no safety impact, because every
  request still completed through the fallback path.
- **Root cause.** A per-key quota on `gemini-3.5-flash`. The runtime treated each failure independently, so every
  new call spent its full retry budget against a model that could not answer.
- **Fix.**
  - A **circuit breaker** per model tier in `src/runtime.py::Runtime._with_retries`: after
    `circuit_breaker_failures` consecutive failures (3), calls short-circuit to the fallback for
    `circuit_breaker_cooldown_s` (120 s). Both are set in `config/limits.yaml`, and each short-circuit is audited
    as `degraded` / `circuit_open_fallback`.
  - The run configuration moved to `gemini-3.5-flash-lite` for both tiers. It had quota left and answered
    reliably. See `docs/model-card.md`.
- **Verification.** `tests/test_loops.py::test_llm_circuit_breaker_stops_hammering_a_failing_model` checks that
  calls 4–6 against a failing model never reach it and that the tiers are independent.

### F-06 — Per-minute quota (15 RPM) exceeded: calls not paced

- **Evidence.** In run_id `8c8720c7acf401a84cf52177731a074a` (request R-013a, Gemini mode), the LLM span
  span_id `9513d107dffaa698` (`ChatGoogleGenerativeAI`, status ERROR) shows
  `429 RESOURCE_EXHAUSTED ... quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier, quotaValue: 15`
  with a provider-suggested `retryDelay`. The coordinator's audit record `traces/failures/F-06_logs.jsonl` line 14
  shows `action: degraded`. Snapshot: `traces/failures/F-06_spans.jsonl`. In that run the intake classifier and
  LangMem extraction degraded too, and the eval stage completed 0 cases.
- **Symptom.** Answers after the first few requests used the rule-based fallbacks, and the DeepEval judge stalled.
- **Impact.** High for evidence quality: judge metrics were missing and LLM outputs were replaced by fallbacks.
- **Root cause.** After F-05 every tier (agents, LangMem, judge) shared one model with a hard 15-requests-per-minute
  quota. Nothing paced the calls, and the retry policy used its own backoff instead of the provider's
  `retryDelay`, so retries landed while the window was still full.
- **Fix.**
  - `src/ratelimit.py` adds a per-model sliding-window limiter capped at `gemini_rpm: 12` in `config/limits.yaml`,
    below the quota.
  - Every Gemini call acquires a slot from it: agents via `src/runtime.py::Runtime._with_retries`, LangMem in
    `src/memory/long_term.py`, and the judge in `scripts/run_eval.py`.
  - On a 429, `_with_retries` waits exactly the `retryDelay` the provider returns.
- **Verification.**
  - `tests/test_loops.py::test_rate_limiter_caps_requests_per_minute` checks the pacing.
  - `tests/test_loops.py::test_retry_delay_is_read_from_provider_429` checks the delay parsing.
  - The degraded count in `logs/agent_actions.jsonl` from the rate-limited regeneration shows the effect.

---

## Cross-cutting lessons

1. **Masking must know what it is masking.** F-02 and F-03 were both caused by privacy controls applied without
   context. We now separate *patient-supplied* text (strict) from *system-generated* text and operational
   fields.
2. **Rules need the same context as the model.** F-01 showed that a deterministic fallback can quietly ignore
   short-term memory. The fallback and the LLM path now get the same selected context
   (`src/context/select.py`).
3. **Fail fast on an unhealthy dependency, and pace a healthy one.** F-04 to F-06 showed that retries alone turn a provider outage into a
   pipeline stall. A circuit breaker bounds the total cost, and a rate limiter keeps the pipeline inside its quota.
4. **Capture before fixing.** Evidence is snapshotted with `scripts/capture_failure.py` before the code changes.

## Adding a failure (Gemini-mode runs)

1. Reproduce it with `python -m src.cli run --input <file>` and find the request in Phoenix (localhost:6006) or
   in `logs/agent_actions.jsonl`.
2. Snapshot the evidence **before** fixing:
   `python scripts/capture_failure.py --name F-04 --request-id <id> --note "<symptom>"`.
3. Write it up here using the same fields as above, then run `python scripts/verify_citations.py`.
