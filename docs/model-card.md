# Model / System Card — Patient Intake & Care-Coordination Copilot

## System overview
A LangGraph multi-agent copilot for front-desk and care-coordination staff. It takes in a synthetic patient
request and does four things:
1. Classifies the reason for visit.
2. Verifies coverage.
3. Retrieves and cites the applicable care-pathway policy.
4. Drafts a next step: `schedule`, `refer`, `escalate`, `clarify` or `decline`.

A clinician keeps every clinical decision. Architecture: `docs/architecture.md`; graph: `src/graph.py`.

## Models
All models are Google Gemini, the only provider used.

| Use | Model (env var) | Settings |
|---|---|---|
| Intake classification, RAG relevance grading, summarization, LangMem fact extraction | `GEMINI_MODEL_LIGHT` (default `gemini-3.5-flash-lite`) | temperature 0, structured output (Pydantic), 60 s timeout, 4 retries |
| Coordinator wording (the decision itself is rule-based) | `GEMINI_MODEL` (default `gemini-3.5-flash`) | temperature 0, structured output `DraftText` |
| Evaluation judge (DeepEval) | `GEMINI_MODEL` | `scripts/run_eval.py::make_judge` |

**Run configuration used for the committed evidence.** Both tiers used `gemini-3.5-flash-lite`: the
`gemini-3.5-flash` quota for the key was exhausted (see `docs/failure-analysis.md` F-05). `system_model` and
`judge_model` in `reports/eval_report.json` record the models that actually ran.

**Non-LLM components**
- Embeddings: Sentence-Transformers `all-MiniLM-L6-v2`, or the same model via ONNX.
- Vector store: Chroma.
- PHI detection: Presidio plus deterministic recognisers.
- Decisions: coverage rules (`mcp_server/store.py`), routing (`src/agents/supervisor.py`) and next-step selection
  (`src/agents/coordinator.py`) are deterministic, so they can be audited.

**Degraded mode.** Without a key, or when Gemini fails, the classifier and grader fall back to rules in
`src/agents/heuristics.py`. The output is tagged `source="heuristic_fallback"` and a `degraded` audit record is
written.

## Data
- **Synthetic only.** It is produced by `scripts/generate_synthetic_data.py`: 25 patients, 11 policies, samples,
  a 24-case golden set and 15 red-team attacks. No real patient data was used for building, testing or evaluation.
- Identifiers are pseudonymised as `PT-xxxxxxxx` (HMAC) before storage or tracing (`src/guardrails/phi.py`).
- The models are not fine-tuned; the system uses prompting, retrieval and rules.

## Intended use
- **Users:** trained front-desk and care-coordination staff. Patients see drafts only after staff review.
- **Tasks:** intake triage *routing* (not clinical triage), eligibility checks against plan rules, drafting a
  booking, referral or escalation with policy citations, and recalling scheduling preferences.

## Out of scope
- Diagnosis, symptom interpretation, medication or dosing advice, or clinical prioritisation beyond "escalate
  now".
- Emergency response. The system escalates and advises calling emergency services; it doesn't dispatch.
- Real PHI, real EHR or scheduling integrations, and autonomous booking.
- Languages other than English. Preferences are recorded, but the classifier is English-only.

## Evaluation
- Harness: `scripts/run_eval.py`. Report: `reports/eval_report.json`. Golden set: `data/golden/golden_set.jsonl`.
- **Deterministic agent metrics:** intent and action accuracy, escalation recall (gate = 1.0), coverage-rule
  accuracy, policy recall, citation validity, no-diagnosis rate.
- **LLM-as-judge (Gemini via DeepEval):** hallucination, faithfulness, answer relevancy, and a
  `NoDiagnosisSafeRouting` GEval metric.
- **Red team:** `reports/redteam_results.json` covers injection, cross-patient access, prompt leak, tag escape,
  diagnosis coercion and PHI exfiltration.
- **Operations:** `reports/golden_signals.json` reports latency, tokens and cost.
- Check `judge_status` in the eval report. When it says `skipped`, the committed report was produced without an
  API key, so only the deterministic metrics are present.

## Known limitations and failure modes
- **Red flags are a finite vocabulary.** Unusual descriptions of emergencies can reach the model unflagged. The
  model is instructed to classify them as urgent, but recall outside the golden set isn't guaranteed.
- **Rules-mode classification is keyword-based.** It gets the golden set right but is brittle on paraphrases.
  Gemini mode is the intended configuration.
- **Presidio NER can miss unusual names in free text.** The synthetic identifier formats are always caught.
- **Observed failure modes:** see `docs/failure-analysis.md`.
  - F-01: multi-turn context was lost, so the patient was mis-routed.
  - F-02: over-masking made the answer unusable.
  - F-03: masking corrupted the audit evidence.
  - F-04 and F-05: provider overload and quota exhaustion. They are handled by fallback, backoff and a circuit
    breaker.
- The reference date for coverage periods is fixed (`COPILOT_REFERENCE_DATE`) so runs are reproducible.

## Human oversight
- The output-risk tiers and gates are in `docs/output-risk.md`.
- Optional HITL interrupt in `src/agents/handoff.py::human_escalation`.
- Every response carries an AI disclosure: `src/state.py::FinalResponse.disclosure`.
- Every consequential action is audited in `logs/agent_actions.jsonl`.
