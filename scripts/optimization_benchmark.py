"""Before/after optimization benchmark -> two Phoenix-derived golden-signals reports + docs/optimization.md.

Optimization under test (config/limits.yaml `rag_llm_grading`):
  before = "always"        the care-pathway agent asks Gemini to grade the retrieved policy on every RAG round
  after  = "on_rule_miss"  the LLM grader is skipped when the deterministic check already found the expected
                           pathway policy; the LLM still grades whenever the rules miss

Each variant runs the committed sample inputs (data/samples/*.jsonl) in its OWN process, with its own log dir,
span mirror, checkpoint and memory store, so the variants cannot contaminate each other or the main evidence.
Spans are pulled back from Phoenix (falls back to the variant's OTel mirror) and fed to scripts/golden_signals.py.

    python scripts/optimization_benchmark.py            # runs both variants, then writes docs/optimization.md
    python scripts/optimization_benchmark.py --report   # rebuild docs/optimization.md from existing results
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "reports" / "optimization"
VARIANTS = {"before": "always", "after": "on_rule_miss"}
SAMPLES = [ROOT / "data/samples/intake_requests.jsonl", ROOT / "data/samples/session_return_visit.jsonl"]


def run_variant(name: str) -> None:
    """Runs in a child process: env must be set before any `src` import."""
    vdir = OUTDIR / name
    (vdir / "logs").mkdir(parents=True, exist_ok=True)
    for f in list((vdir / "logs").glob("*.jsonl")) + [vdir / "spans_mirror.jsonl"]:
        f.write_text("", encoding="utf-8")
    os.environ["COPILOT_LOG_DIR"] = str(vdir / "logs")
    sys.path.insert(0, str(ROOT))

    import asyncio

    from scripts.golden_signals import compute
    from src.cli import read_jsonl
    from src.config import get_settings
    from src.observability.tracing import flush, get_spans_dataframe, init_tracing
    from src.service import Copilot

    get_settings().limits["rag_llm_grading"] = VARIANTS[name]
    started = datetime.now(timezone.utc)
    init_tracing(launch_ui=False, mirror_path=vdir / "spans_mirror.jsonl")

    async def go() -> list[dict]:
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            async with Copilot(checkpoint_path=Path(tmp) / "cp.sqlite", memory_path=Path(tmp) / "mem.sqlite") as cp:
                for path in SAMPLES:
                    for r in read_jsonl(path):
                        f = await cp.handle(r["request_id"], r["session_id"], r["patient_id"], r["text"])
                        rows.append({"request_id": r["request_id"], **f.model_dump()})
                        print(f"[{name}] {r['request_id']}: {f.action} {f.pathway_id}", flush=True)
        return rows

    rows = asyncio.run(go())
    flush()
    (vdir / "outputs.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in rows), encoding="utf-8")

    run_ids = {x["run_id"] for x in rows}
    df, source = get_spans_dataframe(since=started), "phoenix"
    records = []
    if df is not None and len(df):
        records = [r for r in json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))
                   if r.get("context.trace_id") in run_ids]
    if not records:
        source = "otel_jsonl_mirror"
        records = [json.loads(l) for l in (vdir / "spans_mirror.jsonl").read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    (vdir / "spans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    compute(vdir / "spans.jsonl", out=vdir / "golden_signals.json", span_source=source,
            tool_log=vdir / "logs" / "tool_calls.jsonl")
    print(f"[{name}] spans={len(records)} source={source}", flush=True)


def _pct(before: float, after: float) -> str:
    return "n/a" if not before else f"{(after - before) / before * 100:+.1f}%"


def write_note() -> Path:
    sys.path.insert(0, str(ROOT))
    from src.observability.spans import load_spans

    g = {v: json.loads((OUTDIR / v / "golden_signals.json").read_text(encoding="utf-8")) for v in VARIANTS}
    out = {v: {r["request_id"]: r for r in (json.loads(l) for l in (OUTDIR / v / "outputs.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip())} for v in VARIANTS}
    sp = {v: load_spans(OUTDIR / v / "spans.jsonl") for v in VARIANTS}

    def llm(df, agent=None, ok=True):
        m = df["category"] == "thinking"
        if agent:
            m &= df["agent"] == agent
        m &= (df["status"] != "ERROR") if ok else (df["status"] == "ERROR")
        return int(m.sum())

    ids = sorted(set(out["before"]) & set(out["after"]))
    same = [i for i in ids if (out["before"][i]["action"], out["before"][i]["pathway_id"],
                               [x["rule_id"] for x in out["before"][i]["coverage_gaps"]]) ==
            (out["after"][i]["action"], out["after"][i]["pathway_id"],
             [x["rule_id"] for x in out["after"][i]["coverage_gaps"]])]
    diff = [i for i in ids if i not in same]
    b, a = g["before"], g["after"]
    rows = [
        ("Successful LLM calls (all agents)", llm(sp["before"]), llm(sp["after"])),
        ("Successful LLM calls in care_pathway", llm(sp["before"], "care_pathway"), llm(sp["after"], "care_pathway")),
        ("Provider errors (503/504, retried)", llm(sp["before"], ok=False), llm(sp["after"], ok=False)),
        ("Tokens (prompt + completion)", b["tokens"]["total"], a["tokens"]["total"]),
        ("Estimated cost (USD)", b["cost_usd"]["total"], a["cost_usd"]["total"]),
        ("End-to-end latency p50 (ms)", b["latency_ms"]["end_to_end"].get("p50"), a["latency_ms"]["end_to_end"].get("p50")),
        ("End-to-end latency p95 (ms)", b["latency_ms"]["end_to_end"].get("p95"), a["latency_ms"]["end_to_end"].get("p95")),
        ("care_pathway node latency p50 (ms)", (b["latency_ms"]["by_agent"].get("care_pathway") or {}).get("p50"),
         (a["latency_ms"]["by_agent"].get("care_pathway") or {}).get("p50")),
    ]
    table = "\n".join(f"| {n} | {bv} | {av} | {_pct(bv or 0, av or 0)} |" for n, bv, av in rows)
    note = f"""# Optimization Note — rule-first relevance grading in agentic RAG

*Generated by `scripts/optimization_benchmark.py` on {datetime.now(timezone.utc).isoformat(timespec='seconds')}.
Do not edit by hand; rerun the script.*

## Change
The care-pathway agent (`src/agents/care_pathway.py`) retrieves policy, then asks Gemini whether the excerpts are
sufficient, before deciding whether to rewrite the query and retrieve again. The deterministic `rule_grade` check
already overrides the model when the expected pathway policy is missing. When the rules *confirm* the expected
policy is present, the LLM grading call adds cost and latency without being able to change the outcome safely.

- **before**: `rag_llm_grading: always` (the setting the committed evidence was generated with)
- **after**: `rag_llm_grading: on_rule_miss`, so the LLM grades only when the rules do not find the expected policy

Setting: `config/limits.yaml`. Benchmark: the committed sample inputs (`data/samples/intake_requests.jsonl` and
`data/samples/session_return_visit.jsonl`, {len(ids)} requests), each variant in its own process with a fresh
memory store. Model: `{b['tokens']['by_model'] and next(iter(b['tokens']['by_model']))}`. Span source: before =
`{b['sources']['span_source']}`, after = `{a['sources']['span_source']}`.

## Measured result (two Phoenix-derived golden-signals reports)

| Signal | before | after | change |
|---|---|---|---|
{table}

Reports: `reports/optimization/before/golden_signals.json` and `reports/optimization/after/golden_signals.json`.
Spans: `reports/optimization/before/spans.jsonl` and `reports/optimization/after/spans.jsonl`.

**Quality check.** {len(same)}/{len(ids)} requests produced the identical action, pathway and coverage-gap rules
in both variants{(' (differences: ' + ', '.join(diff) + ')') if diff else ''}. Outputs:
`reports/optimization/before/outputs.jsonl` and `reports/optimization/after/outputs.jsonl`.

**Caveats.**
- Successful LLM calls and tokens are the robust signal. Provider errors (Google 503/504 during the run) are
  retried attempts and are shown separately so they don't distort the comparison.
- Latency also depends on the provider and on the client-side rate limiter (`src/ratelimit.py`, 12 requests per
  minute), so single-run latency differences are indicative, not statistically tested.
- The committed main evidence (`reports/golden_signals.json`) was produced with `always`. Switch the setting to
  `on_rule_miss` to adopt the optimization.
"""
    path = ROOT / "docs" / "optimization.md"
    path.write_text(note, encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--report", action="store_true", help="only rebuild docs/optimization.md")
    a = ap.parse_args()
    if a.variant:
        run_variant(a.variant)
        return 0
    if not a.report:
        for v in VARIANTS:
            rc = subprocess.run([sys.executable, "-u", __file__, "--variant", v], cwd=ROOT).returncode
            if rc:
                return rc
    print("wrote", write_note())
    return 0


if __name__ == "__main__":
    sys.exit(main())
