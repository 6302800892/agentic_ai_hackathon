"""Golden-signals report (AC-09) -> reports/golden_signals.json

Reads the exported spans (traces/phoenix_spans.parquet, else .jsonl), computes:
  latency   p50/p95/max by span category (thinking = LLM, acting = graph nodes, tool = tool calls),
            by agent, by tool, and end-to-end per request
  tokens    prompt / completion / total from LLM spans (llm.token_count.*), by model and agent
  cost      tokens x config/pricing.yaml, per run / agent / model
  traffic   runs, LLM calls, tool calls;  errors: error-span rate + tool status mix (logs/tool_calls.jsonl)
  quality   accuracy + hallucination rate imported from reports/eval_report.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import LOGS, REPORTS, TRACES, get_settings  # noqa: E402
from src.observability.spans import load_spans  # noqa: E402

OUT = REPORTS / "golden_signals.json"


def spans_file() -> Path:
    pq = TRACES / "phoenix_spans.parquet"
    return pq if pq.exists() else TRACES / "phoenix_spans.jsonl"


def pct(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "p50": round(float(s.quantile(0.5)), 2), "p95": round(float(s.quantile(0.95)), 2),
            "max": round(float(s.max()), 2), "mean": round(float(s.mean()), 2)}


def compute(spans_path: Path | None = None) -> dict:
    path = spans_path or spans_file()
    df = load_spans(path)
    manifest_path = TRACES / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    llm = df[df["category"] == "thinking"]
    tools = df[df["category"] == "tool"]
    acting = df[(df["category"] == "acting") & (df["name"] != "copilot.request")]
    roots = df[df["name"] == "copilot.request"]
    runs = int(roots["trace_id"].nunique())

    tool_status = Counter()
    tool_log = LOGS / "tool_calls.jsonl"
    if tool_log.exists():
        for line in tool_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tool_status[json.loads(line)["status"]] += 1

    eval_path = REPORTS / "eval_report.json"
    ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    agg = ev.get("aggregate", {})

    by = lambda frame, key, col: {str(k): pct(g[col]) for k, g in frame.groupby(key)}  # noqa: E731
    total_cost = float(df["cost_usd"].sum())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": "scripts/golden_signals.py",
        "sources": {"spans": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "span_source": manifest.get("source", "unknown"),
                    "eval_report": "reports/eval_report.json" if ev else None,
                    "tool_log": "logs/tool_calls.jsonl", "pricing": "config/pricing.yaml",
                    "pricing_as_of": get_settings().pricing.get("as_of")},
        "traffic": {"runs": runs, "spans": int(len(df)), "llm_calls": int(len(llm)), "tool_calls": int(len(tools))},
        "latency_ms": {
            "end_to_end": pct(roots["latency_ms"]),
            "by_category": {"thinking": pct(llm["latency_ms"]), "acting": pct(acting["latency_ms"]),
                            "tool": pct(tools["latency_ms"])},
            "by_agent": by(acting, "name", "latency_ms"),
            "by_tool": by(tools, "name", "latency_ms"),
        },
        "tokens": {
            "prompt": int(llm["prompt_tokens"].sum()), "completion": int(llm["completion_tokens"].sum()),
            "total": int(llm["total_tokens"].sum()),
            "per_run_avg": round(float(llm["total_tokens"].sum()) / runs, 1) if runs else 0,
            "by_model": {str(k): int(v) for k, v in llm.groupby(llm["model"].fillna("unknown"))["total_tokens"].sum().items()},
            "by_agent": {str(k): int(v) for k, v in llm.groupby(llm["agent"].fillna("unknown"))["total_tokens"].sum().items()},
        },
        "cost_usd": {
            "total": round(total_cost, 6), "per_run_avg": round(total_cost / runs, 6) if runs else 0,
            "by_model": {str(k): round(float(v), 6) for k, v in llm.groupby(llm["model"].fillna("unknown"))["cost_usd"].sum().items()},
            "by_agent": {str(k): round(float(v), 6) for k, v in llm.groupby(llm["agent"].fillna("unknown"))["cost_usd"].sum().items()},
        },
        "errors": {"error_span_rate": round(float((df["status"] == "ERROR").mean()), 4) if len(df) else 0,
                   "tool_status_counts": dict(tool_status)},
        "quality": {"accuracy": agg.get("accuracy"), "intent_accuracy": agg.get("intent_accuracy"),
                    "action_accuracy": agg.get("action_accuracy"), "escalation_recall": agg.get("escalation_recall"),
                    "hallucination_rate": ev.get("hallucination_rate"), "judge_status": ev.get("judge_status")},
    }
    if not len(llm):
        report["notes"] = ["No LLM spans in this export: the run used rules-only mode (no GOOGLE_API_KEY), so "
                           "thinking latency, tokens and cost are zero. Re-run regenerate_all with a key."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(compute(), indent=2))
