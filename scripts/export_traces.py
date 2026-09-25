"""Export spans -> traces/phoenix_spans.parquet (+ traces/phoenix_spans.jsonl) and traces/export_manifest.json.

Source preference:
  1. Phoenix `get_spans_dataframe()` (in-process Phoenix or PHOENIX_COLLECTOR_ENDPOINT)       source="phoenix"
  2. traces/otel_spans_live.jsonl - the same OTel spans, written by src/observability/tracing.py
     JsonlSpanExporter in the same TracerProvider                                              source="otel_jsonl_mirror"
The manifest records which source was used and validates the export (full run, multiple agents, every tool).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import TRACES  # noqa: E402
from src.observability.spans import AGENT_NODES, normalize, read_raw  # noqa: E402
from src.observability.tracing import LIVE_SPANS, get_spans_dataframe  # noqa: E402

PARQUET = TRACES / "phoenix_spans.parquet"
JSONL = TRACES / "phoenix_spans.jsonl"
MANIFEST = TRACES / "export_manifest.json"
EXPECTED_TOOLS = {"search_care_policy", "get_patient_record", "check_coverage", "list_available_slots"}


def _jsonable(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def run_started_at():
    """Earliest span in the live mirror = start of the current regeneration (the mirror is reset per run)."""
    from datetime import datetime
    first = None
    if LIVE_SPANS.exists():
        with LIVE_SPANS.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = datetime.fromisoformat(json.loads(line)["start_time"])
                    first = t if first is None or t < first else first
    return first


def export(project: str | None = None) -> dict:
    TRACES.mkdir(parents=True, exist_ok=True)
    df, source = get_spans_dataframe(project, since=run_started_at()), "phoenix"
    if df is None or len(df) == 0:
        df, source = read_raw(LIVE_SPANS), "otel_jsonl_mirror"
    if "context.span_id" not in df.columns:
        df = df.reset_index()

    records = _jsonable(df)
    JSONL.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    parquet_written, parquet_error = False, None
    try:
        out = df.copy()
        for c in out.columns:  # nested attribute dicts -> JSON strings so parquet is portable
            if out[c].map(lambda v: isinstance(v, (dict, list))).any():
                out[c] = out[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
        out.to_parquet(PARQUET, index=False)
        parquet_written = True
    except Exception as e:
        parquet_error = f"{type(e).__name__}: {str(e)[:200]}"

    n = normalize(read_raw(JSONL))
    roots = n[n["name"] == "copilot.request"]
    tools_seen = {x.replace("tool.", "") for x in n.loc[n["category"] == "tool", "name"]}
    per_trace_agents = n[n["name"].isin(AGENT_NODES)].groupby("trace_id")["name"].nunique()
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": "scripts/export_traces.py",
        "source": source,
        "phoenix_spans": int(len(df)) if source == "phoenix" else 0,
        "mirror_spans": sum(1 for l in LIVE_SPANS.open(encoding="utf-8") if l.strip()) if LIVE_SPANS.exists() else 0,
        "files": {"jsonl": str(JSONL.relative_to(ROOT)).replace("\\", "/"),
                  "parquet": str(PARQUET.relative_to(ROOT)).replace("\\", "/") if parquet_written else None},
        "parquet_error": parquet_error,
        "spans": int(len(n)), "runs": int(roots["trace_id"].nunique()),
        "llm_spans": int((n["category"] == "thinking").sum()),
        "tool_spans": int((n["category"] == "tool").sum()),
        "tools_seen": sorted(tools_seen),
        "checks": {
            "has_full_run": bool(len(roots) > 0),
            "max_agents_in_one_run": int(per_trace_agents.max()) if len(per_trace_agents) else 0,
            "multiple_agents_in_a_run": bool(len(per_trace_agents) and per_trace_agents.max() >= 3),
            "every_tool_called": EXPECTED_TOOLS.issubset(tools_seen),
            "latencies_present": bool(n["latency_ms"].notna().all()),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
