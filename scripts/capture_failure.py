"""Preserve the evidence for a real failure BEFORE fixing it (traces/logs are regenerated later).

Copies every span of the failing run (trace) plus its tool-log / audit / MCP-transcript records into
traces/failures/<name>_spans.jsonl and traces/failures/<name>_logs.jsonl, and appends an index entry to
traces/failures/index.json. docs/failure-analysis.md cites these files, so citations keep resolving.

    python scripts/capture_failure.py --name F-01 --request-id R-013b [--occurrence 1]
    python scripts/capture_failure.py --name F-02 --run-id <32-hex trace id>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_FILES = ["logs/tool_calls.jsonl", "logs/agent_actions.jsonl", "logs/mcp_transcript.jsonl"]
SPAN_SOURCES = ["traces/otel_spans_live.jsonl", "traces/phoenix_spans.jsonl"]
OUT = ROOT / "traces" / "failures"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_ids_for_request(request_id: str) -> list[str]:
    ids = []
    for rec in _jsonl(ROOT / "logs" / "agent_actions.jsonl"):
        if rec.get("request_id") == request_id and rec.get("run_id") and rec["run_id"] not in ids:
            ids.append(rec["run_id"])
    return ids


def capture(name: str, run_id: str, note: str = "") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    spans = []
    for src in SPAN_SOURCES:
        spans += [s for s in _jsonl(ROOT / src) if s.get("context.trace_id") == run_id]
        if spans:
            break
    logs = []
    for lf in LOG_FILES:
        for i, rec in enumerate(_jsonl(ROOT / lf), start=1):
            if rec.get("run_id") == run_id:
                logs.append({"source": lf, "line": i, "record": rec})
    (OUT / f"{name}_spans.jsonl").write_text("".join(json.dumps(s) + "\n" for s in spans), encoding="utf-8")
    (OUT / f"{name}_logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs), encoding="utf-8")
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    index[name] = {"run_id": run_id, "spans": len(spans), "log_records": len(logs), "note": note,
                   "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "span_names": sorted({s["name"] for s in spans})}
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--request-id")
    ap.add_argument("--occurrence", type=int, default=1, help="which run of that request (1 = first)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    run_id = a.run_id
    if not run_id:
        ids = run_ids_for_request(a.request_id)
        if len(ids) < a.occurrence:
            sys.exit(f"no run #{a.occurrence} found for request {a.request_id} (found {len(ids)})")
        run_id = ids[a.occurrence - 1]
    print(json.dumps({a.name: capture(a.name, run_id, a.note)}, indent=2))


if __name__ == "__main__":
    main()
