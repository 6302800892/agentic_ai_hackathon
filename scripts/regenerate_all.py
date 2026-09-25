"""Regenerate ALL evidence from committed inputs with one command (NFR-02, NFR-06).

    python scripts/regenerate_all.py [--no-phoenix] [--no-judge] [--max-eval-cases N] [--keep-ui]

Steps (one process, one tracer, so every artifact shares run_ids):
  0. reset evidence logs + local state (traces/failures/ is preserved)
  1. synthetic data (if missing) + vector index
  2. run_demo      -> reports/demo_outputs.jsonl (+ tool log, audit trail, MCP transcript, spans)
  3. run_eval      -> reports/eval_report.json
  4. run_redteam   -> reports/redteam_results.json
  5. export_traces -> traces/phoenix_spans.parquet|.jsonl + traces/export_manifest.json
  6. golden_signals-> reports/golden_signals.json
  7. dashboard     -> reports/dashboard_data.csv + reports/dashboard_chart.png (+ dashboard.png via Playwright)
  8. pytest        -> agent tests (writes logs/memory_test.log)
  9. verify_citations -> reports/citation_check.json
 10. reports/manifest.json (file, rows, sha256, producer)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.config import DATA, LOGS, REPORTS, STATE_DIR, TRACES, get_settings  # noqa: E402

EVIDENCE_LOGS = [LOGS / "tool_calls.jsonl", LOGS / "agent_actions.jsonl", LOGS / "mcp_transcript.jsonl",
                 TRACES / "otel_spans_live.jsonl"]
PRODUCERS = {
    "logs/tool_calls.jsonl": "src/tools/logging_middleware.py",
    "logs/agent_actions.jsonl": "src/audit/audit.py",
    "logs/mcp_transcript.jsonl": "mcp_server/server.py + src/tools/logging_middleware.py",
    "logs/memory_test.log": "tests/test_memory_persistence.py",
    "traces/otel_spans_live.jsonl": "src/observability/tracing.py",
    "traces/phoenix_spans.jsonl": "scripts/export_traces.py",
    "traces/phoenix_spans.parquet": "scripts/export_traces.py",
    "traces/export_manifest.json": "scripts/export_traces.py",
    "reports/demo_outputs.jsonl": "scripts/run_demo.py",
    "reports/eval_report.json": "scripts/run_eval.py",
    "reports/redteam_results.json": "scripts/run_redteam.py",
    "reports/golden_signals.json": "scripts/golden_signals.py",
    "reports/dashboard_data.csv": "scripts/dashboard.py",
    "reports/dashboard_chart.png": "scripts/dashboard.py",
    "reports/dashboard.png": "scripts/capture_dashboard.py (Phoenix UI screenshot)",
    "reports/citation_check.json": "scripts/verify_citations.py",
}


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def reset() -> None:
    for p in EVIDENCE_LOGS:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    for name in ("checkpoints.sqlite", "memory.sqlite"):
        for suffix in ("", "-wal", "-shm"):
            f = STATE_DIR / (name + suffix)
            if f.exists():
                f.unlink()


def manifest() -> dict:
    files = {}
    for rel, producer in PRODUCERS.items():
        p = ROOT / rel
        if not p.exists():
            files[rel] = {"present": False, "producer": producer}
            continue
        data = p.read_bytes()
        rows = data.count(b"\n") if p.suffix in (".jsonl", ".csv", ".log") else None
        files[rel] = {"present": True, "bytes": len(data), "rows": rows, "producer": producer,
                      "sha256": hashlib.sha256(data).hexdigest()}
    m = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "command": "python scripts/regenerate_all.py", "mode": "gemini" if get_settings().has_llm else "rules-only",
         "files": files}
    (REPORTS / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


async def run_agents(a) -> None:
    from scripts.run_demo import run_demo
    from scripts.run_eval import run_eval
    from scripts.run_redteam import run_redteam
    from src.service import Copilot

    async with Copilot(transport=a.transport) as cp:
        print(f"mode={cp.mode} tools={cp.transport_used}")
        step("2. demo run (samples + multi-turn + return visits)")
        await run_demo(cp)
        step("3. evaluation (golden set)")
        rep = await run_eval(cp, judge=not a.no_judge, max_cases=a.max_eval_cases)
        print(json.dumps(rep["aggregate"]), "| judge:", rep["judge_status"])
        step("4. red team")
        rt = await run_redteam(cp, verbose=False)
        print(f"defense rate {rt['defense_rate']} ({rt['defended']}/{rt['attacks']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-phoenix", action="store_true", help="do not launch Phoenix in-process")
    ap.add_argument("--no-judge", action="store_true", help="skip DeepEval LLM-as-judge metrics")
    ap.add_argument("--max-eval-cases", type=int)
    ap.add_argument("--transport", default="mcp")
    ap.add_argument("--keep-ui", action="store_true", help="keep Phoenix open at the end (screenshot)")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--reports-only", action="store_true",
                    help="re-run steps 5-10 on the evidence of the last run (no reset, no agent calls)")
    a = ap.parse_args()

    from src.observability.tracing import flush, init_tracing, phoenix_url
    if a.reports_only:
        init_tracing(launch_ui=False)  # connects to PHOENIX_COLLECTOR_ENDPOINT for the export only
        print("reports-only; phoenix:", phoenix_url() or "not configured")
    else:
        step("0. reset evidence logs + local state")
        reset()
        step("1. synthetic data + index")
        if not (DATA / "patients" / "patients.json").exists():
            subprocess.run([sys.executable, "scripts/generate_synthetic_data.py"], check=True)
        if get_settings().rag_backend == "chroma":
            subprocess.run([sys.executable, "scripts/build_index.py"], check=False)
        init_tracing(launch_ui=not a.no_phoenix)
        print("phoenix:", phoenix_url() or "not running (OTel JSONL mirror only)")
        asyncio.run(run_agents(a))
        flush()

    step("5. export traces")
    from scripts.export_traces import export
    print(json.dumps(export()["checks"]))
    step("6. golden signals")
    from scripts.golden_signals import compute
    gs = compute()
    print(json.dumps({"latency": gs["latency_ms"]["by_category"], "tokens": gs["tokens"]["total"],
                      "cost_usd": gs["cost_usd"]["total"], "quality": gs["quality"]}))
    step("7. dashboard")
    from scripts.dashboard import build
    build()
    if phoenix_url():
        from scripts.capture_dashboard import capture
        capture(phoenix_url())

    rc = 0
    if not a.skip_tests:
        step("8. agent tests")
        rc |= subprocess.run([sys.executable, "-m", "pytest", "-q"]).returncode
    step("9. verify citations / secrets / PHI")
    rc |= subprocess.run([sys.executable, "scripts/verify_citations.py"]).returncode
    step("10. manifest")
    m = manifest()
    missing = [k for k, v in m["files"].items() if not v["present"]]
    print("missing artifacts:", missing or "none")
    if a.keep_ui and phoenix_url():
        print(f"Phoenix UI at {phoenix_url()} - take the dashboard screenshot, then press Enter.")
        sys.stdin.readline()
    return rc


if __name__ == "__main__":
    sys.exit(main())
