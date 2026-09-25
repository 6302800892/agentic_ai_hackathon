"""Custom MCP server (stdio): care-coordination tools + policy/schema resources.

Tools     : get_patient_record, check_coverage, list_available_slots
Resources : policy://intake/rules  (INTAKE-001 text), patients://schema  (masked record JSON schema)

Run standalone:  python -m mcp_server.server
Consumed by   :  src/tools/mcp_client.py via langchain-mcp-adapters.
Every request/response is appended to logs/mcp_transcript.jsonl (side="server"); session tokens are redacted
and records are already masked by mcp_server/store.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server import store  # noqa: E402
from mcp_server.schemas import PatientRecordOut  # noqa: E402

TRANSCRIPT = Path(os.getenv("COPILOT_LOG_DIR") or ROOT / "logs") / "mcp_transcript.jsonl"
mcp = FastMCP("care-coordination", log_level="WARNING")


def _log(method: str, name: str, args: dict, result, latency_ms: float, status: str) -> None:
    safe_args = {k: ("[TOKEN]" if k == "session_token" else v) for k, v in args.items()}
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
           "side": "server", "direction": "response", "method": method, "tool_name": name, "args": safe_args,
           "result": result, "latency_ms": round(latency_ms, 2), "status": status}
    TRANSCRIPT.parent.mkdir(parents=True, exist_ok=True)
    with TRANSCRIPT.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _run(name: str, fn, **args) -> str:
    t0 = time.perf_counter()
    try:
        result = fn(**args)
        status = "denied" if result.get("error") == "ACCESS_DENIED" else ("error" if "error" in result else "ok")
    except Exception as e:  # never crash the server; return a structured error
        result, status = {"error": "INVALID_INPUT", "detail": str(e)}, "error"
    _log("tools/call", name, args, result, (time.perf_counter() - t0) * 1000, status)
    return json.dumps(result)


@mcp.tool()
def get_patient_record(patient_ref: str, session_id: str, session_token: str) -> str:
    """Return the masked, coverage-relevant record for the session's patient (no name/DOB/contact details).
    Denied unless session_token binds session_id to patient_ref."""
    return _run("get_patient_record", store.get_patient_record, patient_ref=patient_ref,
                session_id=session_id, session_token=session_token)


@mcp.tool()
def check_coverage(patient_ref: str, service_code: str, session_id: str, session_token: str,
                   service_date: str | None = None) -> str:
    """Check eligibility of a service for the session's patient using rules COV-R1..COV-R6.
    Returns eligible, plan_status, gaps[{rule_id, description}], referral_required, rules_applied."""
    return _run("check_coverage", store.check_coverage, patient_ref=patient_ref, service_code=service_code,
                session_id=session_id, session_token=session_token, service_date=service_date)


@mcp.tool()
def list_available_slots(pathway_id: str, urgency: str = "routine") -> str:
    """List synthetic appointment slots for a care pathway (e.g. CP-MSK-002, MSK, PEDS, GENERAL, ADMIN)."""
    return _run("list_available_slots", store.list_available_slots, pathway_id=pathway_id, urgency=urgency)


@mcp.resource("policy://intake/rules")
def intake_rules() -> str:
    """Current intake & triage policy (INTAKE-001)."""
    t0 = time.perf_counter()
    text = (ROOT / "data" / "policy_corpus" / "INTAKE-001.md").read_text(encoding="utf-8")
    _log("resources/read", "policy://intake/rules", {}, {"chars": len(text)}, (time.perf_counter() - t0) * 1000, "ok")
    return text


@mcp.resource("patients://schema")
def patient_schema() -> str:
    """JSON schema of the masked patient record returned by get_patient_record."""
    return json.dumps(PatientRecordOut.model_json_schema())


if __name__ == "__main__":
    mcp.run(transport="stdio")
