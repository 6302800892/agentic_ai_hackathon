"""Coverage-check agent: verifies eligibility from the synthetic record via the MCP server.

Eligibility is decided deterministically by the MCP `check_coverage` tool (rules COV-R1..R6); the agent never
lets a model decide coverage. Any gap is reported with the rule that produced it (AC-02).
"""
from __future__ import annotations

from src.context.isolate import merge_result
from src.state import CopilotState, CoverageGap, CoverageResult, ErrorRecord

AGENT = "coverage_checker"


def explain(data: dict) -> str:
    if data["eligible"]:
        return f"Eligible for {data['service_code']} (rules applied: {', '.join(data['rules_applied'])})."
    return "Coverage gaps: " + "; ".join(f"{g['rule_id']} - {g['description']}" for g in data["gaps"])


async def run(state: CopilotState, runtime) -> dict:
    intake = state["intake"]
    auth = {"patient_ref": state["patient_ref"], "session_id": state["session_id"],
            "session_token": state["session_token"]}
    errors = list(state.get("errors") or [])

    record = await runtime.tools.call("get_patient_record", auth, agent=AGENT)
    if not record.ok:
        errors.append(ErrorRecord(node=AGENT, error=f"get_patient_record {record.status}: {record.error}", fatal=True))
        return merge_result(AGENT, {"errors": errors, "coverage": CoverageResult(
            eligible=False, plan_status="unknown", service_code=intake.service_code,
            gaps=[CoverageGap(rule_id="TOOL-" + record.status.upper(), description=record.error or "")])})

    cov = await runtime.tools.call("check_coverage", {**auth, "service_code": intake.service_code}, agent=AGENT)
    if not cov.ok:
        errors.append(ErrorRecord(node=AGENT, error=f"check_coverage {cov.status}: {cov.error}", fatal=True))
        return merge_result(AGENT, {"errors": errors, "coverage": CoverageResult(
            eligible=False, plan_status=record.data.get("plan_status", "unknown"), service_code=intake.service_code,
            gaps=[CoverageGap(rule_id="TOOL-" + cov.status.upper(), description=cov.error or "")])})

    d = cov.data
    return merge_result(AGENT, {"coverage": CoverageResult(
        eligible=d["eligible"], plan_status=d["plan_status"], service_code=d["service_code"],
        gaps=[CoverageGap(**g) for g in d["gaps"]], referral_required=d["referral_required"],
        explanation=explain(d))})
