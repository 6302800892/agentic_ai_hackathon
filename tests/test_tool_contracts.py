"""Tool-contract tests: every tool's input schema, output schema, and at least one error path.
Also reconciles tool names in the committed tool log with the tools the code registers (AC-07)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server import store
from mcp_server.schemas import CoverageOut, PatientRecordOut, SlotsOut
from src.tools.logging_middleware import ToolExecutor, parse_tool_output
from src.tools.mcp_client import MCP_TOOL_NAMES, MCPToolProvider, local_tools
from src.tools.rag_tool import TOOL_NAME, KeywordIndex, RagOutput, build_rag_tool, load_chunks
from tests.conftest import auth

ROOT = Path(__file__).resolve().parents[1]
ALL_TOOLS = MCP_TOOL_NAMES | {TOOL_NAME}


@pytest.fixture(scope="module")
def executor():
    tools = local_tools()
    rag = build_rag_tool(KeywordIndex(load_chunks()))
    tools[rag.name] = rag
    return ToolExecutor(tools)


# ---------------------------------------------------------------- input schemas
@pytest.mark.parametrize("name,required", [
    ("get_patient_record", {"patient_ref", "session_id", "session_token"}),
    ("check_coverage", {"patient_ref", "service_code", "session_id", "session_token"}),
    ("list_available_slots", {"pathway_id"}),
    (TOOL_NAME, {"query"}),
])
def test_input_schema_required_fields(executor, name, required):
    schema = executor.tools[name].args_schema.model_json_schema()
    assert set(schema["required"]) == required


def test_patient_ref_format_is_enforced(executor):
    with pytest.raises(ValidationError):
        executor.tools["get_patient_record"].args_schema(patient_ref="SYN-P-00005", session_id="s", session_token="t")


# ---------------------------------------------------------------- output schemas (happy path)
async def test_get_patient_record_output_is_masked(executor):
    res = await executor.call("get_patient_record", auth(), agent="test")
    assert res.status == "ok"
    rec = PatientRecordOut.model_validate(res.data)
    dumped = json.dumps(res.data)
    for leaked in ("SYN-P-", "SYN-MRN-", "Synthetic Person", "555-0100", "dob"):
        assert leaked not in dumped
    assert rec.plan_id.startswith("PLAN-")


async def test_check_coverage_output_schema_and_rules(executor):
    res = await executor.call("check_coverage", {**auth(patient_id="SYN-P-00004"), "service_code": "PHYSIO"},
                              agent="test")
    out = CoverageOut.model_validate(res.data)
    assert out.eligible is False and [g.rule_id for g in out.gaps] == ["COV-R3"]
    assert {"COV-R1", "COV-R2", "COV-R3"} <= set(out.rules_applied)


async def test_list_available_slots_output_schema(executor):
    res = await executor.call("list_available_slots", {"pathway_id": "CP-MSK-002", "urgency": "routine"}, agent="test")
    out = SlotsOut.model_validate(res.data)
    assert len(out.slots) == 4 and {s.period for s in out.slots} == {"morning", "afternoon"}


async def test_rag_output_schema_and_citations(executor):
    res = await executor.call(TOOL_NAME, {"query": "physiotherapy back pain routing", "pathway_hint": "MSK"},
                              agent="test")
    out = RagOutput.model_validate(res.data)
    assert out.chunks and out.chunks[0].policy_id == "CP-MSK-002"
    assert all("#§" in c.chunk_id for c in out.chunks)


# ---------------------------------------------------------------- error paths
async def test_error_access_denied_for_other_patient(executor):
    a = auth(patient_id="SYN-P-00005")
    other = auth(patient_id="SYN-P-00003")["patient_ref"]
    res = await executor.call("get_patient_record", {**a, "patient_ref": other}, agent="test")
    assert res.status == "denied" and res.data["error"] == "ACCESS_DENIED"


async def test_error_unknown_service_code_reports_cov_r6(executor):
    res = await executor.call("check_coverage", {**auth(), "service_code": "ASTROLOGY"}, agent="test")
    assert res.status == "ok" and res.data["gaps"][0]["rule_id"] == "COV-R6"


async def test_error_invalid_pathway(executor):
    res = await executor.call("list_available_slots", {"pathway_id": "NOPE"}, agent="test")
    assert res.status == "error" and res.data["error"] == "INVALID_INPUT"


async def test_error_rag_rejects_invalid_input(executor):
    res = await executor.call(TOOL_NAME, {"query": "x", "k": 99}, agent="test")
    assert res.status == "error" and "validation" in res.error.lower()


async def test_error_unknown_tool(executor):
    res = await executor.call("delete_all_records", {}, agent="test")
    assert res.status == "error"


def test_parse_tool_output_handles_mcp_content_blocks():
    assert parse_tool_output([{"type": "text", "text": '{"a": 1}'}]) == {"a": 1}
    assert parse_tool_output('{"a": 2}') == {"a": 2}
    assert parse_tool_output(("{\"a\": 3}", None)) == {"a": 3}


# ---------------------------------------------------------------- log format + reconciliation
async def test_tool_log_record_format(executor):
    from src.tools.logging_middleware import TOOL_LOG
    await executor.call("list_available_slots", {"pathway_id": "PEDS"}, agent="test")
    rec = json.loads(TOOL_LOG.read_text(encoding="utf-8").splitlines()[-1])
    assert {"timestamp", "agent", "tool_name", "args", "result", "latency_ms", "status"} <= set(rec)


def test_committed_tool_log_names_reconcile_with_code():
    log = ROOT / "logs" / "tool_calls.jsonl"
    if not log.exists() or not log.read_text(encoding="utf-8").strip():
        pytest.skip("no committed tool log yet - run scripts/regenerate_all.py")
    names = {json.loads(l)["tool_name"] for l in log.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert names <= ALL_TOOLS, f"unknown tool names in log: {names - ALL_TOOLS}"


# ---------------------------------------------------------------- real MCP server over stdio
@pytest.mark.integration
async def test_mcp_server_contract_over_stdio():
    async with MCPToolProvider() as p:
        assert set(p.tools) == MCP_TOOL_NAMES
        schema = p.tools["check_coverage"].args_schema
        schema = schema if isinstance(schema, dict) else schema.model_json_schema()
        assert {"patient_ref", "service_code", "session_id", "session_token"} <= set(schema["required"])
        ex = ToolExecutor(p.tools, mcp_tool_names=MCP_TOOL_NAMES)
        ok = await ex.call("check_coverage", {**auth(), "service_code": "PHYSIO"}, agent="test")
        assert ok.status == "ok" and CoverageOut.model_validate(ok.data).eligible
        denied = await ex.call("get_patient_record", {**auth(), "session_token": "forged"}, agent="test")
        assert denied.status == "denied"
        rules = await p.read_resource("policy://intake/rules")
        assert "INTAKE-001" in rules and "§3" in rules
        assert store.expected_token("S-T", auth()["patient_ref"]) == auth()["session_token"]
