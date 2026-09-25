"""MCP client: consumes the custom care-coordination MCP server via langchain-mcp-adapters.

`MCPToolProvider` keeps ONE stdio session open for the lifetime of the copilot (no per-call process spawn),
loads the server's tools as LangChain tools (`load_mcp_tools`) and reads resources (`load_mcp_resources`).

`local_tools()` exposes the same functions in-process with identical names and contracts. It is used by the
offline test-suite and as a degraded fallback if the MCP server cannot be started.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import AsyncExitStack

from langchain_core.tools import BaseTool, StructuredTool

from mcp_server import store
from mcp_server.schemas import CheckCoverageInput, GetPatientRecordInput, ListSlotsInput
from src.config import ROOT
from src.observability.run_context import jsonl_append, now_iso
from src.tools.logging_middleware import MCP_TRANSCRIPT

log = logging.getLogger(__name__)

SERVER_NAME = "care"
MCP_TOOL_NAMES = {"get_patient_record", "check_coverage", "list_available_slots"}
MCP_RESOURCES = ["policy://intake/rules", "patients://schema"]


def server_config() -> dict:
    return {SERVER_NAME: {
        "command": sys.executable,
        "args": ["-m", "mcp_server.server"],
        "transport": "stdio",
        "cwd": str(ROOT),
        "env": {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
    }}


class MCPToolProvider:
    def __init__(self):
        self._stack = AsyncExitStack()
        self.session = None
        self.tools: dict[str, BaseTool] = {}

    async def __aenter__(self) -> "MCPToolProvider":
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        t0 = time.perf_counter()
        self.client = MultiServerMCPClient(server_config())
        self.session = await self._stack.enter_async_context(self.client.session(SERVER_NAME))
        tools = await load_mcp_tools(self.session)
        self.tools = {t.name: t for t in tools}
        jsonl_append(MCP_TRANSCRIPT, {"ts": now_iso(), "side": "client", "direction": "response",
                                      "method": "tools/list", "tool_name": SERVER_NAME, "args": {},
                                      "result": sorted(self.tools), "status": "ok",
                                      "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
        missing = MCP_TOOL_NAMES - set(self.tools)
        if missing:
            raise RuntimeError(f"MCP server is missing tools: {missing}")
        return self

    async def read_resource(self, uri: str) -> str:
        from langchain_mcp_adapters.resources import load_mcp_resources

        t0 = time.perf_counter()
        blobs = await load_mcp_resources(self.session, uris=[uri])
        text = blobs[0].as_string() if blobs else ""
        jsonl_append(MCP_TRANSCRIPT, {"ts": now_iso(), "side": "client", "direction": "response",
                                      "method": "resources/read", "tool_name": uri, "args": {},
                                      "result": {"chars": len(text)}, "status": "ok",
                                      "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
        return text

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()


def local_tools() -> dict[str, BaseTool]:
    """In-process equivalents of the MCP tools (same names + input schemas)."""
    return {
        "get_patient_record": StructuredTool.from_function(
            func=store.get_patient_record, name="get_patient_record", args_schema=GetPatientRecordInput,
            description="Masked coverage-relevant record for the session's patient."),
        "check_coverage": StructuredTool.from_function(
            func=store.check_coverage, name="check_coverage", args_schema=CheckCoverageInput,
            description="Eligibility check using COV-R1..R6."),
        "list_available_slots": StructuredTool.from_function(
            func=store.list_available_slots, name="list_available_slots", args_schema=ListSlotsInput,
            description="Synthetic appointment slots for a pathway."),
    }


def local_resource(uri: str) -> str:
    if uri == "policy://intake/rules":
        return (ROOT / "data" / "policy_corpus" / "INTAKE-001.md").read_text(encoding="utf-8")
    raise KeyError(uri)
