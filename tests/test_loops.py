"""Loop / cascade guards: max-steps and recursion limits stop runaway loops; tool failures escalate, never spin."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import StructuredTool

from src.config import get_settings
from src.tools.logging_middleware import ToolExecutor
from src.tools.mcp_client import local_tools
from src.tools.rag_tool import RagInput


def always(worker):
    return lambda state: worker


async def test_max_worker_calls_forces_escalation(make_copilot):
    limit = get_settings().limits["max_worker_calls"]
    async with make_copilot(router=always("intake_classifier")) as cp:
        final = await cp.handle("L-1", "S-loop", "SYN-P-00005", "book physio for my back")
        history = (await cp.graph.aget_state({"configurable": {"thread_id": "S-loop"}})).values["route_history"]
    assert final.action == "escalate"
    assert history[-1] == "human_escalation"
    assert history[:-1] == ["intake_classifier"] * limit
    assert "loop_guard" in final.staff_note


async def test_recursion_limit_stops_runaway_graph(make_copilot):
    async with make_copilot(router=always("intake_classifier"), max_worker_calls=10_000) as cp:
        final = await cp.handle("L-2", "S-rec", "SYN-P-00005", "book physio for my back", recursion_limit=12)
    assert final.action == "escalate" and final.risk_tier == "high"
    assert "recursion_limit" in final.guard_notes


async def test_agentic_rag_stops_after_max_rounds(make_copilot):
    calls = []

    async def useless_search(query: str, pathway_hint: str | None = None, k: int = 4) -> dict:
        calls.append(query)
        return {"chunks": [{"chunk_id": "ZZ#§1", "policy_id": "ZZ-000", "section": "§1", "title": "irrelevant",
                            "pathway": "NONE", "text": "nothing useful", "score": 0.1}], "backend": "fake"}

    tools = {**local_tools(), "search_care_policy": StructuredTool.from_function(
        coroutine=useless_search, name="search_care_policy", args_schema=RagInput, description="fake")}
    async with make_copilot(tools_override=tools) as cp:
        cp.runtime.tools.tools["search_care_policy"] = tools["search_care_policy"]
        final = await cp.handle("L-3", "S-rag", "SYN-P-00005", "book physio for my back")
    assert len(calls) == get_settings().limits["rag_max_rounds"]
    assert final.action == "escalate" and final.pathway_id == "NONE"


async def test_tool_failure_cascades_to_escalation_not_retry_loop(make_copilot):
    attempts = []

    def broken_coverage(**kwargs):
        attempts.append(1)
        raise ConnectionError("coverage service down")

    tools = {**local_tools()}
    tools["check_coverage"] = StructuredTool.from_function(func=broken_coverage, name="check_coverage",
                                                           args_schema=tools["check_coverage"].args_schema,
                                                           description="broken")
    async with make_copilot(tools_override=tools) as cp:
        final = await cp.handle("L-4", "S-casc", "SYN-P-00005", "book physio for my back")
    assert len(attempts) == get_settings().limits["tool_retries"] + 1  # bounded retries
    assert final.action == "escalate"
    assert "check_coverage" in final.staff_note


async def test_tool_timeout_is_bounded():
    async def slow(query: str, pathway_hint: str | None = None, k: int = 4) -> dict:
        await asyncio.sleep(5)
        return {}

    tool = StructuredTool.from_function(coroutine=slow, name="search_care_policy", args_schema=RagInput,
                                        description="slow")
    ex = ToolExecutor({"search_care_policy": tool}, timeout_s=0.1, retries=1)
    res = await asyncio.wait_for(ex.call("search_care_policy", {"query": "back pain"}, agent="test"), timeout=3)
    assert res.status == "timeout" and res.attempts == 2


async def test_llm_circuit_breaker_stops_hammering_a_failing_model(monkeypatch):
    """F-05: after N consecutive failures the model is skipped (fallback) instead of stalling the run."""
    from pydantic import BaseModel

    from src.runtime import Runtime

    limits = get_settings().limits
    monkeypatch.setitem(limits, "llm_retries", 0)
    monkeypatch.setitem(limits, "circuit_breaker_failures", 3)
    calls = []

    class Down:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            calls.append(1)
            raise ConnectionError("429 RESOURCE_EXHAUSTED")

    class Out(BaseModel):
        ok: bool

    rt = Runtime(tools=None, memory=None, llm=Down(), llm_light=Down())
    results = [await rt.structured(Out, [], agent="test") for _ in range(6)]
    assert results == [None] * 6
    assert len(calls) == 3  # calls 4-6 short-circuited
    assert await rt.structured(Out, [], light=True, agent="test") is None and len(calls) == 4  # tiers independent


def test_rate_limiter_caps_requests_per_minute():
    """F-06: the per-model limiter never hands out more than `rpm` slots in a 60 s window."""
    from src.ratelimit import SlidingWindowLimiter
    lim = SlidingWindowLimiter(rpm=3)
    waits = [lim._reserve() for _ in range(5)]
    assert waits[:3] == [0.0, 0.0, 0.0]
    assert all(55 < w <= 60 for w in waits[3:])  # 4th and 5th must wait for the window to roll over


def test_retry_delay_is_read_from_provider_429():
    from src.ratelimit import is_rate_limited, retry_delay_seconds
    err = RuntimeError("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'retryDelay': '57s'}]}}")
    assert is_rate_limited(err) and retry_delay_seconds(err) == 58.0
    assert retry_delay_seconds(RuntimeError("503 UNAVAILABLE")) is None
