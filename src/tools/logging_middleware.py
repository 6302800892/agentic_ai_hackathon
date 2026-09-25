"""Tool-invocation logging middleware (AC-07) + resilient executor (NFR-04).

Every tool call in the system goes through `ToolExecutor.call()` (or the `@logged_tool` decorator), which:
  * opens a `tool.<name>` span (span.category=tool) so the call is visible in Phoenix,
  * enforces a timeout and retries transient failures (timeouts / connection errors) with backoff,
  * appends one PHI-masked record to logs/tool_calls.jsonl:
      {timestamp, run_id, span_id, agent, tool_name, args, result, latency_ms, status, attempts}
  * writes a client-side MCP transcript line for MCP tools, and an audit record for the call.
"""
from __future__ import annotations

import asyncio
import functools
import json
import time
from typing import Any, Awaitable, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.audit.audit import audit
from src.config import LOGS, get_settings
from src.guardrails.phi import mask_obj
from src.observability.run_context import current_span_id, jsonl_append, now_iso, run_id_var
from src.observability.tracing import get_tracer

TOOL_LOG = LOGS / "tool_calls.jsonl"
MCP_TRANSCRIPT = LOGS / "mcp_transcript.jsonl"
TRANSIENT = (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)


class ToolResult(BaseModel):
    tool_name: str
    status: str               # ok | error | timeout | denied
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def parse_tool_output(out: Any) -> Any:
    """Normalise MCP / LangChain tool outputs (str JSON, content blocks, tuples, dicts) to Python data."""
    if isinstance(out, tuple) and len(out) == 2:
        out = out[0]
    if isinstance(out, BaseModel):
        return out.model_dump()
    if isinstance(out, list):
        texts = [b.get("text") if isinstance(b, dict) else getattr(b, "text", None) for b in out]
        texts = [t for t in texts if t]
        if texts and len(texts) == len(out):
            out = "".join(texts)
        else:
            return out
    if isinstance(out, str):
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"text": out}
    return out


def _status_of(data: Any) -> str:
    if isinstance(data, dict) and "error" in data:
        return "denied" if data.get("error") == "ACCESS_DENIED" else "error"
    return "ok"


def write_tool_log(*, agent: str, tool_name: str, args: dict, result: Any, latency_ms: float, status: str,
                   attempts: int = 1, span_id: str | None = None) -> dict:
    rec = {"timestamp": now_iso(), "run_id": run_id_var.get(), "span_id": span_id or current_span_id(),
           "agent": agent, "tool_name": tool_name,
           "args": {k: ("[TOKEN]" if k == "session_token" else v) for k, v in (args or {}).items()},
           "result": result, "latency_ms": round(latency_ms, 2), "status": status, "attempts": attempts}
    rec = mask_obj(rec)
    jsonl_append(TOOL_LOG, rec)
    return rec


def _summarise(data: Any) -> Any:
    """Keep logs readable: long RAG chunk texts are truncated in the log (not in the agent)."""
    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        return {**data, "chunks": [{**c, "text": (c.get("text", "")[:160] + "...")} for c in data["chunks"]]}
    return data


class ToolExecutor:
    def __init__(self, tools: dict[str, BaseTool], mcp_tool_names: set[str] | None = None,
                 timeout_s: float | None = None, retries: int | None = None):
        limits = get_settings().limits
        self.tools = tools
        self.mcp_tool_names = mcp_tool_names or set()
        self.timeout_s = timeout_s if timeout_s is not None else limits.get("tool_timeout_s", 10)
        self.retries = retries if retries is not None else limits.get("tool_retries", 2)

    @property
    def names(self) -> set[str]:
        return set(self.tools)

    async def call(self, name: str, args: dict, agent: str) -> ToolResult:
        tracer = get_tracer()
        masked_args = mask_obj({k: ("[TOKEN]" if k == "session_token" else v) for k, v in args.items()})
        with tracer.start_as_current_span(f"tool.{name}", attributes={
            "span.category": "tool", "openinference.span.kind": "TOOL", "tool.name": name, "agent": agent,
            "input.value": json.dumps(masked_args), "run_id": run_id_var.get() or ""}) as span:
            span_id = current_span_id()
            t0 = time.perf_counter()
            attempts, status, data, error = 0, "error", None, None
            if name not in self.tools:
                status, error = "error", f"unknown tool {name}"
            else:
                if name in self.mcp_tool_names:
                    self._transcript("request", name, args, None, 0.0, "sent")
                try:
                    async for attempt in AsyncRetrying(
                            stop=stop_after_attempt(self.retries + 1),
                            wait=wait_exponential(multiplier=0.2, max=2),
                            retry=retry_if_exception_type(TRANSIENT), reraise=True):
                        with attempt:
                            attempts += 1
                            raw = await asyncio.wait_for(self.tools[name].ainvoke(args), timeout=self.timeout_s)
                    data = parse_tool_output(raw)
                    status = _status_of(data)
                    if status != "ok":
                        error = str(data.get("detail") or data.get("error"))
                except TRANSIENT as e:
                    status = "timeout" if isinstance(e, (asyncio.TimeoutError, TimeoutError)) else "error"
                    error = f"{type(e).__name__}: {e}"
                except Exception as e:  # contract violations, validation errors, server errors
                    status, error = "error", f"{type(e).__name__}: {e}"
            latency_ms = (time.perf_counter() - t0) * 1000
            logged = _summarise(data) if data is not None else {"error": error}
            span.set_attribute("output.value", json.dumps(mask_obj(logged), default=str)[:4000])
            span.set_attribute("tool.status", status)
            if status != "ok":
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, error or status))
            write_tool_log(agent=agent, tool_name=name, args=args, result=logged, latency_ms=latency_ms,
                           status=status, attempts=max(attempts, 1), span_id=span_id)
            if name in self.mcp_tool_names:
                self._transcript("response", name, args, logged, latency_ms, status)
            audit(agent, "access_denied" if status == "denied" else "tool_call", status, tool=name,
                  reason=error or "", details={"latency_ms": round(latency_ms, 1)})
        return ToolResult(tool_name=name, status=status, data=data, error=error, latency_ms=latency_ms,
                          attempts=max(attempts, 1))

    @staticmethod
    def _transcript(direction: str, name: str, args: dict, result: Any, latency_ms: float, status: str) -> None:
        rec = {"ts": now_iso(), "side": "client", "direction": direction, "method": "tools/call", "tool_name": name,
               "run_id": run_id_var.get(), "span_id": current_span_id(),
               "args": {k: ("[TOKEN]" if k == "session_token" else v) for k, v in args.items()},
               "result": result, "latency_ms": round(latency_ms, 2), "status": status}
        jsonl_append(MCP_TRANSCRIPT, mask_obj(rec))


def logged_tool(agent: str, name: str | None = None) -> Callable:
    """Decorator for plain async functions used as tools (same log format as ToolExecutor)."""
    def deco(fn: Callable[..., Awaitable[Any]]):
        tool_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(**kwargs):
            t0 = time.perf_counter()
            try:
                result = await fn(**kwargs)
                status = _status_of(result)
                return result
            except Exception as e:
                result, status = {"error": str(e)}, "error"
                raise
            finally:
                write_tool_log(agent=agent, tool_name=tool_name, args=kwargs, result=_summarise(result),
                               latency_ms=(time.perf_counter() - t0) * 1000, status=status)
        return wrapper
    return deco
