"""Runtime dependencies injected into the graph (LLMs, tools, memory). Swappable in tests (fake/no LLM,
in-process tools) without touching graph code.

All model calls go through `structured()` / `text()`: async, per-call timeout, bounded retries, and a `None`
return on failure so the calling node can degrade gracefully instead of crashing the run (NFR-04).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel
from src.audit.audit import audit
from src.config import get_settings
from src.ratelimit import is_rate_limited, limiter_for, retry_delay_seconds
from src.memory.long_term import LongTermMemory
from src.tools.logging_middleware import ToolExecutor

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def make_gemini(light: bool = False):
    """Gemini is the only model provider (Open-Source & Gemini-Only Rule)."""
    s = get_settings()
    if not s.has_llm:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=s.gemini_model_light if light else s.gemini_model, temperature=0,
                                  max_retries=0, timeout=s.limits.get("llm_timeout_s", 30),
                                  google_api_key=s.google_api_key)


@dataclass
class Runtime:
    tools: ToolExecutor
    memory: LongTermMemory
    llm: Any = None
    llm_light: Any = None
    intake_rules: str = ""
    hitl: bool = False
    rag_tool_name: str = "search_care_policy"
    extra: dict = field(default_factory=dict)
    # circuit breaker per model tier (see docs/failure-analysis.md F-05)
    _failures: dict = field(default_factory=lambda: {"main": 0, "light": 0})
    _open_until: dict = field(default_factory=lambda: {"main": 0.0, "light": 0.0})

    async def _with_retries(self, coro_factory, agent: str, tier: str = "main"):
        limits = get_settings().limits
        if time.monotonic() < self._open_until[tier]:
            audit(agent, "degraded", "circuit_open_fallback", reason=f"{tier} model circuit open after repeated failures")
            return None
        s = get_settings()
        limiter = limiter_for(s.gemini_model_light if tier == "light" else s.gemini_model)
        retries = limits.get("llm_retries", 2)
        try:
            for attempt in range(retries + 1):
                await limiter.acquire()  # stay under the per-model RPM quota (F-06)
                try:
                    result = await asyncio.wait_for(coro_factory(), timeout=limits.get("llm_timeout_s", 30))
                    self._failures[tier] = 0
                    return result
                except Exception as e:  # noqa: PERF203 - retry loop
                    if attempt == retries:
                        raise
                    # 429: wait exactly what the provider asks; otherwise exponential backoff 2..20 s (F-04)
                    delay = retry_delay_seconds(e) if is_rate_limited(e) else None
                    await asyncio.sleep(delay or min(20.0, 2.0 * 2 ** attempt))
        except Exception as e:
            self._failures[tier] += 1
            log.warning("LLM call for %s failed after retries: %s", agent, e)
            audit(agent, "degraded", "llm_unavailable_fallback", reason=f"{type(e).__name__}: {str(e)[:200]}")
            if self._failures[tier] >= limits.get("circuit_breaker_failures", 3):
                self._open_until[tier] = time.monotonic() + limits.get("circuit_breaker_cooldown_s", 120)
                self._failures[tier] = 0
                audit("system", "degraded", "circuit_opened", reason=f"{tier} model failed repeatedly; "
                      f"fallback for {limits.get('circuit_breaker_cooldown_s', 120)}s")
            return None

    async def structured(self, schema: Type[T], messages: list, *, light: bool = False, agent: str = "") -> T | None:
        llm = self.llm_light if light else self.llm
        if llm is None:
            return None
        runnable = llm.with_structured_output(schema)
        result = await self._with_retries(lambda: runnable.ainvoke(messages), agent, "light" if light else "main")
        if result is not None and not isinstance(result, schema):
            try:
                result = schema.model_validate(result)
            except Exception:
                return None
        return result

    async def text(self, messages: list, *, light: bool = True, agent: str = "") -> str | None:
        llm = self.llm_light if light else self.llm
        if llm is None:
            return None
        msg = await self._with_retries(lambda: llm.ainvoke(messages), agent, "light" if light else "main")
        if msg is None:
            return None
        content = msg.content
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        return str(content).strip() or None
