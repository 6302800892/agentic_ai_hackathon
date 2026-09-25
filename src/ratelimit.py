"""Client-side rate limiting for Gemini calls (cost & quota governance, docs/failure-analysis.md F-06).

The free tier allows 15 requests per minute per model. Every Gemini call in the system (agents via
src/runtime.py, LangMem extraction in src/memory/long_term.py and the DeepEval judge in scripts/run_eval.py)
acquires a slot from the same per-model limiter, so a run stays under `gemini_rpm` (config/limits.yaml)
instead of burning retries on 429s. On a 429 the provider's own `retryDelay` is honoured.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from collections import deque

from src.config import get_settings

_RETRY_RE = re.compile(r"retryDelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s")


class SlidingWindowLimiter:
    """At most `rpm` acquisitions in any 60 s window. Thread-safe (sync judge) and async-friendly."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def _reserve(self) -> float:
        """Reserve a slot; return how long the caller must wait before using it."""
        with self._lock:
            now = time.monotonic()
            while self._stamps and now - self._stamps[0] >= 60:
                self._stamps.popleft()
            if len(self._stamps) < self.rpm:
                self._stamps.append(now)
                return 0.0
            start = self._stamps[-self.rpm] + 60  # when the oldest slot in the window frees up
            self._stamps.append(start)
            return max(0.0, start - now)

    async def acquire(self) -> None:
        delay = self._reserve()
        if delay:
            await asyncio.sleep(delay)

    def acquire_sync(self) -> None:
        delay = self._reserve()
        if delay:
            time.sleep(delay)


_limiters: dict[str, SlidingWindowLimiter] = {}
_guard = threading.Lock()


def limiter_for(model: str) -> SlidingWindowLimiter:
    with _guard:
        if model not in _limiters:
            _limiters[model] = SlidingWindowLimiter(get_settings().limits.get("gemini_rpm", 12))
        return _limiters[model]


def retry_delay_seconds(exc: BaseException) -> float | None:
    """Parse the provider's suggested retry delay from a 429 error, if present."""
    m = _RETRY_RE.search(str(exc))
    return float(m.group(1)) + 1.0 if m else None


def is_rate_limited(exc: BaseException) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or " 429" in text or "429 " in text
