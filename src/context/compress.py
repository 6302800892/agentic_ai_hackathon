"""Compress: keep prompts inside a token budget (history trimming + per-chunk truncation)."""
from __future__ import annotations

from langchain_core.messages import AnyMessage


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def messages_tokens(messages: list[AnyMessage]) -> int:
    return sum(estimate_tokens(str(m.content)) for m in messages)


def trim_history(messages: list[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """Keep the most recent messages that fit the budget (oldest dropped first)."""
    kept, total = [], 0
    for m in reversed(messages):
        t = estimate_tokens(str(m.content))
        if kept and total + t > max_tokens:
            break
        kept.append(m)
        total += t
    return list(reversed(kept))


def truncate(text: str, max_tokens: int) -> str:
    limit = max_tokens * 4
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."
