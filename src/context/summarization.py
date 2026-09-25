"""Summarization middleware: when the thread history exceeds the budget, older turns are folded into
`state.summary` (LLM-written, fallback: extractive) and removed from `messages` with RemoveMessage."""
from __future__ import annotations

from langchain_core.messages import RemoveMessage

from src.config import get_settings
from src.context.compress import messages_tokens, truncate
from src.context.select import context_for_summary
from src.state import CopilotState


def _extractive(previous: str, messages: list) -> str:
    lines = [f"{m.type}: {truncate(str(m.content), 40)}" for m in messages]
    return truncate(((previous + " | ") if previous else "") + " ; ".join(lines), 200)


async def summarize_if_needed(state: CopilotState, runtime) -> dict:
    limits = get_settings().limits
    messages = state.get("messages", [])
    if messages_tokens(messages) <= limits.get("summary_trigger_tokens", 1200):
        return {}
    keep = limits.get("summary_keep_messages", 4)
    old = messages[:-keep]
    if not old:
        return {}
    summary = None
    if runtime.llm_light is not None:
        prompt = context_for_summary(old)
        if state.get("summary"):
            prompt[-1].content = f"Previous summary: {state['summary']}\n\n" + prompt[-1].content
        summary = await runtime.text(prompt, light=True, agent="summarization")
    summary = summary or _extractive(state.get("summary", ""), old)
    return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old if m.id]}
