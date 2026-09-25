"""Per-request context shared by tracing, tool log and audit trail.

run_id == the OpenTelemetry trace id of the request's root span, so every log line can be joined to
its Phoenix trace (`context.trace_id`) and span (`context.span_id`).
"""
from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry import trace

run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
patient_ref_var: ContextVar[str | None] = ContextVar("patient_ref", default=None)

_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def current_span_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.span_id, "016x") if ctx and ctx.is_valid else None


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx and ctx.is_valid else None


def jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock, path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
