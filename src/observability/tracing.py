"""Arize Phoenix + OpenInference tracing, wired into the run path (called from src/service.py and the CLI).

Span pipeline:
  LangChainInstrumentor (openinference)  ->  TracerProvider
      -> MaskingExporter -> OTLP HTTP -> Phoenix (in-process `px.launch_app()` or PHOENIX_COLLECTOR_ENDPOINT)
      -> MaskingExporter -> JsonlSpanExporter -> traces/otel_spans_live.jsonl (offline mirror / fallback)

Every exported span passes through MaskingExporter, which PHI-masks attribute values (defence in depth; text is
already masked at ingress). Custom spans carry `span.category` = thinking | acting | tool.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor, SimpleSpanProcessor, SpanExporter,
                                            SpanExportResult)

from src.config import TRACES, get_settings
from src.guardrails.phi import mask_identifiers

log = logging.getLogger(__name__)

LIVE_SPANS = TRACES / "otel_spans_live.jsonl"
TRACER_NAME = "patient-intake-copilot"

_state: dict = {"provider": None, "session": None, "endpoint": None, "started_at": None}


def _ns_to_iso(ns: int | None) -> str | None:
    if not ns:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def _mask_value(v):
    if isinstance(v, str):
        return mask_identifiers(v)
    if isinstance(v, (list, tuple)):
        return type(v)(_mask_value(x) for x in v)
    return v


class MaskingExporter(SpanExporter):
    """Wraps an exporter and PHI-masks every string attribute before export."""

    def __init__(self, inner: SpanExporter):
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        masked = []
        for s in spans:
            attrs = {k: _mask_value(v) for k, v in (s.attributes or {}).items()}
            masked.append(ReadableSpan(
                name=s.name, context=s.context, parent=s.parent, resource=s.resource, attributes=attrs,
                events=s.events, links=s.links, kind=s.kind, status=s.status, start_time=s.start_time,
                end_time=s.end_time, instrumentation_scope=s.instrumentation_scope))
        return self.inner.export(masked)

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return getattr(self.inner, "force_flush", lambda *_: True)(timeout_millis)


class JsonlSpanExporter(SpanExporter):
    """Writes spans in the same column vocabulary as Phoenix `get_spans_dataframe()`."""

    def __init__(self, path: Path = LIVE_SPANS):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            for s in spans:
                attrs = dict(s.attributes or {})
                f.write(json.dumps({
                    "name": s.name,
                    "span_kind": attrs.get("openinference.span.kind", "UNKNOWN"),
                    "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                    "start_time": _ns_to_iso(s.start_time),
                    "end_time": _ns_to_iso(s.end_time),
                    "status_code": s.status.status_code.name if s.status else "UNSET",
                    "status_message": s.status.description if s.status else None,
                    "context.trace_id": format(s.context.trace_id, "032x"),
                    "context.span_id": format(s.context.span_id, "016x"),
                    "attributes": attrs,
                }, default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def init_tracing(launch_ui: bool = True, project: str | None = None) -> TracerProvider:
    """Initialise tracing once per process. Returns the TracerProvider (idempotent)."""
    if _state["provider"] is not None:
        return _state["provider"]
    settings = get_settings()
    project = project or settings.phoenix_project
    _state["started_at"] = datetime.now(timezone.utc)  # export only this process's spans (Cloud keeps history)

    try:
        from openinference.semconv.resource import ResourceAttributes
        resource = Resource.create({ResourceAttributes.PROJECT_NAME: project, "service.name": TRACER_NAME})
    except Exception:
        resource = Resource.create({"openinference.project.name": project, "service.name": TRACER_NAME})
    provider = TracerProvider(resource=resource)

    endpoint = settings.phoenix_endpoint
    if not endpoint and launch_ui:
        try:
            import phoenix as px
            session = px.launch_app()
            _state["session"] = session
            endpoint = session.url.rstrip("/")
            log.info("Phoenix UI running at %s", endpoint)
        except Exception as e:
            log.warning("Could not launch Phoenix in-process (%s); spans go to JSONL mirror only.", e)
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(MaskingExporter(OTLPSpanExporter(
            endpoint=f"{endpoint.rstrip('/')}/v1/traces", headers=auth_headers(), timeout=30))))
        _state["endpoint"] = endpoint
        log.info("Exporting spans to Phoenix at %s (auth: %s)", endpoint, "api key" if auth_headers() else "none")
    provider.add_span_processor(SimpleSpanProcessor(MaskingExporter(JsonlSpanExporter())))
    trace.set_tracer_provider(provider)

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception as e:
        log.warning("openinference LangChain instrumentation unavailable: %s", e)

    _state["provider"] = provider
    return provider


def auth_headers() -> dict:
    """Phoenix Cloud / secured Phoenix: bearer token (current) + api_key header (legacy spaces)."""
    key = get_settings().phoenix_api_key
    return {"authorization": f"Bearer {key}", "api_key": key} if key else {}


def fetch_spans_rest(endpoint: str, project: str, limit: int = 100000, since: datetime | None = None) -> list[dict]:
    """GET {endpoint}/v1/projects/{project}/spans with cursor pagination (no `phoenix` package import needed).
    Returns records in the same flat vocabulary as JsonlSpanExporter / get_spans_dataframe()."""
    import httpx

    out, cursor = [], None
    with httpx.Client(base_url=endpoint.rstrip("/") + "/", headers={**auth_headers(), "accept": "application/json"},
                      timeout=60) as client:
        while len(out) < limit:
            params = {"limit": 100, **({"cursor": cursor} if cursor else {})}
            start = since or _state["started_at"]
            if start is not None:
                params["start_time"] = start.isoformat()
            r = client.get(f"v1/projects/{project}/spans", params=params)
            r.raise_for_status()
            payload = r.json()
            for s in payload.get("data", []):
                attrs = s.get("attributes") or {}
                out.append({
                    "name": s["name"], "span_kind": s.get("span_kind", "UNKNOWN"), "parent_id": s.get("parent_id"),
                    "start_time": s["start_time"], "end_time": s["end_time"], "status_code": s.get("status_code"),
                    "status_message": s.get("status_message"),
                    "context.trace_id": s["context"]["trace_id"], "context.span_id": s["context"]["span_id"],
                    "attributes": attrs})
            cursor = payload.get("next_cursor")
            if not cursor or not payload.get("data"):
                break
    return out


def get_tracer():
    return trace.get_tracer(TRACER_NAME)


def phoenix_url() -> str | None:
    return _state["endpoint"]


def flush() -> None:
    p = _state["provider"]
    if p is not None:
        p.force_flush()


def get_spans_dataframe(project: str | None = None, since: datetime | None = None):
    """Pull spans back from Phoenix: REST API first (works for Phoenix Cloud and without importing `phoenix`),
    then phoenix.client, then legacy px.Client. None if Phoenix is unavailable."""
    import time

    import pandas as pd

    project = project or get_settings().phoenix_project
    endpoint = _state["endpoint"] or get_settings().phoenix_endpoint
    if not endpoint:
        return None
    flush()
    time.sleep(2)  # let the collector finish ingesting the last batch
    try:
        rows = fetch_spans_rest(endpoint, project, since=since)
        if rows:
            return pd.DataFrame(rows)
    except Exception as e:
        log.info("Phoenix REST span export failed (%s); trying phoenix.client", e)
    try:
        from phoenix.client import Client
        return Client(base_url=endpoint, api_key=get_settings().phoenix_api_key).spans.get_spans_dataframe(
            project_identifier=project, limit=100000)
    except Exception as e:
        log.info("phoenix.client export failed (%s); trying legacy px.Client", e)
    try:
        import phoenix as px
        return px.Client(endpoint=endpoint).get_spans_dataframe(project_name=project)
    except Exception as e:
        log.warning("Phoenix span export failed: %s", e)
        return None
