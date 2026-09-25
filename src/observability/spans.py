"""Normalise Phoenix span exports (get_spans_dataframe parquet) or the OTel JSONL mirror into one table.

Columns: trace_id, span_id, parent_id, name, span_kind, category, agent, start_time, end_time, latency_ms,
status, prompt_tokens, completion_tokens, total_tokens, model, cost_usd

category: thinking (LLM spans) | acting (graph-node / agent spans + root request) | tool (tool.* spans)
          | tool_inner (openinference TOOL span nested inside tool.*, excluded to avoid double counting) | other
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import get_settings

AGENT_NODES = {"input_guard", "refusal", "context_prep", "supervisor", "intake_classifier", "coverage_checker",
               "care_pathway", "coordinator", "clarify", "human_escalation", "memory_write", "output_guard"}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def read_raw(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def _attr_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return attributes as flat columns (without the 'attributes.' prefix)."""
    if "attributes" in df.columns:
        flat = [_flatten(a if isinstance(a, dict) else json.loads(a) if isinstance(a, str) else {})
                for a in df["attributes"]]
        return pd.DataFrame(flat, index=df.index)
    cols = {c: c[len("attributes."):] for c in df.columns if c.startswith("attributes.")}
    return df[list(cols)].rename(columns=cols) if cols else pd.DataFrame(index=df.index)


def _price(model: str | None) -> tuple[float, float]:
    pricing = get_settings().pricing
    name = model.replace("models/", "") if isinstance(model, str) else ""  # NaN for non-LLM spans
    for key, p in (pricing.get("models") or {}).items():
        if name.startswith(key):
            return p["input_per_1m"], p["output_per_1m"]
    d = pricing.get("default", {"input_per_1m": 0.0, "output_per_1m": 0.0})
    return d["input_per_1m"], d["output_per_1m"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop="context.span_id" in df.columns)
    attrs = _attr_frame(df)

    def col(name, default=None):
        return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)

    def acol(name, default=None):
        return attrs[name] if name in attrs.columns else pd.Series([default] * len(df), index=df.index)

    out = pd.DataFrame({
        "trace_id": col("context.trace_id"),
        "span_id": col("context.span_id") if "context.span_id" in df.columns else col("span_id"),
        "parent_id": col("parent_id"),
        "name": col("name"),
        "span_kind": col("span_kind").fillna(acol("openinference.span.kind")),
        "custom_category": acol("span.category"),
        "start_time": pd.to_datetime(col("start_time"), utc=True, format="mixed"),
        "end_time": pd.to_datetime(col("end_time"), utc=True, format="mixed"),
        "status": col("status_code"),
        "prompt_tokens": pd.to_numeric(acol("llm.token_count.prompt"), errors="coerce").fillna(0),
        "completion_tokens": pd.to_numeric(acol("llm.token_count.completion"), errors="coerce").fillna(0),
        "model": acol("llm.model_name"),
        "agent_attr": acol("agent"),
    })
    out["latency_ms"] = (out["end_time"] - out["start_time"]).dt.total_seconds() * 1000
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]

    def category(r) -> str:
        name = str(r["name"])
        if name.startswith("tool."):
            return "tool"
        if r["span_kind"] == "LLM":
            return "thinking"
        if r["span_kind"] in ("TOOL", "RETRIEVER"):
            return "tool_inner"
        if name in AGENT_NODES or name == "copilot.request":
            return "acting"
        return "other"
    out["category"] = out.apply(category, axis=1)

    # agent attribution: nearest ancestor that is a graph node
    names = dict(zip(out["span_id"], out["name"]))
    parents = dict(zip(out["span_id"], out["parent_id"]))

    def agent_of(sid, own_attr):
        if isinstance(own_attr, str) and own_attr:
            return own_attr
        seen = 0
        while sid is not None and seen < 50:
            if names.get(sid) in AGENT_NODES:
                return names[sid]
            sid = parents.get(sid)
            seen += 1
        return None
    out["agent"] = [agent_of(s, a) for s, a in zip(out["span_id"], out["agent_attr"])]

    prices = out["model"].map(_price)
    out["cost_usd"] = [(p * pi + c * po) / 1e6 for p, c, (pi, po) in
                       zip(out["prompt_tokens"], out["completion_tokens"], prices)]
    return out.drop(columns=["agent_attr", "custom_category"])


def load_spans(path: Path) -> pd.DataFrame:
    return normalize(read_raw(path))
