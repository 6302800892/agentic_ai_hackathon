"""Bonus: async FastAPI streaming endpoint (Server-Sent Events).

    uvicorn src.api.app:app --port 8000
    curl -N -X POST localhost:8000/intake -H "content-type: application/json" \
         -d '{"session_id":"api-1","patient_id":"SYN-P-00005","text":"Can I book physio for my back?"}'

Streams one `node` event per graph node as it completes, then a `final` event with the guarded FinalResponse.
Only masked / structured data is streamed (the same ingress guard + output guard as the CLI).
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.guardrails.input_guard import ingress
from src.guardrails.phi import patient_ref, session_token
from src.observability.run_context import current_trace_id, patient_ref_var, request_id_var, run_id_var
from src.observability.tracing import get_tracer, init_tracing
from src.service import Copilot
from src.state import fresh_turn_state

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing(launch_ui=False)
    async with Copilot() as cp:
        STATE["cp"] = cp
        yield


app = FastAPI(title="Patient Intake & Care-Coordination Copilot", lifespan=lifespan)


class IntakeRequest(BaseModel):
    session_id: str
    patient_id: str = Field(pattern=r"^SYN-P-\d{5}$")
    text: str = Field(min_length=1, max_length=2000)


def _summary(node: str, update: dict) -> dict:
    """Structured, PHI-free view of a node's output for streaming."""
    out = {}
    for k, v in (update or {}).items():
        if k in ("messages", "session_token", "input_text"):
            continue
        out[k] = v.model_dump() if hasattr(v, "model_dump") else v
    return {"node": node, "update": out}


@app.get("/health")
async def health():
    cp = STATE.get("cp")
    return {"ok": cp is not None, "mode": cp.mode if cp else None, "tools": cp.transport_used if cp else None}


@app.post("/intake")
async def intake(req: IntakeRequest):
    cp: Copilot = STATE["cp"]
    request_id = f"API-{uuid.uuid4().hex[:8]}"
    ref = patient_ref(req.patient_id)
    ing = ingress(req.text, req.patient_id)
    inp = fresh_turn_state(request_id=request_id, session_id=req.session_id, thread_id=req.session_id,
                           patient_ref=ref, session_token=session_token(req.session_id, ref),
                           input_text=ing.masked_text, ingress_flags=ing.flags)
    config = {"configurable": {"thread_id": req.session_id}, "recursion_limit": 25}

    async def events():
        with get_tracer().start_as_current_span("copilot.request", attributes={
                "openinference.span.kind": "CHAIN", "request_id": request_id, "input.value": ing.masked_text}):
            tokens = [run_id_var.set(current_trace_id()), request_id_var.set(request_id), patient_ref_var.set(ref)]
            try:
                async for chunk in cp.graph.astream(inp, config, stream_mode="updates"):
                    for node, update in chunk.items():
                        yield {"event": "node", "data": json.dumps(_summary(node, update), default=str)}
                        if node == "output_guard":
                            yield {"event": "final",
                                   "data": json.dumps(update["final_response"].model_dump(), default=str)}
            finally:
                for var, tok in zip((run_id_var, request_id_var, patient_ref_var), tokens):
                    var.reset(tok)

    return EventSourceResponse(events())
