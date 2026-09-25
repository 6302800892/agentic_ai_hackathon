"""Run the committed sample inputs (incl. a multi-turn session and cross-session return visits).

Writes reports/demo_outputs.jsonl; tool log, audit trail, MCP transcript and spans are written by the
middleware as a side effect. Used by scripts/regenerate_all.py (or standalone).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli import read_jsonl, show  # noqa: E402
from src.config import REPORTS  # noqa: E402

SAMPLES = [ROOT / "data/samples/intake_requests.jsonl", ROOT / "data/samples/session_return_visit.jsonl"]
OUT = REPORTS / "demo_outputs.jsonl"


async def run_demo(cp, verbose: bool = True) -> list[dict]:
    rows = []
    for path in SAMPLES:
        for r in read_jsonl(path):
            final = await cp.handle(r["request_id"], r["session_id"], r["patient_id"], r["text"])
            if verbose:
                show(r["request_id"], final)
            rows.append({"request_id": r["request_id"], "session_id": r["session_id"], "source": path.name,
                         **final.model_dump()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x, default=str) + "\n" for x in rows), encoding="utf-8")
    return rows


async def _main():
    from src.observability.tracing import flush, init_tracing
    from src.service import Copilot
    init_tracing(launch_ui=False)
    async with Copilot() as cp:
        await run_demo(cp)
    flush()


if __name__ == "__main__":
    asyncio.run(_main())
