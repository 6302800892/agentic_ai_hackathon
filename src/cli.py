"""CLI for the Patient Intake & Care-Coordination Copilot.

  python -m src.cli run  --input data/samples/intake_requests.jsonl      # batch run of committed samples
  python -m src.cli chat --patient SYN-P-00004 --session demo1 [--hitl]  # interactive
  python -m src.cli forget --patient SYN-P-00004                          # erase long-term memory (DPDP)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import typer

from src.config import REPORTS, ROOT
from src.observability.tracing import flush, init_tracing, phoenix_url

app = typer.Typer(add_completion=False, help="Patient Intake & Care-Coordination Copilot (Gemini + LangGraph)")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def show(req_id: str, final) -> None:
    if isinstance(final, dict):
        typer.echo(f"[{req_id}] INTERRUPTED for human review: {final['interrupt']}")
        return
    gaps = ",".join(g.rule_id for g in final.coverage_gaps) or "-"
    cites = ",".join(f"{c.policy_id}{c.section}" for c in final.citations) or "-"
    typer.echo(f"[{req_id}] action={final.action:<8} risk={final.risk_tier:<6} clinician={str(final.requires_clinician):<5} "
               f"intent={final.intent or '-':<17} pathway={final.pathway_id or '-':<13} gaps={gaps:<14} cites={cites}")
    typer.echo(f"         -> {final.patient_message}")
    typer.echo(f"            ({final.disclosure})")
    if final.recalled_memories:
        typer.echo(f"         memory: {final.recalled_memories}")


async def run_file(path: Path, transport: str, out: Path | None) -> list[dict]:
    from src.service import Copilot

    rows = []
    async with Copilot(transport=transport) as cp:
        typer.echo(f"mode={cp.mode} tools={cp.transport_used} requests={path}")
        for r in read_jsonl(path):
            final = await cp.handle(r["request_id"], r["session_id"], r["patient_id"], r["text"])
            show(r["request_id"], final)
            rows.append({"request_id": r["request_id"], "session_id": r["session_id"],
                         **(final.model_dump() if not isinstance(final, dict) else final)})
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
    return rows


@app.command()
def run(input: Path = typer.Option(ROOT / "data/samples/intake_requests.jsonl", help="JSONL of requests"),
        transport: str = typer.Option("mcp", help="mcp | local"),
        phoenix: bool = typer.Option(True, help="launch Phoenix in-process and trace the run"),
        out: Path = typer.Option(REPORTS / "demo_outputs.jsonl", help="append final responses here"),
        keep_ui: bool = typer.Option(False, help="keep Phoenix UI open after the run (for screenshots)")):
    """Run every request in a JSONL file through the copilot."""
    init_tracing(launch_ui=phoenix)
    asyncio.run(run_file(input, transport, out))
    flush()
    if keep_ui and phoenix_url():
        typer.echo(f"Phoenix UI: {phoenix_url()}  (press Enter to exit)")
        sys.stdin.readline()


@app.command()
def chat(patient: str = typer.Option(..., help="synthetic patient id, e.g. SYN-P-00004"),
         session: str = typer.Option("chat-1"), hitl: bool = typer.Option(False, help="pause escalations for review"),
         transport: str = typer.Option("mcp"), phoenix: bool = typer.Option(False)):
    """Interactive session for one synthetic patient (type 'exit' to quit)."""
    init_tracing(launch_ui=phoenix)

    async def loop():
        from src.service import Copilot
        async with Copilot(transport=transport, hitl=hitl) as cp:
            typer.echo(f"mode={cp.mode} tools={cp.transport_used}. Type 'exit' to quit.")
            n = 0
            while True:
                text = input("patient> ").strip()
                if text.lower() in ("exit", "quit"):
                    break
                n += 1
                rid = f"{session}-{n}"
                final = await cp.handle(rid, session, patient, text)
                while isinstance(final, dict) and "interrupt" in final:
                    typer.echo(f"REVIEW NEEDED: {final['interrupt']}")
                    note = input("reviewer note> ")
                    final = await cp.resume(rid, patient, session, {"note": note})
                show(rid, final)

    asyncio.run(loop())
    flush()


@app.command()
def forget(patient: str = typer.Option(...)):
    """Erase a patient's long-term memory (right to erasure)."""
    async def go():
        from src.service import Copilot
        async with Copilot(transport="local", use_llm=False) as cp:
            n = await cp.forget(patient)
            typer.echo(f"deleted {n} memory items")
    init_tracing(launch_ui=False)
    asyncio.run(go())


if __name__ == "__main__":
    app()
