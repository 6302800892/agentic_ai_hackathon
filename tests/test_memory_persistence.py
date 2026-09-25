"""Tiered memory: short-term (thread checkpoints) and long-term (cross-session store) survive a full restart.

Each Copilot instance below opens its own SQLite connections; the second instance is created only after the
first is closed, so recall can only come from what was persisted to disk. A readable trace of the test is
written to logs/memory_test.log (committed evidence).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "memory_test.log"
_lines: list[str] = []


def log(msg: str) -> None:
    _lines.append(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}")


def flush_log() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(_lines) + "\n", encoding="utf-8")


async def test_cross_session_and_cross_restart_memory(make_copilot, tmp_path):
    log("=== tests/test_memory_persistence.py :: cross-session recall (fresh process-level instances) ===")
    log(f"stores: checkpoints={tmp_path.name}/checkpoints.sqlite memory={tmp_path.name}/memory.sqlite")

    # ---- instance 1: session A for patient P11 (states a preference) + turn 1 of a thread for P12
    async with make_copilot() as cp1:
        a = await cp1.handle("M-1", "S-A", "SYN-P-00011",
                             "My 6 year old daughter has an ear ache, can we get an appointment? Mornings only please.")
        log(f"[instance-1] session=S-A patient_ref={_ref('SYN-P-00011')} action={a.action} "
            f"slot={a.proposed_slot and a.proposed_slot['start']} ({a.proposed_slot and a.proposed_slot['period']})")
        t1 = await cp1.handle("M-2", "S-THREAD", "SYN-P-00012", "I twisted my ankle playing football on Saturday.")
        log(f"[instance-1] thread=S-THREAD turn-1 pathway={t1.pathway_id}")
        stored = await cp1.runtime.memory.store.asearch(("patients", _ref("SYN-P-00011"), "memories"), limit=50)
        log(f"[instance-1] long-term items stored for P11: {[i.value['text'] for i in stored]}")
    log("[instance-1] closed (all SQLite connections released)")

    # ---- instance 2: brand-new objects on the same files
    async with make_copilot() as cp2:
        b = await cp2.handle("M-3", "S-B", "SYN-P-00011", "Hi again, can you book the follow-up visit for my daughter?")
        log(f"[instance-2] NEW session=S-B recalled={b.recalled_memories}")
        log(f"[instance-2] slot={b.proposed_slot['start']} period={b.proposed_slot['period']} msg={b.patient_message!r}")
        assert any("morning" in m.lower() for m in b.recalled_memories), "preference not recalled across sessions"
        assert any("daughter" in m.lower() for m in b.recalled_memories)
        assert b.proposed_slot["period"] == "morning", "recalled preference not applied"

        other = await cp2.handle("M-4", "S-C", "SYN-P-00013", "Can I book a check-up?")
        log(f"[instance-2] other patient session=S-C recalled={other.recalled_memories}")
        assert not any("daughter" in m.lower() or "morning" in m.lower() for m in other.recalled_memories), \
            "memory leaked across patients"

        t2 = await cp2.handle("M-5", "S-THREAD", "SYN-P-00012", "Can I get an appointment for it? I prefer afternoons.")
        log(f"[instance-2] SAME thread=S-THREAD turn-2 pathway={t2.pathway_id} slot_period="
            f"{t2.proposed_slot and t2.proposed_slot['period']}")
        assert t2.pathway_id == "CP-MSK-002", "short-term (checkpointed) context lost across restart"
        assert t2.proposed_slot["period"] == "afternoon"
    log("RESULT: PASS - long-term recall across sessions, patient isolation, short-term thread recall across restart")
    flush_log()


def _ref(pid: str) -> str:
    from src.guardrails.phi import patient_ref
    return patient_ref(pid)
