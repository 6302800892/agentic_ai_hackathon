"""Offline test fixtures: no API key, no network, no writes to the committed evidence logs.

  * GOOGLE_API_KEY is blanked -> every model call degrades to the deterministic rules (the fallback path).
  * COPILOT_LOG_DIR -> a temp dir, so tool/audit/MCP logs from tests never mix with committed evidence.
  * keyword RAG backend (no embedding model download); tools in-process unless a test opts into MCP.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["GOOGLE_API_KEY"] = ""
os.environ["COPILOT_LOG_DIR"] = tempfile.mkdtemp(prefix="copilot-test-logs-")
os.environ["COPILOT_RAG_BACKEND"] = "keyword"
os.environ["COPILOT_DISABLE_PRESIDIO"] = "1"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.guardrails.phi import patient_ref, session_token  # noqa: E402
from src.state import (CoverageGap, CoverageResult, IntakeClassification, PathwayResult,  # noqa: E402
                       QuarantinedText)


@pytest.fixture
def make_copilot(tmp_path):
    """Factory: make_copilot(**kw) -> un-entered Copilot using temp SQLite files (shareable across instances)."""
    from src.service import Copilot

    def _make(**kw):
        kw.setdefault("transport", "local")
        kw.setdefault("use_llm", False)
        kw.setdefault("rag_backend", "keyword")
        kw.setdefault("checkpoint_path", tmp_path / "checkpoints.sqlite")
        kw.setdefault("memory_path", tmp_path / "memory.sqlite")
        return Copilot(**kw)
    return _make


@pytest.fixture
async def copilot(make_copilot):
    async with make_copilot() as cp:
        yield cp


def auth(session_id: str = "S-T", patient_id: str = "SYN-P-00005") -> dict:
    ref = patient_ref(patient_id)
    return {"patient_ref": ref, "session_id": session_id, "session_token": session_token(session_id, ref)}


def intake(intent="schedule", **kw) -> IntakeClassification:
    kw.setdefault("reason_for_visit_category", "MSK")
    kw.setdefault("service_code", "PHYSIO")
    kw.setdefault("confidence", 0.8)
    return IntakeClassification(intent=intent, **kw)


def coverage(eligible=True, gaps=()) -> CoverageResult:
    return CoverageResult(eligible=eligible, plan_status="active", service_code="PHYSIO",
                          gaps=[CoverageGap(rule_id=g, description=g) for g in gaps])


def pathway(pid="CP-MSK-002") -> PathwayResult:
    return PathwayResult(pathway_id=pid, recommended_action="")


def quarantined(text="test") -> QuarantinedText:
    return QuarantinedText(quarantine_id="q-test", masked_text=text)
