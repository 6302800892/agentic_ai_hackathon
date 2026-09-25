"""Synthetic patient store + coverage rules (COV-001) + session scoping.

Pure functions; the only I/O is reading data/patients/patients.json. Returned records are masked: no name, DOB,
phone, email, MRN or raw patient id ever leaves this module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from mcp_server.schemas import SERVICE_CODES, CoverageOut, PatientRecordOut, Slot, SlotsOut, ToolError

ROOT = Path(__file__).resolve().parents[1]
PATIENTS_FILE = ROOT / "data" / "patients" / "patients.json"


@lru_cache(maxsize=1)
def _salt() -> bytes:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return os.getenv("PHI_HMAC_SALT", "dev-only-salt-change-me").encode()


def _ref(patient_id: str) -> str:
    return "PT-" + hmac.new(_salt(), patient_id.strip().upper().encode(), hashlib.sha256).hexdigest()[:8]


def expected_token(session_id: str, patient_ref: str) -> str:
    return hmac.new(_salt(), f"{session_id}:{patient_ref}".encode(), hashlib.sha256).hexdigest()[:32]


@lru_cache(maxsize=1)
def _patients_by_ref() -> dict[str, dict]:
    patients = json.loads(PATIENTS_FILE.read_text(encoding="utf-8"))
    return {_ref(p["patient_id"]): p for p in patients}


def _reference_date() -> str:
    return os.getenv("COPILOT_REFERENCE_DATE", "2026-09-25")


def authorize(patient_ref: str, session_id: str, session_token: str) -> ToolError | None:
    if not hmac.compare_digest(expected_token(session_id, patient_ref), session_token or ""):
        return ToolError(error="ACCESS_DENIED", detail="session is not authorised for this patient")
    if patient_ref not in _patients_by_ref():
        return ToolError(error="NOT_FOUND", detail="unknown patient_ref")
    return None


def get_patient_record(patient_ref: str, session_id: str, session_token: str) -> dict:
    err = authorize(patient_ref, session_id, session_token)
    if err:
        return err.model_dump()
    p = _patients_by_ref()[patient_ref]
    return PatientRecordOut(
        patient_ref=patient_ref, plan_id=p["plan"]["plan_id"], plan_status=p["plan"]["status"],
        plan_effective=p["plan"]["effective"], plan_term=p["plan"]["term"],
        covered_services=p["covered_services"], requires_referral=p["requires_referral"],
        requires_prior_auth=p["requires_prior_auth"], referrals_on_file=p["referrals_on_file"],
        prior_auth_on_file=p["prior_auth_on_file"], preferred_language=p["preferred_language"],
    ).model_dump()


def check_coverage(patient_ref: str, service_code: str, session_id: str, session_token: str,
                   service_date: str | None = None) -> dict:
    """Apply COV-R1..R6 (data/policy_corpus/COV-001.md) deterministically."""
    err = authorize(patient_ref, session_id, session_token)
    if err:
        return err.model_dump()
    p = _patients_by_ref()[patient_ref]
    plan = p["plan"]
    gaps: list[dict] = []
    applied: list[str] = []
    code = (service_code or "").upper()

    if code not in SERVICE_CODES:
        gaps.append({"rule_id": "COV-R6", "description": f"Service code '{service_code}' cannot be determined."})
        return CoverageOut(eligible=False, plan_status=plan["status"], service_code=code or "UNKNOWN",
                           gaps=gaps, referral_required=False, rules_applied=["COV-R6"]).model_dump()
    if code == "ADMIN":
        return CoverageOut(eligible=True, plan_status=plan["status"], service_code=code, gaps=[],
                           referral_required=False, rules_applied=[]).model_dump()

    applied.append("COV-R1")
    if plan["status"] != "active":
        gaps.append({"rule_id": "COV-R1", "description": f"Plan status is '{plan['status']}', not active."})
    applied.append("COV-R2")
    d = service_date or _reference_date()
    if not (plan["effective"] <= d <= plan["term"]):
        gaps.append({"rule_id": "COV-R2", "description":
                     f"Service date {d} is outside the coverage period {plan['effective']}..{plan['term']}."})
    applied.append("COV-R3")
    if code not in p["covered_services"]:
        gaps.append({"rule_id": "COV-R3", "description": f"{code} is not a covered service under {plan['plan_id']}."})
    referral_required = code in p["requires_referral"]
    if referral_required:
        applied.append("COV-R4")
        if code not in p["referrals_on_file"]:
            gaps.append({"rule_id": "COV-R4", "description": f"{code} requires a primary-care referral; none on file."})
    if code in p["requires_prior_auth"]:
        applied.append("COV-R5")
        status = p["prior_auth_on_file"].get(code)
        if status != "approved":
            gaps.append({"rule_id": "COV-R5", "description":
                         f"{code} requires approved prior authorisation; status is '{status or 'none'}'."})
    return CoverageOut(eligible=not gaps, plan_status=plan["status"], service_code=code, gaps=gaps,
                       referral_required=referral_required, rules_applied=applied).model_dump()


CLINICS = {"MSK": "Physiotherapy Unit B", "DERM": "Dermatology Clinic", "CARDIO": "Cardiology Outpatients",
           "MENTAL_HEALTH": "Wellbeing Centre", "PEDS": "Childrens Clinic", "GENERAL": "Primary Care Suite 2",
           "ADMIN": "Front Desk"}
_PATHWAY_KEYS = {"MSK": "MSK", "DERM": "DERM", "CARDIO": "CARDIO", "MH": "MENTAL_HEALTH",
                 "MENTAL_HEALTH": "MENTAL_HEALTH", "PEDS": "PEDS", "GEN": "GENERAL", "GENERAL": "GENERAL",
                 "ADMIN": "ADMIN"}


def pathway_key(pathway_id: str) -> str | None:
    """'CP-MSK-002' -> 'MSK', 'MENTAL_HEALTH' -> 'MENTAL_HEALTH'."""
    pid = (pathway_id or "").upper()
    if pid in _PATHWAY_KEYS:
        return _PATHWAY_KEYS[pid]
    parts = pid.split("-")
    return _PATHWAY_KEYS.get(parts[1]) if len(parts) >= 2 and parts[0] == "CP" else None


def list_available_slots(pathway_id: str, urgency: str = "routine") -> dict:
    key = pathway_key(pathway_id)
    if key is None:
        return ToolError(error="INVALID_INPUT", detail=f"unknown pathway_id '{pathway_id}'").model_dump()
    base = datetime.fromisoformat(_reference_date()) + timedelta(days=1)
    day = base + timedelta(days={"urgent": 0, "soon": 1, "routine": 2}.get(urgency, 2))
    slots: list[Slot] = []
    while len(slots) < 4:
        if day.weekday() < 5:
            for hour, period in ((9, "morning"), (14, "afternoon")):
                start = day.replace(hour=hour, minute=30 if key in ("MSK", "PEDS") else 0)
                slots.append(Slot(slot_id=f"SL-{key[:3]}-{start:%m%d%H%M}", clinic=CLINICS[key],
                                  start=start.isoformat(timespec="minutes"), period=period))
        day += timedelta(days=1)
    return SlotsOut(pathway_id=pathway_id, slots=slots[:4]).model_dump()
