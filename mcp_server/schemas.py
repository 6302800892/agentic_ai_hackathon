"""Tool contracts shared by the MCP server, the in-process adapter and tests/test_tool_contracts.py."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SERVICE_CODES = ["PRIMARY_CARE", "PHYSIO", "DERM_REFERRAL", "CARDIO_REFERRAL", "MENTAL_HEALTH", "PEDIATRICS",
                 "IMAGING_MRI", "ADMIN"]
PATHWAYS = ["MSK", "DERM", "CARDIO", "MENTAL_HEALTH", "PEDS", "GENERAL", "ADMIN"]


class GetPatientRecordInput(BaseModel):
    patient_ref: str = Field(pattern=r"^PT-[0-9a-f]{8}$")
    session_id: str
    session_token: str


class PatientRecordOut(BaseModel):
    patient_ref: str
    plan_id: str
    plan_status: str
    plan_effective: str
    plan_term: str
    covered_services: list[str]
    requires_referral: list[str]
    requires_prior_auth: list[str]
    referrals_on_file: list[str]
    prior_auth_on_file: dict[str, str]
    preferred_language: str


class CheckCoverageInput(BaseModel):
    patient_ref: str = Field(pattern=r"^PT-[0-9a-f]{8}$")
    service_code: str
    session_id: str
    session_token: str
    service_date: Optional[str] = None


class CoverageGapOut(BaseModel):
    rule_id: str
    description: str


class CoverageOut(BaseModel):
    eligible: bool
    plan_status: str
    service_code: str
    gaps: list[CoverageGapOut]
    referral_required: bool
    rules_applied: list[str]


class ListSlotsInput(BaseModel):
    pathway_id: str
    urgency: Literal["routine", "soon", "urgent"] = "routine"


class Slot(BaseModel):
    slot_id: str
    clinic: str
    start: str
    period: Literal["morning", "afternoon"]


class SlotsOut(BaseModel):
    pathway_id: str
    slots: list[Slot]


class ToolError(BaseModel):
    error: Literal["ACCESS_DENIED", "INVALID_INPUT", "NOT_FOUND"]
    detail: str
