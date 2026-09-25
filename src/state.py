"""Typed graph state + Pydantic structured outputs used at every node boundary."""
from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Intent = Literal["schedule", "referral", "coverage_question", "admin", "clinical_question",
                 "urgent", "ambiguous", "out_of_scope"]
Category = Literal["MSK", "DERM", "CARDIO", "MENTAL_HEALTH", "PEDS", "ADMIN", "GENERAL", "UNKNOWN"]
ServiceCode = Literal["PRIMARY_CARE", "PHYSIO", "DERM_REFERRAL", "CARDIO_REFERRAL", "MENTAL_HEALTH",
                      "PEDIATRICS", "IMAGING_MRI", "ADMIN", "UNKNOWN"]
Action = Literal["schedule", "refer", "escalate", "clarify", "decline"]
RiskTier = Literal["low", "medium", "high"]
WorkerName = Literal["intake_classifier", "coverage_checker", "care_pathway", "coordinator",
                     "clarify", "human_escalation"]


class QuarantinedText(BaseModel):
    """Untrusted patient-supplied text. Always data, never instructions (NFR-03)."""
    quarantine_id: str
    masked_text: str
    flags: list[str] = Field(default_factory=list)
    injection_score: float = 0.0

    def wrapped(self) -> str:
        return (f'<untrusted_patient_input id="{self.quarantine_id}">\n{self.masked_text}\n'
                f"</untrusted_patient_input>")


class GuardResult(BaseModel):
    decision: Literal["allow", "sanitize", "block"]
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class IntakeClassification(BaseModel):
    intent: Intent
    reason_for_visit_category: Category = "UNKNOWN"
    service_code: ServiceCode = "UNKNOWN"
    urgency: Literal["routine", "soon", "urgent"] = "routine"
    red_flags: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    stated_preferences: list[str] = Field(default_factory=list,
                                          description="e.g. 'mornings only' stated by the patient")
    source: Literal["llm", "heuristic_fallback"] = "llm"


class CoverageGap(BaseModel):
    rule_id: str
    description: str


class CoverageResult(BaseModel):
    eligible: bool
    plan_status: str
    service_code: str
    gaps: list[CoverageGap] = Field(default_factory=list)
    referral_required: bool = False
    explanation: str = ""
    source: str = "mcp:check_coverage"


class Citation(BaseModel):
    policy_id: str
    section: str
    chunk_id: str


class PathwayResult(BaseModel):
    pathway_id: str
    recommended_action: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_rounds: int = 1
    queries: list[str] = Field(default_factory=list)


class RelevanceGrade(BaseModel):
    relevant: bool
    missing: str = ""
    rewritten_query: str = ""


class NextStepDraft(BaseModel):
    action: Action
    rationale: str
    citations: list[Citation] = Field(default_factory=list)
    requires_clinician: bool = False
    patient_message: str
    staff_note: str = ""
    proposed_slot: Optional[dict] = None


class DraftText(BaseModel):
    """What the LLM is allowed to write for the coordinator: wording only, not the decision."""
    patient_message: str
    staff_note: str
    rationale: str


class SupervisorDecision(BaseModel):
    next: Literal["intake_classifier", "coverage_checker", "care_pathway", "coordinator",
                  "clarify", "human_escalation"]
    reason: str


class MemoryItem(BaseModel):
    key: str
    text: str
    kind: str = "fact"
    score: float = 0.0


class ErrorRecord(BaseModel):
    node: str
    error: str
    fatal: bool = False


class FinalResponse(BaseModel):
    request_id: str
    action: Action
    risk_tier: RiskTier
    requires_clinician: bool
    patient_message: str
    staff_note: str = ""
    citations: list[Citation] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    intent: Optional[str] = None
    pathway_id: Optional[str] = None
    proposed_slot: Optional[dict] = None
    guard_notes: list[str] = Field(default_factory=list)
    recalled_memories: list[str] = Field(default_factory=list)
    run_id: Optional[str] = None
    # Transparency (EU AI Act Art. 50): every response discloses that it was drafted by an AI system.
    disclosure: str = ("Drafted by an AI care-coordination assistant. It does not give diagnoses; "
                       "a clinician makes all clinical decisions.")


class CopilotState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request_id: str
    session_id: str
    thread_id: str
    patient_ref: str               # masked HMAC token, never the raw patient id
    session_token: str             # binds this session to one patient (MCP access control)
    input_text: str                # already PHI-masked at ingress
    ingress_flags: list[str]       # flags computed on raw text at ingress (e.g. cross-patient reference)
    quarantined_input: Optional[QuarantinedText]
    guard_input: Optional[GuardResult]
    intake: Optional[IntakeClassification]
    coverage: Optional[CoverageResult]
    pathway: Optional[PathwayResult]
    retrieved_chunks: list[dict]
    next_step: Optional[NextStepDraft]
    supervisor: Optional[SupervisorDecision]
    recalled_memories: list[MemoryItem]
    summary: str
    route_history: list[str]
    step_count: int
    errors: list[ErrorRecord]
    final_response: Optional[FinalResponse]


def fresh_turn_state(**kw) -> dict:
    """Per-turn fields reset on every request (checkpointer persists messages/summary across turns)."""
    base = dict(quarantined_input=None, guard_input=None, intake=None, coverage=None, pathway=None,
                retrieved_chunks=[], next_step=None, supervisor=None, recalled_memories=[],
                route_history=[], step_count=0, errors=[], final_response=None)
    base.update(kw)
    return base
