"""Input guardrail (AC-06, AC-10). Two stages on the agent's input path:

1. `ingress()` - runs at the service boundary BEFORE anything is persisted, traced or sent to a model:
   detects cross-patient references on the raw text, then PHI-masks it.
2. `evaluate_input()` - called by the graph's first node (`input_guard`): length, prompt injection
   (patterns + tag forgery + role spoofing + optional LLM Guard), cross-patient flags, scope flags.
   Decision: allow | sanitize | block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import get_settings
from src.guardrails.phi import PATIENT_ID_RE, SYN_NAME_RE, mask_text, patient_ref
from src.state import GuardResult, QuarantinedText


@dataclass
class IngressResult:
    masked_text: str
    flags: list[str] = field(default_factory=list)


def ingress(raw_text: str, session_patient_id: str) -> IngressResult:
    flags: list[str] = []
    session_pid = session_patient_id.strip().upper()
    for m in PATIENT_ID_RE.finditer(raw_text):
        if m.group(0).upper() != session_pid:
            flags.append(f"cross_patient_id:{patient_ref(m.group(0))}")
    session_num = int(session_pid.rsplit("-", 1)[-1]) if session_pid[-1:].isdigit() else -1
    for m in SYN_NAME_RE.finditer(raw_text):
        if int(m.group(0).rsplit(" ", 1)[-1]) != session_num:
            flags.append("cross_patient_identity_claim")
    masked = mask_text(raw_text)
    if masked != raw_text:
        flags.append("phi_masked")
    return IngressResult(masked_text=masked, flags=flags)


@lru_cache(maxsize=1)
def _compiled(key: str) -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in get_settings().guardrails.get(key, [])]


def is_diagnosis_request(text: str) -> bool:
    return any(p.search(text) for p in _compiled("diagnosis_request_patterns"))


def evaluate_input(q: QuarantinedText, ingress_flags: list[str]) -> GuardResult:
    cfg = get_settings().guardrails
    reasons, flags = [], list(ingress_flags) + list(q.flags)
    if len(q.masked_text) > cfg.get("max_input_chars", 2000):
        reasons.append("input_too_long")
    if any(f.startswith("cross_patient") for f in flags):
        reasons.append("cross_patient_access_attempt")
    if q.injection_score >= cfg.get("injection_block_score", 0.6):
        reasons.append(f"prompt_injection(score={q.injection_score:.2f})")
    if is_diagnosis_request(q.masked_text):
        flags.append("diagnosis_request")  # not blocked: routed to a clinician by the supervisor
    if reasons:
        return GuardResult(decision="block", reasons=reasons, flags=flags)
    if "phi_masked" in flags or q.injection_score > 0:
        return GuardResult(decision="sanitize", reasons=["phi_masked_or_suspicious_content"], flags=flags)
    return GuardResult(decision="allow", flags=flags)
