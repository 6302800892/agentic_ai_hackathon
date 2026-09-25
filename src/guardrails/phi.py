"""PHI / PII masking (NFR-05, AC-06).

Two layers:
  1. Deterministic recognisers for the synthetic identifier formats (patient id, MRN, phone, email, name, DOB).
     Patient ids become a stable HMAC token `PT-xxxxxxxx` so logs stay joinable without exposing the id.
  2. Microsoft Presidio (PERSON / PHONE_NUMBER / EMAIL_ADDRESS) for free text, loaded lazily with en_core_web_sm.

Used at ingress (before anything is persisted or sent to the model), on every log / audit / trace write,
and on the final answer (output guard).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from functools import lru_cache
from typing import Any

from src.config import get_settings

log = logging.getLogger(__name__)

PATIENT_ID_RE = re.compile(r"\bSYN-P-\d{5}\b", re.I)
MRN_RE = re.compile(r"\bSYN-MRN-\d{4,}\b", re.I)
SYN_NAME_RE = re.compile(r"\bSynthetic Person \d+\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
DOB_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
PATIENT_REF_RE = re.compile(r"\bPT-[0-9a-f]{8}\b")

# Patterns that must NEVER appear in committed logs / traces / reports (used by tests + verify_citations).
PLAINTEXT_PHI_PATTERNS = {
    "patient_id": PATIENT_ID_RE,
    "mrn": MRN_RE,
    "synthetic_name": SYN_NAME_RE,
    "synthetic_phone": re.compile(r"\b555-0100-\d{3}\b"),
    "synthetic_email": re.compile(r"synthetic\.person\d+@", re.I),
}

SENSITIVE_KEYS = {"name", "dob", "phone", "email", "mrn", "address"}


def patient_ref(patient_id: str) -> str:
    """Stable, non-reversible pseudonym for a patient id."""
    salt = get_settings().phi_salt.encode()
    digest = hmac.new(salt, patient_id.strip().upper().encode(), hashlib.sha256).hexdigest()
    return f"PT-{digest[:8]}"


def session_token(session_id: str, ref: str) -> str:
    """Token binding one session to one patient_ref; verified by the MCP server."""
    salt = get_settings().phi_salt.encode()
    return hmac.new(salt, f"{session_id}:{ref}".encode(), hashlib.sha256).hexdigest()[:32]


def mask_identifiers(text: str, dates: bool = False) -> str:
    """Fast deterministic masking of the synthetic identifier formats (dates only for free text)."""
    if not text:
        return text
    text = PATIENT_ID_RE.sub(lambda m: patient_ref(m.group(0)), text)
    text = MRN_RE.sub("[MRN]", text)
    text = SYN_NAME_RE.sub("[NAME]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    if dates:
        text = DOB_RE.sub("[DATE]", text)
    text = PHONE_RE.sub(lambda m: "[PHONE]" if sum(c.isdigit() for c in m.group(0)) >= 7 else m.group(0), text)
    return text


@lru_cache(maxsize=1)
def _presidio():
    if os.getenv("COPILOT_DISABLE_PRESIDIO") == "1":
        return None
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)  # unmapped spaCy labels are noise

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        return analyzer, AnonymizerEngine()
    except Exception as e:  # graceful degradation: regex layer still applies
        log.warning("Presidio unavailable, using regex PHI masking only: %s", e)
        return None


def mask_text(text: str, use_presidio: bool = True, dates: bool = True) -> str:
    """Full masking for free text. dates=True for patient-supplied text (a date may be a DOB); system-generated
    text (slot times, visit dates) is masked with dates=False."""
    if not text:
        return text
    text = mask_identifiers(text, dates=dates)
    engines = _presidio() if use_presidio else None
    if engines:
        from presidio_anonymizer.entities import OperatorConfig

        analyzer, anonymizer = engines
        results = analyzer.analyze(text=text, language="en",
                                   entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"], score_threshold=0.75)
        # never re-mask our own placeholders / pseudonyms
        results = [r for r in results if not PATIENT_REF_RE.search(text[r.start:r.end])
                   and "[" not in text[r.start:r.end]]
        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results, operators={
                "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            }).text
    return text


def mask_obj(obj: Any) -> Any:
    """Recursively mask a JSON-like object before it is written to any log / trace."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in SENSITIVE_KEYS and v not in (None, ""):
                out[k] = "[REDACTED]"
            elif k == "patient_id" and isinstance(v, str):
                out[k] = patient_ref(v)
            else:
                out[k] = mask_obj(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [mask_obj(v) for v in obj]
    if isinstance(obj, str):
        return mask_identifiers(obj)
    return obj


def find_plaintext_phi(text: str) -> list[str]:
    """Return names of PHI patterns found (empty list == clean)."""
    return [name for name, rx in PLAINTEXT_PHI_PATTERNS.items() if rx.search(text or "")]
