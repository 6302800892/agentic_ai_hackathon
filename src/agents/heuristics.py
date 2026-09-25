"""Deterministic rules used in two ways:

1. **Safety net** - red-flag and clinical-question detection runs on every request, even when Gemini classified
   it. If the rules see a red flag, the case is escalated whatever the model said (AC-03).
2. **Graceful degradation** - if Gemini is unavailable (no key, timeout, quota), the intake classifier falls back
   to these rules so the copilot still routes safely (NFR-04). Outputs are tagged source="heuristic_fallback".
"""
from __future__ import annotations

import re

from src.state import IntakeClassification

RED_FLAGS = {
    "chest pain": r"chest (pain|pressure|tightness)|crushing chest",
    "radiating pain": r"(spread|radiat)\w* (to|down|into) (my |the )?(left )?(arm|jaw)",
    "breathing difficulty": r"(hard|difficult\w*|trouble|can'?t|cannot) (to )?breath|short(ness)? of breath|breathless",
    "stroke signs": r"slurred speech|speech is slurred|face (is )?droop|drooping|one side of my (face|body)",
    "severe bleeding": r"(severe|heavy|won'?t stop) bleed",
    "loss of consciousness": r"passed out|fainted|unconscious|blacked out",
    "seizure": r"seizure|convuls",
    "self-harm": r"suicid|kill myself|hurt(ing)? myself|self[- ]harm|end my life",
    "overdose": r"overdose|took too many (pills|tablets)",
    "anaphylaxis": r"anaphyla|throat (is )?(closing|swelling)|tongue swelling",
}
CLINICAL_Q = r"what do i have|diagnos|what('?s| is) wrong with me|is (it|this) (cancer|serious)|" \
             r"what (disease|condition|illness)|(which|what) (medication|medicine|drug)|what dose|how many mg|" \
             r"should i take"
OUT_OF_SCOPE = r"recipe|weather|stock (price|market)|poem|movie|lottery|crypto|homework|joke|football score"

CATEGORY_RULES = [
    ("PEDS", r"\b(daughter|son|child|kid|baby|toddler|\d+ ?(year|yr)s?[ -]old)\b"),
    ("DERM", r"skin|rash|mole|eczema|acne|dermatolog|itch"),
    ("CARDIO", r"heart|cardio|blood pressure|cholesterol|palpitation"),
    ("MENTAL_HEALTH", r"stress|anxi|depress|low mood|can'?t sleep|insomnia|counsel|\btherap(y|ist)|talk to someone"),
    ("MSK", r"back|knee|shoulder|ankle|neck|joint|sprain|strain|twist|physio|muscle|hip|wrist"),
    ("ADMIN", r"record|bill|invoice|reschedul|cancel|address|paperwork|copy of"),
]
SERVICE_FOR = {"MSK": "PHYSIO", "DERM": "DERM_REFERRAL", "CARDIO": "CARDIO_REFERRAL",
               "MENTAL_HEALTH": "MENTAL_HEALTH", "PEDS": "PEDIATRICS", "ADMIN": "ADMIN", "GENERAL": "PRIMARY_CARE"}


def red_flags(text: str) -> list[str]:
    t = text.lower()
    return [name for name, rx in RED_FLAGS.items() if re.search(rx, t)]


def is_clinical_question(text: str) -> bool:
    return bool(re.search(CLINICAL_Q, text.lower()))


def category_of(text: str) -> str:
    t = text.lower()
    for cat, rx in CATEGORY_RULES:
        if re.search(rx, t):
            return cat
    if re.search(r"check-?up|gp|doctor|appointment|general|physical", t):
        return "GENERAL"
    return "UNKNOWN"


def preferences_of(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"\b(mornings?|afternoons?|evenings?)( only)?\b", text, re.I)]


def classify(current: str, history: str = "") -> IntakeClassification:
    """Rule-based intake classification. `history` = earlier turns of this session (short-term memory)."""
    t = current.lower()
    flags = red_flags(current)
    cat = category_of(current)
    if cat in ("UNKNOWN", "GENERAL") and history:
        earlier = category_of(history)  # use facts stated earlier in the session (AC-05)
        if earlier not in ("UNKNOWN", "GENERAL"):
            cat = earlier
    service = SERVICE_FOR.get(cat, "UNKNOWN")
    if re.search(r"\bmri\b|scan|x-?ray|imaging", t):
        service = "IMAGING_MRI" if "mri" in t else service

    def out(intent, conf, **kw):
        return IntakeClassification(intent=intent, reason_for_visit_category=kw.pop("cat", cat),
                                    service_code=kw.pop("service", service), confidence=conf,
                                    stated_preferences=preferences_of(current), source="heuristic_fallback", **kw)

    if flags:
        return out("urgent", 0.9, urgency="urgent", red_flags=flags)
    if is_clinical_question(current):
        return out("clinical_question", 0.85, urgency="soon")
    if re.search(OUT_OF_SCOPE, t) and cat in ("UNKNOWN", "GENERAL"):
        return out("out_of_scope", 0.8, cat="UNKNOWN", service="UNKNOWN")
    if re.search(r"covered|coverage|insurance|eligib|copay|does my plan|my plan cover", t):
        return out("coverage_question", 0.75)
    if cat == "ADMIN":
        return out("admin", 0.75)
    if re.search(r"referr|refer me|specialist|dermatologist|cardiologist", t) or service in ("DERM_REFERRAL",
                                                                                              "CARDIO_REFERRAL"):
        return out("referral", 0.75)
    if cat != "UNKNOWN" or re.search(r"book|appointment|schedule|see (a|the|someone)|visit|check-?up|talk to someone", t):
        if cat == "UNKNOWN":
            cat, service = "GENERAL", "PRIMARY_CARE"
        return out("schedule", 0.7, cat=cat, service=service)
    return out("ambiguous", 0.4, needs_clarification=True,
               clarifying_question="Could you tell me a little more about what you need help with today - for "
                                   "example booking an appointment, a referral, a coverage question, or records?")
