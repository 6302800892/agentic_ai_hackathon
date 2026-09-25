"""Intake-classification agent: reason for visit, intent, urgency, service code, stated preferences.

Gemini-light with structured output (IntakeClassification). A deterministic red-flag / clinical-question
safety net always runs and can only make the result *safer* (escalate), never less safe.
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from src.agents import heuristics
from src.context.isolate import merge_result
from src.context.select import context_for_classifier
from src.state import CopilotState, IntakeClassification

_TAG = re.compile(r"</?untrusted_patient_input[^>]*>")


def _history_text(state: CopilotState) -> str:
    human = [m for m in state.get("messages", [])[:-1] if isinstance(m, HumanMessage)]
    return " ".join(_TAG.sub("", str(m.content)) for m in human[-4:]) + " " + state.get("summary", "")


def apply_safety_net(result: IntakeClassification, rules: IntakeClassification, flags: list[str]) -> IntakeClassification:
    if rules.red_flags and result.intent != "urgent":
        return result.model_copy(update={"intent": "urgent", "urgency": "urgent",
                                         "red_flags": sorted(set(result.red_flags) | set(rules.red_flags))})
    if (rules.intent == "clinical_question" or "diagnosis_request" in flags) and \
            result.intent not in ("urgent", "clinical_question"):
        return result.model_copy(update={"intent": "clinical_question", "urgency": "soon"})
    return result


async def run(state: CopilotState, runtime) -> dict:
    q = state["quarantined_input"]
    history = _history_text(state)
    rules = heuristics.classify(q.masked_text, history)
    result = await runtime.structured(IntakeClassification, context_for_classifier(state, runtime.intake_rules),
                                      light=True, agent="intake_classifier")
    if result is None:
        result = rules
    else:
        result = result.model_copy(update={"source": "llm",
                                           "stated_preferences": result.stated_preferences or rules.stated_preferences})
    guard = state.get("guard_input")
    result = apply_safety_net(result, rules, guard.flags if guard else [])
    return merge_result("intake_classifier", {"intake": result})
