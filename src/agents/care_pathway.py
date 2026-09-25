"""Care-pathway retrieval agent - agentic RAG (retrieval-in-the-loop).

  round 1..N (N = rag_max_rounds):  retrieve (search_care_policy)  ->  grade relevance (Gemini-light,
  structured RelevanceGrade; rule-based grade if no model)  ->  if insufficient, rewrite the query and retrieve again.
Then a targeted retrieval for any coverage gap / referral rule so the draft can cite COV-001 / REF-001.
If no care-pathway policy is found after N rounds -> pathway_id="NONE" (supervisor escalates; never invented).
"""
from __future__ import annotations

from src.config import get_settings
from src.context.isolate import merge_result
from src.context.select import context_for_grader
from src.state import Citation, CopilotState, ErrorRecord, PathwayResult, RelevanceGrade

AGENT = "care_pathway"
PATHWAY_POLICY = {"MSK": "CP-MSK-002", "DERM": "CP-DERM-001", "CARDIO": "CP-CARDIO-003", "MENTAL_HEALTH": "CP-MH-001",
                  "PEDS": "CP-PEDS-001", "GENERAL": "CP-GEN-001", "ADMIN": "CP-ADMIN-001"}
CATEGORY_TERMS = {"MSK": "musculoskeletal back joint sprain physiotherapy", "DERM": "dermatology skin mole rash",
                  "CARDIO": "cardiology blood pressure non-urgent", "MENTAL_HEALTH": "mental health stress anxiety counselling",
                  "PEDS": "paediatrics child under 16", "GENERAL": "general primary care check-up",
                  "ADMIN": "administrative records billing rescheduling"}


def initial_query(intake) -> str:
    cat = intake.reason_for_visit_category
    return f"{CATEGORY_TERMS.get(cat, 'care pathway')} routing entry criteria {intake.service_code.lower().replace('_', ' ')}"


def rule_grade(chunks: list[dict], hint: str | None, intake) -> RelevanceGrade:
    want = PATHWAY_POLICY.get(hint or "")
    if want and any(c["policy_id"] == want for c in chunks):
        return RelevanceGrade(relevant=True)
    if not want and chunks:
        return RelevanceGrade(relevant=True)
    return RelevanceGrade(relevant=False, missing=f"no {hint} pathway section",
                          rewritten_query=f"{want or ''} care pathway {hint or ''} {CATEGORY_TERMS.get(hint or '', '')} routing")


def _routing_text(chunks: list[dict], policy_id: str) -> str:
    for c in chunks:
        if c["policy_id"] == policy_id and c["section"] == "§2":
            return c["text"].split("\n", 1)[-1].strip()[:400]
    for c in chunks:
        if c["policy_id"] == policy_id:
            return c["text"][:400]
    return ""


async def run(state: CopilotState, runtime) -> dict:
    limits = get_settings().limits
    intake = state["intake"]
    hint = intake.reason_for_visit_category if intake.reason_for_visit_category in PATHWAY_POLICY else None
    query, queries, found, rounds = initial_query(intake), [], {}, 0
    errors = list(state.get("errors") or [])

    for rounds in range(1, limits.get("rag_max_rounds", 3) + 1):
        queries.append(query)
        res = await runtime.tools.call(runtime.rag_tool_name, {"query": query, "pathway_hint": hint,
                                                               "k": limits.get("rag_top_k", 4)}, agent=AGENT)
        if not res.ok:
            errors.append(ErrorRecord(node=AGENT, error=f"rag {res.status}: {res.error}", fatal=False))
            break
        chunks = res.data["chunks"]
        found.update({c["chunk_id"]: c for c in chunks})
        grade = await runtime.structured(RelevanceGrade, context_for_grader(query, chunks), light=True, agent=AGENT)
        # the model may judge relevance, but a pathway policy must actually be present to stop
        rules = rule_grade(chunks, hint, intake)
        if grade is None or (grade.relevant and not rules.relevant):
            grade = rules
        if grade.relevant:
            break
        query = grade.rewritten_query or rules.rewritten_query

    # targeted retrieval for coverage / referral rules that the draft must cite
    cov = state.get("coverage")
    rule_ids = [g.rule_id for g in cov.gaps] if cov else []
    if rule_ids or intake.intent == "referral":
        q2 = f"coverage eligibility rule {' '.join(rule_ids)} referral requirement".strip()
        queries.append(q2)
        res = await runtime.tools.call(runtime.rag_tool_name, {"query": q2, "k": 3}, agent=AGENT)
        if res.ok:
            found.update({c["chunk_id"]: c for c in res.data["chunks"]})

    chunks = sorted(found.values(), key=lambda c: -c["score"])
    want = PATHWAY_POLICY.get(hint or "")
    pathway_chunks = [c for c in chunks if c["policy_id"].startswith("CP-")]
    if want and any(c["policy_id"] == want for c in chunks):
        pathway_id = want
    elif pathway_chunks:
        pathway_id = pathway_chunks[0]["policy_id"]
    else:
        pathway_id = "NONE"

    cite = [c for c in chunks if c["policy_id"] == pathway_id][:2]
    cite += [c for c in chunks if c["policy_id"] in ("COV-001", "REF-001") and (rule_ids or intake.intent == "referral")][:2]
    result = PathwayResult(
        pathway_id=pathway_id, recommended_action=_routing_text(chunks, pathway_id) if pathway_id != "NONE" else "",
        citations=[Citation(policy_id=c["policy_id"], section=c["section"], chunk_id=c["chunk_id"]) for c in cite],
        retrieval_rounds=rounds, queries=queries)
    return merge_result(AGENT, {"pathway": result, "retrieved_chunks": chunks, "errors": errors})
