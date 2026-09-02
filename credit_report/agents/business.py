"""
Business & Management specialist subgraph.

Unlike Finance/Hygiene, this agent doesn't fetch its own primary data — it
reasons over what Finance and Hygiene already found (handed to it as
context, not re-fetched) plus the rating rationale text when the entity has
no ticker (reusing tools/rating_agency_tools.py, which caches, so this
doesn't re-hit CRISIL for data Finance already pulled), with search_web_news
as its only real tool for anything not already covered.

The LLM must tag each of the 4 dimensions grounded/inferred/unavailable
rather than write one blended paragraph — completeness is then counted
deterministically from those tags (grounded_count / 4), never trusted from
the LLM's own claim, same discipline as Finance and Hygiene.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent

from credit_report.model import build_model
from credit_report.prompt_loader import load_prompt
from credit_report.schemas import BusinessLLMAssessment, BusinessResult, _BUSINESS_DIMENSIONS
from credit_report.tool_capture import capturing
from tools.rating_agency_tools import get_rating_rationale
from tools.search_tools import search_web_news

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("business")

# Named so tests can assert on it directly (tool-isolation check) — this
# agent gets exactly search_web_news; rating-rationale fetching happens
# deterministically below, not as an agent-invoked tool.
_BASE_TOOLS = (search_web_news,)


def _build_context(state: dict[str, Any]) -> str:
    parts: list[str] = []

    finance = state.get("finance_result") or {}
    if finance.get("summary"):
        parts.append(
            f"Finance assessment (score {finance.get('raw_score')}/100, "
            f"data source: {finance.get('data_path')}): {finance['summary']}"
        )

    hygiene = state.get("hygiene_result") or {}
    if hygiene.get("summary"):
        parts.append(f"Hygiene/AML assessment (score {hygiene.get('raw_score')}/100): {hygiene['summary']}")

    if not state.get("has_ticker"):
        entity_name = state.get("company_name") or state.get("company_reference", "")
        rationale = get_rating_rationale(entity_name)
        if rationale["status"] == "ok":
            parts.append(
                f"CRISIL rating rationale ({rationale.get('rating_date', 'undated')}, "
                f"source: {rationale['document_url']}):\n\n{rationale['text']}"
            )

    return "\n\n---\n\n".join(parts) if parts else "No prior context available from other specialists."


def _compute_deterministic_stats(assessment: BusinessLLMAssessment) -> dict[str, Any]:
    """Independently tallies grounded/inferred/unavailable across the 4
    dimensions — never trusts the LLM's own completeness claim."""
    grounded = inferred = unavailable = 0
    for dim in _BUSINESS_DIMENSIONS:
        finding = getattr(assessment, dim)
        g = (finding.grounding or "").strip().lower()
        if g == "grounded":
            grounded += 1
        elif g == "inferred":
            inferred += 1
        else:
            unavailable += 1

    return {
        "grounded_count": grounded,
        "inferred_count": inferred,
        "unavailable_count": unavailable,
        "completeness_ratio": round(grounded / len(_BUSINESS_DIMENSIONS), 3),
    }


def run_business_agent(state: dict[str, Any]) -> dict[str, Any]:
    entity_name = state.get("company_name") or state.get("company_reference", "")

    captured: list[dict[str, Any]] = []
    tools = [capturing(fn, captured) for fn in _BASE_TOOLS]

    agent = create_agent(
        model=build_model(),
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        response_format=BusinessLLMAssessment,
    )

    context = _build_context(state)
    user_msg = f"Entity: {entity_name}\n\nContext from other specialists:\n\n{context}"

    issues: list[dict[str, Any]] = []
    try:
        agent_result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
        assessment: BusinessLLMAssessment = agent_result["structured_response"]
    except Exception as exc:
        logger.warning("Business agent failed for %r: %s", entity_name, exc)
        issues.append({
            "stage": "business_agent",
            "message": f"Business agent run failed: {exc}",
            "severity": "error",
        })
        from credit_report.schemas import BusinessDimensionFinding
        unavailable = BusinessDimensionFinding(assessment="unavailable", grounding="unavailable")
        assessment = BusinessLLMAssessment(
            raw_score=50,
            summary="Business & Management assessment could not be completed this run due to an agent error.",
            promoter_management_pedigree=unavailable,
            business_model_moat=unavailable,
            operating_trajectory=unavailable,
            industry_macro_headwinds=unavailable,
        )

    stats = _compute_deterministic_stats(assessment)

    result = BusinessResult(
        entity_screened=entity_name,
        raw_score=assessment.raw_score,
        summary=assessment.summary,
        promoter_management_pedigree=assessment.promoter_management_pedigree.model_dump(),
        business_model_moat=assessment.business_model_moat.model_dump(),
        operating_trajectory=assessment.operating_trajectory.model_dump(),
        industry_macro_headwinds=assessment.industry_macro_headwinds.model_dump(),
        **stats,
    )

    return {"business_result": result.model_dump(), "issues": issues}
