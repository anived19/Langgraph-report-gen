"""
Business & Management specialist subgraph.

Unlike Finance/Hygiene, this agent doesn't fetch its own primary data — it
reasons over what Finance and Hygiene already found (handed to it as
context, not re-fetched) plus the rating rationale text when the entity has
no ticker (reusing tools/rating_agency_tools.py, which caches, so this
doesn't re-hit CRISIL for data Finance already pulled), with search_web_news
as its only real tool for anything not already covered. When a CIN was
supplied, the board and KMP from MCA records (Directors node) are handed in
too — names, roles, dates and counts only, never DIN/PAN or PEP results.

The LLM must tag each of the 4 dimensions grounded/inferred/unavailable
rather than write one blended paragraph — completeness is then counted
deterministically from those tags (grounded_count / 4), never trusted from
the LLM's own claim, same discipline as Finance and Hygiene.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from credit_report.agents.directors import board_context
from credit_report.model import build_agent
from credit_report.prompt_loader import load_prompt
from credit_report.schemas import BusinessDimensionFinding, BusinessLLMAssessment, BusinessResult, _BUSINESS_DIMENSIONS
from credit_report.tool_capture import capturing
from credit_report.validation import (
    Attempt,
    check_dimensions,
    check_numbers_grounded,
    check_summary,
    evidence_from,
    run_with_validation,
)
from tools.rating_agency_tools import get_rating_rationale
from tools.search_tools import search_web_news

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("business")

# This agent gets exactly search_web_news; rating-rationale fetching happens
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

    board = board_context(state.get("directors_result"))
    if board:
        parts.append(board)

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


def validate_dimensions(attempt: Attempt, evidence_texts: list[str]) -> list[str]:
    """Shared with the Banking agent: summary present, all four dimensions
    filled and validly tagged, and every figure traceable to the context or
    a search result."""
    a = attempt.assessment
    fields = {"summary": a.summary}
    fields.update({f"{dim}.assessment": getattr(a, dim).assessment for dim in _BUSINESS_DIMENSIONS})
    return (check_summary(a.summary)
            + check_dimensions(a, _BUSINESS_DIMENSIONS)
            + check_numbers_grounded(fields, evidence_from(attempt.captured, *evidence_texts, str(a.raw_score))))


def run_business_agent(state: dict[str, Any]) -> dict[str, Any]:
    entity_name = state.get("company_name") or state.get("company_reference", "")

    context = _build_context(state)
    user_msg = f"Entity: {entity_name}\n\nContext from other specialists:\n\n{context}"

    def run_once(feedback: Optional[str]) -> Attempt:
        captured: list[dict[str, Any]] = []
        agent = build_agent([capturing(fn, captured) for fn in _BASE_TOOLS], _SYSTEM_PROMPT, BusinessLLMAssessment)
        content = user_msg + (f"\n\n{feedback}" if feedback else "")
        result: dict[str, Any] = {}
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": content}]})
            return Attempt(result["structured_response"], captured, result.get("messages", []))
        except Exception as exc:
            logger.warning("Business agent failed for %r: %s", entity_name, exc)
            return Attempt(None, captured, result.get("messages", []), exc)

    attempt, issues, traces = run_with_validation(
        "business", entity_name, run_once, lambda a: validate_dimensions(a, [user_msg, _SYSTEM_PROMPT]),
    )
    assessment = attempt.assessment
    if assessment is None:
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

    return {"business_result": result.model_dump(), "issues": issues, "traces": traces}
