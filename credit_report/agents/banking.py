"""
Banking specialist subgraph — only invoked when the entity is classified as
a bank (tools/bank_classifier.py, gated in credit_report/routing.py).

Same shape as Business & Management: reasons over what Finance/Hygiene/
Business already found plus RBI peer-group benchmarks (deterministically
fetched, not agent-invoked — tools/banking_tools.py), with search_web_news
as its only real tool. Same grounded/inferred/unavailable tagging and
deterministic completeness counting as Business & Management — see
credit_report/agents/business.py for the pattern this mirrors.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from credit_report.agents.business import validate_dimensions
from credit_report.model import build_agent
from credit_report.prompt_loader import load_prompt
from credit_report.schemas import BankingLLMAssessment, BankingResult, BusinessDimensionFinding, _BUSINESS_DIMENSIONS
from credit_report.tool_capture import capturing
from credit_report.validation import Attempt, run_with_validation
from tools.banking_tools import get_banking_sector_benchmark
from tools.search_tools import search_web_news

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("banking")

# This agent gets exactly search_web_news; the RBI benchmark lookup happens
# deterministically below, not as an agent-invoked tool.
_BASE_TOOLS = (search_web_news,)


def _build_context(state: dict[str, Any], benchmark: dict[str, Any]) -> str:
    parts: list[str] = []

    finance = state.get("finance_result") or {}
    if finance.get("summary"):
        parts.append(f"Finance assessment (score {finance.get('raw_score')}/100): {finance['summary']}")

    hygiene = state.get("hygiene_result") or {}
    if hygiene.get("summary"):
        parts.append(f"Hygiene/AML assessment (score {hygiene.get('raw_score')}/100): {hygiene['summary']}")

    business = state.get("business_result") or {}
    if business.get("summary"):
        parts.append(f"Business & Management assessment (score {business.get('raw_score')}/100): {business['summary']}")

    if benchmark.get("status") == "ok":
        b = benchmark["benchmarks"]
        parts.append(
            f"RBI peer-group benchmark for {benchmark['bank_category']} as of {benchmark['as_of']} "
            f"(source: {benchmark['source']}, {benchmark['source_url']}) — PEER-GROUP AVERAGE, not this "
            f"bank's own figures: CRAR {b['crar_pct']}%, Tier 1 {b['tier1_capital_ratio_pct']}%, "
            f"Gross NPA {b['gross_npa_pct']}%, Net NPA {b['net_npa_pct']}%, "
            f"Leverage ratio {b['leverage_ratio_pct']}%, LCR {b['liquidity_coverage_ratio_pct']}%, "
            f"Spread {b['spread_pct']}%."
        )
    else:
        parts.append(f"No RBI peer-group benchmark available: {benchmark.get('reason', 'unknown category')}.")

    return "\n\n---\n\n".join(parts) if parts else "No prior context available from other specialists."


def _compute_deterministic_stats(assessment: BankingLLMAssessment) -> dict[str, Any]:
    """Independently tallies grounded/inferred/unavailable — never trusts
    the LLM's own completeness claim. Identical logic to Business &
    Management's (credit_report/agents/business.py), duplicated rather than
    shared because the two assessment types are distinct Pydantic models."""
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


def run_banking_agent(state: dict[str, Any]) -> dict[str, Any]:
    entity_name = state.get("company_name") or state.get("company_reference", "")
    bank_category = state.get("bank_category")

    benchmark = get_banking_sector_benchmark(bank_category)

    context = _build_context(state, benchmark)
    user_msg = f"Entity: {entity_name} (bank category: {bank_category or 'unknown'})\n\nContext:\n\n{context}"

    def run_once(feedback: Optional[str]) -> Attempt:
        captured: list[dict[str, Any]] = []
        agent = build_agent([capturing(fn, captured) for fn in _BASE_TOOLS], _SYSTEM_PROMPT, BankingLLMAssessment)
        content = user_msg + (f"\n\n{feedback}" if feedback else "")
        result: dict[str, Any] = {}
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": content}]})
            return Attempt(result["structured_response"], captured, result.get("messages", []))
        except Exception as exc:
            logger.warning("Banking agent failed for %r: %s", entity_name, exc)
            return Attempt(None, captured, result.get("messages", []), exc)

    attempt, issues, traces = run_with_validation(
        "banking", entity_name, run_once, lambda a: validate_dimensions(a, [user_msg, _SYSTEM_PROMPT]),
    )
    assessment = attempt.assessment
    if assessment is None:
        unavailable = BusinessDimensionFinding(assessment="unavailable", grounding="unavailable")
        assessment = BankingLLMAssessment(
            raw_score=50,
            summary="Banking assessment could not be completed this run due to an agent error.",
            promoter_management_pedigree=unavailable,
            business_model_moat=unavailable,
            operating_trajectory=unavailable,
            industry_macro_headwinds=unavailable,
        )

    if benchmark.get("status") != "ok":
        issues.append({
            "stage": "banking_agent",
            "message": benchmark.get("reason", "RBI peer-group benchmark unavailable for this bank category."),
            "severity": "info",
        })

    stats = _compute_deterministic_stats(assessment)

    result = BankingResult(
        entity_screened=entity_name,
        bank_category=bank_category,
        raw_score=assessment.raw_score,
        summary=assessment.summary,
        promoter_management_pedigree=assessment.promoter_management_pedigree.model_dump(),
        business_model_moat=assessment.business_model_moat.model_dump(),
        operating_trajectory=assessment.operating_trajectory.model_dump(),
        industry_macro_headwinds=assessment.industry_macro_headwinds.model_dump(),
        rbi_benchmark_available=benchmark.get("status") == "ok",
        rbi_benchmark=benchmark if benchmark.get("status") == "ok" else None,
        **stats,
    )

    return {"banking_result": result.model_dump(), "issues": issues, "traces": traces}
