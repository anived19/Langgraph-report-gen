"""
Finance specialist subgraph — two branches:

- ticker: market and fundamental data from yfinance (tools/finance_tools.py).
- no ticker (private/unlisted): a CRISIL rating rationale if one exists,
  otherwise search_web_news.

Same pattern as the Hygiene agent (credit_report/agents/hygiene.py): the LLM
only produces a qualitative judgment (raw_score + summary); every tool call
is captured via a closure so completeness is computed independently in plain
Python, reusing tools/finance_tools.py:assemble_market_metrics()'s existing
field-availability check rather than inventing a second definition of
"expected data". Each branch's output is checked in code
(credit_report/validation.py) — required tools called, summary present,
every figure traceable to a tool result or the rationale text — and the
agent alone is rerun with the failures if it doesn't pass.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from credit_report.model import build_agent
from credit_report.prompt_loader import load_prompt
from credit_report.schemas import FinanceLLMAssessment, FinanceResult
from credit_report.tool_capture import capturing
from credit_report.validation import (
    Attempt,
    check_numbers_grounded,
    check_summary,
    check_tools_called,
    evidence_from,
    run_with_validation,
)
from tools.finance_tools import (
    _MARKET_METRICS_FIELD_KEYS,
    assemble_market_metrics,
    compute_custom_financial_metric,
    get_fundamentals,
    get_ownership,
    get_price_snapshot,
    get_quarterly_financials,
    get_technicals,
    get_valuation_multiples,
)
from tools.rating_agency_tools import get_rating_rationale
from tools.search_tools import search_web_news

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("finance")
_NO_TICKER_SYSTEM_PROMPT = load_prompt("finance_no_ticker")

# No-ticker completeness is a coarse 3-tier heuristic, not a field count like
# the ticker branch's 32-field checklist — a rating rationale is unstructured
# prose, not discrete numeric fields, so there's no equivalent thing to count.
# Deliberately conservative: even a full rationale document caps below the
# ticker branch's achievable 1.0, since it's still a single secondary source
# rather than 32 independently-verifiable data points.
_COMPLETENESS_RATING_RATIONALE = 0.75
_COMPLETENESS_WEB_SEARCH_ONLY = 0.30
_COMPLETENESS_NO_DATA = 0.0

# No-ticker branch gets only search_web_news — rating-rationale fetching is
# deterministic (called directly below, not agent-invoked), so it isn't a
# tool the LLM chooses to call.
_NO_TICKER_TOOLS = (search_web_news,)

# This agent gets exactly these tools and nothing from Hygiene/Business/Banking.
_BASE_TOOLS = (
    get_price_snapshot,
    get_valuation_multiples,
    get_fundamentals,
    get_quarterly_financials,
    get_technicals,
    get_ownership,
    compute_custom_financial_metric,
)
# The six data tools the finance prompt tells the agent to always call.
_REQUIRED_TICKER_TOOLS = [fn.__name__ for fn in _BASE_TOOLS if fn is not compute_custom_financial_metric]

# assemble_market_metrics() checks len(_MARKET_METRICS_FIELD_KEYS) fields plus
# one more (quarterly_financials, checked separately) — see tools/finance_tools.py.
_EXPECTED_METRIC_COUNT = len(_MARKET_METRICS_FIELD_KEYS) + 1


def _merge_ticker_tool_results(captured: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge the dict-returning granular fetchers into the flat shape
    assemble_market_metrics() expects. compute_custom_financial_metric's
    output is intentionally excluded here — it's a named custom metric, not
    part of the standard MarketMetrics field set this completeness check
    covers.
    """
    merged: dict[str, Any] = {}
    quarterly: list[dict] = []
    for call in captured:
        name, result = call["tool"], call.get("result")
        if name == "get_quarterly_financials" and isinstance(result, list):
            quarterly = result
        elif name != "compute_custom_financial_metric" and isinstance(result, dict):
            merged.update(result)
    if quarterly:
        merged["quarterly_financials"] = quarterly
    return merged


def _run_attempt(tools_fns, system_prompt: str, user_msg: str, entity_name: str):
    """Returns a run_once(feedback) closure for run_with_validation."""
    def run_once(feedback: Optional[str]) -> Attempt:
        captured: list[dict[str, Any]] = []
        agent = build_agent([capturing(fn, captured) for fn in tools_fns], system_prompt, FinanceLLMAssessment)
        content = user_msg + (f"\n\n{feedback}" if feedback else "")
        result: dict[str, Any] = {}
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": content}]})
            return Attempt(result["structured_response"], captured, result.get("messages", []))
        except Exception as exc:
            logger.warning("Finance agent failed for %r: %s", entity_name, exc)
            return Attempt(None, captured, result.get("messages", []), exc)
    return run_once


def _run_no_ticker_finance(entity_name: str) -> dict[str, Any]:
    """
    No-ticker (private/unlisted company) Finance path. CRISIL rating
    rationale is the primary source (see tools/rating_agency_tools.py for why
    GST/MCA scraping was deliberately not built); search_web_news is the
    fallback when no rationale exists. Never fails — an unrated, undocumented
    company still gets a cautious, honestly-labeled result.
    """
    issues: list[dict[str, Any]] = []
    rationale = get_rating_rationale(entity_name)

    if rationale["status"] == "ok":
        user_msg = (
            f"Entity: {entity_name} (no ticker — private/unlisted company)\n\n"
            f"CRISIL rating rationale ({rationale.get('rating_date', 'undated')}, "
            f"source: {rationale['document_url']}):\n\n{rationale['text']}"
        )
    else:
        user_msg = (
            f"Entity: {entity_name} (no ticker — private/unlisted company). "
            "No CRISIL rating rationale was found for this entity. Use search_web_news "
            "if useful, per your instructions."
        )
        issues.append({
            "stage": "finance_agent",
            "message": f"No CRISIL rating rationale found for {entity_name!r} (status: {rationale['status']}).",
            "severity": "info",
        })

    def validate(attempt: Attempt) -> list[str]:
        a = attempt.assessment
        return (check_summary(a.summary)
                + check_numbers_grounded({"summary": a.summary},
                                         evidence_from(attempt.captured, user_msg, _NO_TICKER_SYSTEM_PROMPT, str(a.raw_score))))

    attempt, run_issues, traces = run_with_validation(
        "finance", entity_name,
        _run_attempt(_NO_TICKER_TOOLS, _NO_TICKER_SYSTEM_PROMPT, user_msg, entity_name), validate,
    )
    issues.extend(run_issues)
    assessment = attempt.assessment or FinanceLLMAssessment(
        raw_score=50,
        summary="Finance assessment could not be completed this run due to an agent error.",
    )

    # Tavily near-always returns *some* results for any query, even
    # irrelevant ones — a non-empty list isn't a meaningful "found
    # something" signal. Require at least one result with actual content
    # text, not just a non-empty results list.
    search_results_found = any(
        (r.get("content") or "").strip()
        for call in attempt.captured
        for r in (call.get("result") or [])
    )

    if rationale["status"] == "ok":
        data_path = "no_ticker_rating_rationale"
        completeness_ratio = _COMPLETENESS_RATING_RATIONALE
        source_name, source_url = rationale["source"], rationale["document_url"]
    elif search_results_found:
        data_path = "no_ticker_web_search_only"
        completeness_ratio = _COMPLETENESS_WEB_SEARCH_ONLY
        source_name, source_url = "Web search (Tavily)", None
    else:
        data_path = "no_ticker_no_data"
        completeness_ratio = _COMPLETENESS_NO_DATA
        source_name, source_url = None, None
        issues.append({
            "stage": "finance_agent",
            "message": f"No rating rationale and no usable web search results for {entity_name!r} — Finance assessment rests on no verifiable data.",
            "severity": "warning",
        })

    result = FinanceResult(
        entity_screened=entity_name,
        ticker=None,
        data_path=data_path,
        raw_score=assessment.raw_score,
        summary=assessment.summary,
        # Forced None regardless of what the model returned — there's no
        # price/analyst/technical data for an unlisted company, so any
        # six_month_outlook here would be fabricated. Don't trust prompt
        # compliance for something structurally impossible to ground.
        six_month_outlook=None,
        metrics_fetched_count=0,
        metrics_expected_count=0,
        completeness_ratio=completeness_ratio,
        source_name=source_name,
        source_url=source_url,
    )
    return {"finance_result": result.model_dump(), "issues": issues, "traces": traces}


def run_finance_agent(state: dict[str, Any]) -> dict[str, Any]:
    entity_name = state.get("company_name") or state.get("company_reference", "")
    ticker = state.get("ticker")

    if not state.get("has_ticker") or not ticker:
        return _run_no_ticker_finance(entity_name)

    user_msg = f"Entity: {entity_name} (ticker: {ticker})"

    def validate(attempt: Attempt) -> list[str]:
        a = attempt.assessment
        failures = check_summary(a.summary) + check_tools_called(attempt.captured, _REQUIRED_TICKER_TOOLS)
        failures += check_summary(a.six_month_outlook, "six_month_outlook")
        failures += check_numbers_grounded(
            {"summary": a.summary, "six_month_outlook": a.six_month_outlook},
            evidence_from(attempt.captured, user_msg, _SYSTEM_PROMPT, str(a.raw_score)),
        )
        return failures

    attempt, issues, traces = run_with_validation(
        "finance", entity_name, _run_attempt(_BASE_TOOLS, _SYSTEM_PROMPT, user_msg, entity_name), validate,
    )
    assessment = attempt.assessment or FinanceLLMAssessment(
        raw_score=50,
        summary="Finance assessment could not be completed this run due to an agent error.",
    )

    merged = _merge_ticker_tool_results(attempt.captured)
    metrics = assemble_market_metrics(ticker, merged)
    fetched_count = _EXPECTED_METRIC_COUNT - len(metrics.unavailable_fields)
    completeness_ratio = fetched_count / _EXPECTED_METRIC_COUNT if _EXPECTED_METRIC_COUNT else 0.0

    if not attempt.captured:
        issues.append({
            "stage": "finance_agent",
            "message": "No Finance tools were actually called this run (agent may not have used them).",
            "severity": "warning",
        })

    finance_result = FinanceResult(
        entity_screened=entity_name,
        ticker=ticker,
        data_path="ticker",
        raw_score=assessment.raw_score,
        summary=assessment.summary,
        six_month_outlook=assessment.six_month_outlook,
        metrics_fetched_count=fetched_count,
        metrics_expected_count=_EXPECTED_METRIC_COUNT,
        completeness_ratio=round(completeness_ratio, 3),
        market_metrics=metrics.model_dump(mode="json"),
    )

    return {"finance_result": finance_result.model_dump(), "issues": issues, "traces": traces}
