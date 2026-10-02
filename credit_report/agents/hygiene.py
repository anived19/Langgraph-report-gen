"""
Hygiene / AML specialist subgraph.

The LLM agent decides when/whether to call the AML tools (real agentic tool
choice — this is what makes it an "agent" rather than a wrapped deterministic
function, per the architecture doc's "LLM decides within a bounded node"
principle) and produces only a qualitative severity judgment (raw_score +
summary). It never counts or tallies findings itself.

Each tool is wrapped so the same values the agent's own tool calls returned
are also captured here for deterministic post-processing (findings counts,
completeness ratio) — this avoids both re-parsing serialized tool-message
strings and re-calling the (network-hitting, costed) tools a second time.

Exact-entity screening: the entity's identifiers (CIN / PAN, from the
Directors node) are bound into the sweep in code — the LLM never sees or
passes them. Same-name records with no identifier (`analyst_review`) are
stripped from what the LLM sees, so they can't move the score, and are
carried to the report for the analyst instead.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from credit_report.model import build_agent
from credit_report.prompt_loader import load_prompt
from credit_report.schemas import HygieneLLMAssessment, HygieneResult
from credit_report.validation import Attempt, check_summary, check_tools_called, run_with_validation
from schemas import AMLSeverity
from tools.aml_tools import run_structured_aml_sweep, search_adverse_media

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("hygiene")

_UNAVAILABLE_MARKERS = (
    "unavailable", "could not fetch", "could not be fetched",
    "not reachable", "unscreened",
)
# Every screener in tools/aml_tools.py reports an unreachable source as
# "unscreened" (Watch), never with the same text as a clean result, so this
# text check sees every source that didn't run.


def _compute_deterministic_stats(captured: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently tallies findings from the raw tool results — never
    trusts the LLM's own arithmetic, matching the codebase-wide rule that
    no LLM computes a number that downstream logic relies on."""
    all_findings: list[dict[str, Any]] = []
    for call in captured:
        all_findings.extend(call.get("result") or [])

    counts = {sev.value: 0 for sev in AMLSeverity}
    for f in all_findings:
        sev = f.get("severity", AMLSeverity.NONE.value)
        counts[sev] = counts.get(sev, 0) + 1

    sources_unavailable = sum(
        1 for f in all_findings
        if any(m in (f.get("finding_summary") or "").lower() for m in _UNAVAILABLE_MARKERS)
    )

    sources_checked = len(all_findings)
    completeness_ratio = (
        (sources_checked - sources_unavailable) / sources_checked if sources_checked else 0.0
    )

    return {
        "findings_count": sources_checked,
        "high_severity_count": counts.get(AMLSeverity.HIGH.value, 0),
        "elevated_severity_count": counts.get(AMLSeverity.ELEVATED.value, 0),
        "watch_severity_count": counts.get(AMLSeverity.WATCH.value, 0),
        "none_severity_count": counts.get(AMLSeverity.NONE.value, 0),
        "sources_checked": sources_checked,
        "sources_unavailable": sources_unavailable,
        "completeness_ratio": round(completeness_ratio, 3),
        "analyst_review": list(dict.fromkeys(n for f in all_findings for n in (f.get("analyst_review") or []))),
    }


def _scoring_view(fn, sink: list[dict[str, Any]]):
    """Like tool_capture.capturing(), but the LLM gets each finding without
    its `analyst_review` notes (unconfirmed namesakes must not move the
    score), while `sink` keeps the full result."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        sink.append({"tool": fn.__name__, "args": args, "kwargs": kwargs, "result": result})
        return [{k: v for k, v in f.items() if k != "analyst_review"} for f in result]
    return wrapper


def _sweep_tool(entity_name: str, ticker: str, entity_ids: list[str]):
    """run_structured_aml_sweep with everything bound in code: exact-entity
    matching only means something if the name screened is the registry
    legal name, not whatever string the model chose to pass. The tool the
    LLM sees takes no arguments."""
    def run_structured_aml_sweep_tool() -> list[dict[str, Any]]:
        return run_structured_aml_sweep(entity_name, ticker, entity_ids=entity_ids)
    run_structured_aml_sweep_tool.__name__ = "run_structured_aml_sweep"
    run_structured_aml_sweep_tool.__doc__ = (
        "Deterministic sweep of the entity under review across OFAC SDN, OpenSanctions, World Bank "
        "debarment, UN, EU and SEC EDGAR FCPA, plus jurisdictional risk (TI CPI, FATF). Takes no "
        "arguments: the entity's name and identifiers are fixed by the pipeline."
    )
    return run_structured_aml_sweep_tool


def _validate(attempt: Attempt) -> list[str]:
    return (check_summary(attempt.assessment.summary)
            + check_tools_called(attempt.captured, ["run_structured_aml_sweep"]))


def run_hygiene_agent(state: dict[str, Any]) -> dict[str, Any]:
    # The registry legal name (from --cin) is what watchlists list; the
    # resolver's display name is the fallback.
    entity_name = state.get("legal_name") or state.get("company_name") or state.get("company_reference", "")
    ticker = state.get("ticker") or ""
    entity_ids = list(state.get("entity_ids") or [])

    user_msg = f"Entity: {entity_name}" + (
        f" (ticker: {ticker})" if ticker else " (no ticker — private/unlisted company)"
    )

    def run_once(feedback: Optional[str]) -> Attempt:
        captured: list[dict[str, Any]] = []
        tools = [
            _scoring_view(_sweep_tool(entity_name, ticker, entity_ids), captured),
            _scoring_view(search_adverse_media, captured),
        ]
        agent = build_agent(tools, _SYSTEM_PROMPT, HygieneLLMAssessment)
        content = user_msg + (f"\n\n{feedback}" if feedback else "")
        result: dict[str, Any] = {}
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": content}]})
            return Attempt(result["structured_response"], captured, result.get("messages", []))
        except Exception as exc:
            logger.warning("Hygiene agent failed for %r: %s", entity_name, exc)
            return Attempt(None, captured, result.get("messages", []), exc)

    attempt, issues, traces = run_with_validation("hygiene", entity_name, run_once, _validate)
    assessment = attempt.assessment or HygieneLLMAssessment(
        raw_score=50,
        summary="Hygiene assessment could not be completed this run due to an agent error.",
    )

    stats = _compute_deterministic_stats(attempt.captured)
    if stats["sources_checked"] == 0:
        issues.append({
            "stage": "hygiene_agent",
            "message": "No AML sources were actually checked this run (agent may not have called its tools).",
            "severity": "warning",
        })

    hygiene_result = HygieneResult(
        entity_screened=entity_name,
        raw_score=assessment.raw_score,
        summary=assessment.summary,
        **stats,
    )

    return {"hygiene_result": hygiene_result.model_dump(), "issues": issues, "traces": traces}
