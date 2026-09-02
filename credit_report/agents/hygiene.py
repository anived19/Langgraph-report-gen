"""
Hygiene / AML specialist subgraph.

The LLM agent decides when/whether to call the AML tools (real agentic tool
choice — this is what makes it an "agent" rather than a wrapped deterministic
function, per the architecture doc's "LLM decides within a bounded node"
principle) and produces only a qualitative severity judgment (raw_score +
summary). It never counts or tallies findings itself.

Each tool is wrapped in a capture closure so the same values the agent's own
tool calls returned are also available here for deterministic post-processing
(findings counts, completeness ratio) — this avoids both re-parsing serialized
tool-message strings and re-calling the (network-hitting, costed) tools a
second time just to recover their outputs.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent

from credit_report.model import build_model
from credit_report.prompt_loader import load_prompt
from credit_report.schemas import HygieneLLMAssessment, HygieneResult
from credit_report.tool_capture import capturing
from schemas import AMLSeverity
from tools.aml_tools import run_structured_aml_sweep, search_adverse_media

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("hygiene")

# Named so tests can assert on it directly (tool-isolation check) rather than
# inferring it from create_agent's internals — this agent gets exactly these
# two tools and nothing from Finance/Business/Banking.
_BASE_TOOLS = (run_structured_aml_sweep, search_adverse_media)

_UNAVAILABLE_MARKERS = (
    "unavailable", "could not fetch", "could not be fetched",
    "manual check", "manual review", "manual verification",
    "not reachable", "unscreened",
)
# Known limitation, not fixable here: OFAC/World Bank/UN/OpenSanctions/SEC
# EDGAR deliberately fail closed to the SAME text as a genuine clean result
# (see tools/aml_tools.py and tests/test_aml_tools.py::
# test_screener_clean_status_on_failure — an earlier, intentional design
# choice this session, to avoid false Watch flags on trivial transient
# errors). Only sources that explicitly say so on failure (currently just
# EU, via _EU_XML_URL's honest-failure fix) are detectable here — this
# completeness_ratio is a conservative floor, not a complete signal.


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

    # Findings from unreachable sources surface as a Watch-severity finding
    # whose summary text says so (see tools/aml_tools.py's fail-closed
    # behavior on fetch failure) — there's no separate "unreachable" flag on
    # AMLFinding, so this is a text-based proxy; revisit if that changes.
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
    }


def run_hygiene_agent(state: dict[str, Any]) -> dict[str, Any]:
    entity_name = state.get("company_name") or state.get("company_reference", "")
    ticker = state.get("ticker") or ""

    captured: list[dict[str, Any]] = []
    tools = [capturing(fn, captured) for fn in _BASE_TOOLS]

    agent = create_agent(
        model=build_model(),
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        response_format=HygieneLLMAssessment,
    )

    user_msg = f"Entity: {entity_name}" + (
        f" (ticker: {ticker})" if ticker else " (no ticker — private/unlisted company)"
    )

    issues: list[dict[str, Any]] = []
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
        assessment: HygieneLLMAssessment = result["structured_response"]
    except Exception as exc:
        logger.warning("Hygiene agent failed for %r: %s", entity_name, exc)
        issues.append({
            "stage": "hygiene_agent",
            "message": f"Hygiene agent run failed: {exc}",
            "severity": "error",
        })
        assessment = HygieneLLMAssessment(
            raw_score=50,
            summary="Hygiene assessment could not be completed this run due to an agent error.",
        )

    stats = _compute_deterministic_stats(captured)
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

    return {"hygiene_result": hygiene_result.model_dump(), "issues": issues}
