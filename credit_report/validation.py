"""
Code checks on each agent's output, and the targeted rerun loop.

Every check here is plain Python over the agent's structured answer and the
tool results it actually received — no extra model call, so it costs nothing
against the RPM budget. When an attempt fails, ONLY that agent is rerun, with
the specific failures written into its next prompt (a blind rerun tends to
repeat itself). Reruns are capped (settings.validation_max_retries, 0..2);
after that the best attempt is kept and the failures are logged as a data
issue on the report instead of looping.

Rate limiting is a separate loop: 429s are retried inside the HTTP client
(credit_report/model.py). A rate-limit error that still escapes is not
retried here — rerunning straight into the same quota would only fail again.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from config import settings
from credit_report.trace import build_agent_trace

logger = logging.getLogger(__name__)

MIN_SUMMARY_CHARS = 40
_GROUNDING_TAGS = {"grounded", "inferred", "unavailable"}


@dataclass
class Attempt:
    """One agent run: its structured answer (None if the run raised), the
    captured tool calls, the agent's message history, and any error."""
    assessment: Any
    captured: list[dict[str, Any]] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    error: Optional[BaseException] = None


def _is_rate_limit(exc: BaseException) -> bool:
    return type(exc).__name__ == "RateLimitError" or getattr(exc, "status_code", None) == 429


def run_with_validation(
    agent_name: str,
    entity_name: str,
    run_once: Callable[[Optional[str]], Attempt],
    validate: Callable[[Attempt], list[str]],
) -> tuple[Attempt, list[dict[str, Any]], list[dict[str, Any]]]:
    """Runs `run_once(feedback)` until `validate` passes or the retry budget
    is spent. Returns (best attempt, issues, traces — one per attempt)."""
    max_attempts = 1 + max(0, min(2, settings.validation_max_retries))
    issues: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    best: Optional[tuple[int, int, Attempt, list[str]]] = None  # (failures, -index, attempt, failures list)
    feedback: Optional[str] = None

    for i in range(max_attempts):
        attempt = run_once(feedback)
        trace = build_agent_trace(agent_name, entity_name, attempt.messages)
        trace["attempt"] = i + 1
        failures = [f"agent run failed: {attempt.error}"] if attempt.error else validate(attempt)
        trace["validation_failures"] = failures
        traces.append(trace)

        if not failures:
            if i:
                issues.append({"stage": f"{agent_name}_agent", "severity": "info",
                               "message": f"Passed output checks on attempt {i + 1} after a targeted rerun."})
            return attempt, issues, traces

        logger.warning("%s attempt %d/%d failed checks: %s", agent_name, i + 1, max_attempts, failures)
        rank = (len(failures) + (100 if attempt.assessment is None else 0), -i)
        if best is None or rank < best[:2]:
            best = (*rank, attempt, failures)
        if attempt.error is not None and _is_rate_limit(attempt.error):
            break
        feedback = (
            "YOUR PREVIOUS ATTEMPT WAS REJECTED by automated checks:\n"
            + "\n".join(f"- {f}" for f in failures)
            + "\nFix every point above in this attempt. Do not invent anything to satisfy a check — if a figure "
              "isn't in a tool result or the context you were given, leave it out. A figure you derived yourself "
              "(an upside %, a growth rate) counts as not given: compute it with a tool if you have one for that "
              "(e.g. compute_custom_financial_metric), otherwise describe it in words."
        )

    _, _, attempt, failures = best
    issues.append({
        "stage": f"{agent_name}_agent",
        "severity": "error" if attempt.assessment is None else "warning",
        "message": f"Output still failed checks after {len(traces)} attempt(s): " + "; ".join(failures),
    })
    return attempt, issues, traces


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_summary(text: Optional[str], label: str = "summary") -> list[str]:
    t = (text or "").strip()
    if len(t) < MIN_SUMMARY_CHARS:
        return [f"`{label}` is empty or too short ({len(t)} chars) — write a specific assessment."]
    return []


def check_tools_called(captured: list[dict[str, Any]], required: Iterable[str]) -> list[str]:
    called = {c.get("tool") for c in captured}
    missing = [t for t in required if t not in called]
    return [f"required tool(s) not called: {', '.join(missing)}."] if missing else []


def check_dimensions(assessment: Any, dims: Iterable[str]) -> list[str]:
    failures = []
    for dim in dims:
        finding = getattr(assessment, dim, None)
        if finding is None:
            failures.append(f"`{dim}` is missing.")
            continue
        grounding = (finding.grounding or "").strip().lower()
        if grounding not in _GROUNDING_TAGS:
            failures.append(f"`{dim}.grounding` is {finding.grounding!r}; it must be grounded, inferred or unavailable.")
        if not (finding.assessment or "").strip():
            failures.append(f"`{dim}.assessment` is empty — write 'unavailable' if there is nothing to say.")
        if grounding == "grounded" and not (finding.source or "").strip():
            failures.append(f"`{dim}` is tagged grounded but names no `source`.")
    return failures


# ---------------------------------------------------------------------------
# Numeric grounding: every figure in the prose must appear in the evidence
# ---------------------------------------------------------------------------

# A figure not glued to a preceding word ("FY25", "Q1", "rsi_14" are labels,
# not figures); a trailing unit ("24.5x", "3.2B", "18%") is fine.
_NUM_IN_TEXT = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d.,]\d)")
_NUM_IN_EVIDENCE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_UNIT_AFTER = re.compile(r"^\s*-?\s*(?:day|days|week|weeks|month|months|year|years|yr|yrs|quarter|quarters|"
                         r"dma|d\b|w\b|m\b|q\b|/100|out of 100|point scale|agents?|dimensions?)", re.I)
_SCALES = (1.0, 1e-2, 1e2, 1e3, 1e5, 1e6, 1e7, 1e9, 1e12)


def _numbers_in_evidence(evidence: Iterable[str]) -> set[float]:
    nums: set[float] = set()
    for text in evidence:
        for m in _NUM_IN_EVIDENCE.finditer(text or ""):
            try:
                nums.add(abs(float(m.group().replace(",", ""))))
            except ValueError:
                continue
    return nums


def _matches(value: float, decimals: int, evidence: set[float]) -> bool:
    tol = max(0.5 * 10 ** -decimals, 0.005 * value)
    for e in evidence:
        for scale in _SCALES:
            if abs(e / scale - value) <= tol + 1e-9:
                return True
    return False


def ungrounded_numbers(text: str, evidence: Iterable[str]) -> list[str]:
    """Figures in `text` that appear nowhere in `evidence` (tool results and
    the context the agent was given), allowing for rounding, percent/fraction
    and lakh/crore/million/billion scaling. Years, small counts (<= 12) and
    window lengths ("50-day", "6 months", "RSI-14") are not checked."""
    evidence_nums = _numbers_in_evidence(evidence)
    bad: list[str] = []
    for m in _NUM_IN_TEXT.finditer(text or ""):
        raw = m.group(1)
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if decimals == 0 and (value <= 12 or 1900 <= value <= 2100):
            continue
        after = (text or "")[m.end():m.end() + 14]
        before = (text or "")[max(0, m.start() - 4):m.start()].lower()
        if _UNIT_AFTER.match(after) or before.endswith(("rsi-", "rsi ", "rsi")):
            continue
        if value == 100 and before.endswith(("/", "of ")):  # the "/100" of a score
            continue
        if after.startswith("+"):  # an open threshold ("90+ range", "30+ years"), not a cited figure
            continue
        if not _matches(value, decimals, evidence_nums):
            bad.append(raw)
    return list(dict.fromkeys(bad))


def check_numbers_grounded(fields: dict[str, Optional[str]], evidence: Iterable[str]) -> list[str]:
    evidence = list(evidence)
    failures = []
    for label, text in fields.items():
        bad = ungrounded_numbers(text or "", evidence)
        if bad:
            failures.append(f"`{label}` cites figure(s) found in no tool result or provided context: "
                            f"{', '.join(bad[:8])} — use the exact figure a tool returned, or leave it out.")
    return failures


def evidence_from(captured: list[dict[str, Any]], *texts: Optional[str]) -> list[str]:
    """The texts an agent's numbers may come from: its tool results plus the
    prompt material it was handed (and its own raw_score, which the prose
    may quote)."""
    out = [json.dumps(c.get("result"), default=str) for c in captured]
    out.extend(t for t in texts if t)
    return out
