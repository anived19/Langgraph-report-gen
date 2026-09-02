"""
Score Engine — fully deterministic, no LLM involved anywhere in this file.

Per the architecture doc's Section 11.1: final_score = average of finance,
hygiene, business_management (+ banking, if the entity is a bank). The
LLM-decides-within-a-bounded-node principle stops at each agent's own
raw_score; aggregating those into a final number is arithmetic, not
judgment, and arithmetic belongs in code, not a model call — this is what
keeps the final number reproducible run to run.

Confidence shrinkage (agreed this session): a raw_score built on thin
evidence shouldn't carry the same weight as one built on a rich, mostly-
grounded assessment. Each agent already computes its own completeness_ratio
independently in plain Python (see credit_report/agents/*.py — never
LLM-claimed); this module pulls raw_score toward a neutral 50 baseline by
(1 - completeness_ratio) before averaging:

    shrunk_score = 50 + (raw_score - 50) * completeness_ratio

A raw_score of 90 built on completeness 1.0 stays 90. The same 90 built on
completeness 0.1 shrinks to 54 — barely above neutral, reflecting that
there's almost nothing behind the number. This is a deliberate design
choice, not a hidden default: it means "no data" and "bad news" both pull
toward the same neutral midpoint from opposite directions, rather than a
sparse-but-lucky finding scoring as confidently as a well-evidenced one.
"""
from __future__ import annotations

import logging
from typing import Any

from credit_report.schemas import AgentScoreBreakdown, Scores

logger = logging.getLogger(__name__)

_NEUTRAL_BASELINE = 50.0


def _shrink(raw_score: int, completeness_ratio: float) -> float:
    shrunk = _NEUTRAL_BASELINE + (raw_score - _NEUTRAL_BASELINE) * completeness_ratio
    # raw_score in [0,100] and completeness_ratio in [0,1] already keep this
    # in range mathematically; clamped anyway rather than trusting that.
    return round(max(0.0, min(100.0, shrunk)), 2)


def _breakdown(agent_name: str, result: dict[str, Any]) -> AgentScoreBreakdown:
    raw_score = result.get("raw_score")
    completeness_ratio = result.get("completeness_ratio")
    if raw_score is None or completeness_ratio is None:
        logger.warning(
            "%s result missing raw_score/completeness_ratio (%r/%r) — defaulting to "
            "neutral, zero-completeness rather than fabricating a value.",
            agent_name, raw_score, completeness_ratio,
        )
        raw_score = raw_score if raw_score is not None else 50
        completeness_ratio = completeness_ratio if completeness_ratio is not None else 0.0

    return AgentScoreBreakdown(
        agent=agent_name,
        raw_score=raw_score,
        completeness_ratio=completeness_ratio,
        shrunk_score=_shrink(raw_score, completeness_ratio),
    )


def compute_scores(state: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic Score Engine node: pulls each specialist's raw_score
    toward neutral by its own completeness_ratio, then averages — 3-way for
    a non-bank, 4-way for a bank. Never invents a missing agent's score;
    surfaces a DataIssue instead if one is unexpectedly absent.
    """
    issues: list[dict[str, Any]] = []

    finance = state.get("finance_result") or {}
    hygiene = state.get("hygiene_result") or {}
    business = state.get("business_result") or {}

    for name, result in (("finance", finance), ("hygiene", hygiene), ("business", business)):
        if not result:
            issues.append({
                "stage": "score_engine",
                "message": f"{name}_result was missing when scoring ran — this should never happen given the graph's fan-out; check routing.",
                "severity": "error",
            })

    finance_bd = _breakdown("finance", finance)
    hygiene_bd = _breakdown("hygiene", hygiene)
    business_bd = _breakdown("business", business)
    shrunk_scores = [finance_bd.shrunk_score, hygiene_bd.shrunk_score, business_bd.shrunk_score]

    banking_bd = None
    if state.get("is_bank"):
        banking_result = state.get("banking_result")
        if banking_result:
            banking_bd = _breakdown("banking", banking_result)
            shrunk_scores.append(banking_bd.shrunk_score)
        else:
            issues.append({
                "stage": "score_engine",
                "message": "Entity is classified as a bank but banking_result is missing — scored as a 3-agent average instead of 4-agent. Check routing.",
                "severity": "error",
            })

    final_score = round(sum(shrunk_scores) / len(shrunk_scores), 2)

    scores = Scores(
        finance=finance_bd,
        hygiene=hygiene_bd,
        business=business_bd,
        banking=banking_bd,
        final_score=final_score,
        agents_averaged=len(shrunk_scores),
    )

    return {"scores": scores.model_dump(), "issues": issues}
