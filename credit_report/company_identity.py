"""
Deterministic Company Identity Loader + Classification.

Per the architecture doc: in production this is a direct DB read (the
analyst already selected company_id + report type from Finoscale's DB —
no free-text resolution, no candidate list, no pause). There is no DB yet
during this open-data testing phase, so this wraps the existing entity
resolver instead — but the no-human-in-the-loop rule still holds: an
ambiguous reference auto-selects the highest-confidence candidate rather
than pausing, and the discarded alternatives are recorded as a DataIssue,
not silently dropped. A company with no resolvable ticker is not a
failure — it's routed to the Finance agent's no-ticker branch (private/
unlisted company) in a later phase.
"""
from __future__ import annotations

import logging
from typing import Any

from tools.bank_classifier import classify_bank
from tools.ticker_resolver import resolve_entity

logger = logging.getLogger(__name__)


def load_company_identity(state: dict[str, Any]) -> dict[str, Any]:
    query = state.get("company_reference") or state.get("query", "")
    candidates = resolve_entity(query)

    if not candidates:
        logger.info("No ticker resolved for %r — treating as private/unlisted.", query)
        return {
            "company_name": query,
            "ticker": None,
            "has_ticker": False,
            "resolution_confidence": None,
            "issues": [{
                "stage": "company_identity",
                "message": f"No exchange ticker resolved for {query!r} — treating as a private/unlisted company.",
                "severity": "info",
            }],
        }

    best = max(candidates, key=lambda c: c.get("confidence", 0.0))
    issues: list[dict] = []
    if len(candidates) > 1:
        discarded = [c["ticker"] for c in candidates if c is not best]
        issues.append({
            "stage": "company_identity",
            "message": (
                f"{len(candidates)} candidates resolved for {query!r}; auto-selected "
                f"{best['ticker']} (confidence {best.get('confidence', 0):.2f}) — no human "
                f"pause, per the no-human-in-the-loop design. Discarded: {discarded}."
            ),
            "severity": "warning",
        })

    return {
        "company_name": best.get("name") or query,
        "ticker": best.get("ticker"),
        "has_ticker": True,
        "resolution_confidence": best.get("confidence"),
        "issues": issues,
    }


def classify_company(state: dict[str, Any]) -> dict[str, Any]:
    entity_name = state.get("company_name") or state.get("company_reference", "")
    result = classify_bank(entity_name)

    issues: list[dict] = []
    if result.confidence == "unavailable":
        issues.append({
            "stage": "classify_company",
            "message": result.note or "RBI bank classification source unavailable this run.",
            "severity": "warning",
        })

    return {
        "is_bank": result.is_bank,
        "bank_category": result.bank_category,
        "issues": issues,
    }
