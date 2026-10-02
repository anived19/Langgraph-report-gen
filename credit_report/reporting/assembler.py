"""
Report Synthesizer — deterministic, no LLM call happens at this stage.
Every field in the assembled CreditReport is either copied straight from
graph state or computed by a fixed rule (the score-threshold recommendation
below). Agents reason; this stage only assembles and formats.

`analyst_review` gathers every possible namesake — exact-name watchlist
records with no identifier (Hygiene) and exact-name PEP records for
directors — into one list for the analyst. None of it is scored.
"""
from __future__ import annotations

from typing import Any

from credit_report.schemas import CreditReport, Scores

# Fixed, auditable thresholds — not a model's judgment call. Checked in
# order; first match wins.
_RECOMMENDATION_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (70.0, "Recommended", "strong overall assessment across evaluated dimensions"),
    (50.0, "Recommended with Conditions", "moderate assessment — review flagged concerns and data-completeness gaps before proceeding"),
)


def _recommendation_for_score(final_score: float) -> tuple[str, str]:
    for threshold, label, note in _RECOMMENDATION_THRESHOLDS:
        if final_score >= threshold:
            return label, f"Final score {final_score}/100 — {note}."
    return (
        "Not Recommended",
        f"Final score {final_score}/100 — weak overall assessment; material concerns and/or insufficient verifiable data.",
    )


def _analyst_review(state: dict[str, Any]) -> list[str]:
    items = list((state.get("hygiene_result") or {}).get("analyst_review") or [])
    for d in (state.get("directors_result") or {}).get("current", []):
        pep = d.get("pep") or {}
        if pep.get("status") == "analyst_review":
            items.extend(f"PEP — director {d['name']}: {m} — analyst to confirm it is the same person; not scored."
                         for m in pep.get("matches", []))
    return list(dict.fromkeys(items))


def assemble_credit_report(state: dict[str, Any]) -> dict[str, Any]:
    """
    Parent graph node: builds the final CreditReport from every specialist's
    result plus the Score Engine's output. Always produces a report — a
    thin/incomplete one if data was sparse, never a failure.
    """
    scores_raw = state.get("scores") or {}
    scores = Scores.model_validate(scores_raw)
    recommendation, recommendation_note = _recommendation_for_score(scores.final_score)

    report = CreditReport(
        company_name=state.get("company_name") or state.get("company_reference", "Unknown"),
        company_reference=state.get("company_reference", ""),
        ticker=state.get("ticker"),
        has_ticker=bool(state.get("has_ticker")),
        is_bank=bool(state.get("is_bank")),
        bank_category=state.get("bank_category"),
        scores=scores,
        finance_result=state.get("finance_result") or {},
        hygiene_result=state.get("hygiene_result") or {},
        business_result=state.get("business_result") or {},
        banking_result=state.get("banking_result"),
        directors_result=state.get("directors_result"),
        analyst_review=_analyst_review(state),
        recommendation=recommendation,
        recommendation_note=recommendation_note,
        issues=state.get("issues") or [],
    )

    return {"report": report.model_dump(mode="json")}
