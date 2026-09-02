"""
Report Synthesizer — deterministic, no LLM call happens at this stage.
Every field in the assembled CreditReport is either copied straight from
graph state or computed by a fixed rule (the score-threshold recommendation
below). This mirrors the equity pipeline's own render_aml_markdown() /
compile_pdf() split: agents reason, this stage only assembles and formats.
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
        recommendation=recommendation,
        recommendation_note=recommendation_note,
        issues=state.get("issues") or [],
    )

    return {"report": report.model_dump(mode="json")}
