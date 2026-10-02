import pytest

from config import settings
from credit_report import validation as v
from credit_report.schemas import BusinessDimensionFinding, BusinessLLMAssessment, HygieneLLMAssessment

EVIDENCE = [
    '{"pe_ratio": 24.53, "roe": 0.4774, "market_cap": 1.2345e13, "market_cap_formatted": "Rs. 12.35 Lakh Cr",'
    ' "debt_to_equity": 10.21, "current_price": 3456.78, "rsi_14": 61.2}',
]


@pytest.mark.parametrize("text", [
    "P/E of 24.53 and ROE of 47.74%.",
    "P/E of 24.5x, ROE ~47.7%, market cap Rs. 12.35 Lakh Cr.",
    "Trades at 3,456.78, around 3,457; RSI-14 at 61.2 over the 50-day and 200-day averages.",
    "Debt/equity 10.21 in FY2025 (Q1 FY25), 6-month outlook, 3 agents.",
    "Score 72/100 reflects this.",
    "Held back from the 90+ range; a 30+ year veteran.",
])
def test_grounded_text_passes(text):
    assert v.ungrounded_numbers(text, EVIDENCE) == []


def test_invented_figures_are_caught():
    assert v.ungrounded_numbers("P/E of 31.2 and revenue growth of 18.6%.", EVIDENCE) == ["31.2", "18.6"]


def test_check_dimensions():
    ok = BusinessDimensionFinding(assessment="x", grounding="grounded", source="CRISIL")
    a = BusinessLLMAssessment(
        raw_score=60, summary="s" * 50,
        promoter_management_pedigree=ok,
        business_model_moat=BusinessDimensionFinding(assessment="y", grounding="grounded"),
        operating_trajectory=BusinessDimensionFinding(assessment="", grounding="inferred"),
        industry_macro_headwinds=BusinessDimensionFinding(assessment="z", grounding="maybe"),
    )
    failures = v.check_dimensions(a, ["promoter_management_pedigree", "business_model_moat",
                                      "operating_trajectory", "industry_macro_headwinds"])
    assert len(failures) == 3


def _attempt(summary=None, error=None):
    return v.Attempt(HygieneLLMAssessment(raw_score=70, summary=summary) if summary is not None else None,
                     [], [], error)


def test_rerun_with_feedback_then_pass(monkeypatch):
    monkeypatch.setattr(settings, "validation_max_retries", 1)
    feedbacks = []
    attempts = iter([_attempt("too short"), _attempt("A specific, long enough summary of the screening results.")])

    def run_once(feedback):
        feedbacks.append(feedback)
        return next(attempts)

    attempt, issues, traces = v.run_with_validation("hygiene", "X", run_once, lambda a: v.check_summary(a.assessment.summary))
    assert feedbacks[0] is None and "REJECTED" in feedbacks[1] and "summary" in feedbacks[1]
    assert attempt.assessment.summary.startswith("A specific")
    assert len(traces) == 2 and traces[0]["validation_failures"] and not traces[1]["validation_failures"]
    assert issues[0]["severity"] == "info"


def test_retry_cap_keeps_best_attempt_and_logs(monkeypatch):
    monkeypatch.setattr(settings, "validation_max_retries", 5)  # clamped to 2
    calls = []

    def run_once(feedback):
        calls.append(feedback)
        return _attempt("short") if len(calls) == 2 else _attempt(None, RuntimeError("boom"))

    attempt, issues, traces = v.run_with_validation("hygiene", "X", run_once, lambda a: v.check_summary(a.assessment.summary))
    assert len(calls) == 3
    assert attempt.assessment.summary == "short"  # an answer with one failure beats an agent error
    assert issues[-1]["severity"] == "warning" and "3 attempt" in issues[-1]["message"]


def test_rate_limit_is_not_rerun(monkeypatch):
    monkeypatch.setattr(settings, "validation_max_retries", 2)

    class RateLimitError(Exception):
        pass

    calls = []

    def run_once(feedback):
        calls.append(feedback)
        return _attempt(None, RateLimitError("429"))

    attempt, issues, _ = v.run_with_validation("finance", "X", run_once, lambda a: [])
    assert len(calls) == 1 and issues[-1]["severity"] == "error"
