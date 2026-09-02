"""
Pydantic schemas for the credit-report pipeline. Grows incrementally with
each build phase (see plan) — only what Phase 0 needs lives here for now.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field


class DataIssue(BaseModel):
    """A non-blocking note surfaced from any stage — missing data, an
    auto-resolved ambiguity, a source that couldn't be reached. Never
    causes the pipeline to fail; always shown in the final report rather
    than silently swallowed."""
    stage: str
    message: str
    severity: str = "info"  # "info" | "warning" | "error"


class CompanyIdentity(BaseModel):
    query: str
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    has_ticker: bool = False
    resolution_confidence: Optional[float] = None
    is_bank: bool = False
    bank_category: Optional[str] = None


# ---------------------------------------------------------------------------
# Hygiene / AML specialist (Phase 1)
# ---------------------------------------------------------------------------

class HygieneLLMAssessment(BaseModel):
    """
    The Hygiene agent's own structured output — qualitative judgment only.
    Deliberately excludes counts/tallies: those are computed deterministically
    downstream from the raw tool results (see credit_report/agents/hygiene.py),
    never trusted from the LLM's own arithmetic.
    """
    raw_score: int = Field(ge=0, le=100, description="100 = fully clean, 0 = confirmed severe adverse findings")
    summary: str


class HygieneResult(BaseModel):
    entity_screened: str
    raw_score: int = Field(ge=0, le=100)
    summary: str
    findings_count: int
    high_severity_count: int
    elevated_severity_count: int
    watch_severity_count: int
    none_severity_count: int
    sources_checked: int
    sources_unavailable: int
    completeness_ratio: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Finance specialist (Phase 2 — ticker branch; no-ticker branch in Phase 2b)
# ---------------------------------------------------------------------------

class FinanceLLMAssessment(BaseModel):
    """
    The Finance agent's own structured output — qualitative judgment of
    financial strength only. Field-level completeness is computed
    deterministically downstream (see credit_report/agents/finance.py),
    reusing tools/finance_tools.py:assemble_market_metrics()'s own
    availability check rather than a second definition of "expected fields".
    """
    raw_score: int = Field(ge=0, le=100, description="100 = exceptional financial strength, 0 = severe financial distress")
    summary: str
    six_month_outlook: Optional[str] = Field(
        default=None,
        description=(
            "Ticker branch only — a forward-looking view for the next ~6 months grounded in "
            "analyst price targets/consensus, technical momentum, and the price trend already "
            "fetched. None for the no-ticker branch — there's no price data to base one on."
        ),
    )


class FinanceResult(BaseModel):
    entity_screened: str
    ticker: Optional[str] = None
    data_path: str = Field(
        description=(
            "'ticker' | 'no_ticker_rating_rationale' | 'no_ticker_web_search_only' | "
            "'no_ticker_no_data' — which Finance data source actually backed this result"
        )
    )
    raw_score: int = Field(ge=0, le=100)
    summary: str
    six_month_outlook: Optional[str] = None
    metrics_fetched_count: int
    metrics_expected_count: int
    completeness_ratio: float = Field(ge=0.0, le=1.0)
    market_metrics: dict[str, Any] = Field(default_factory=dict)
    source_name: Optional[str] = Field(default=None, description="e.g. 'CRISIL Ratings' — for citation in the final report")
    source_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Business & Management specialist (Phase 3)
# ---------------------------------------------------------------------------

_BUSINESS_DIMENSIONS = (
    "promoter_management_pedigree",
    "business_model_moat",
    "operating_trajectory",
    "industry_macro_headwinds",
)


class BusinessDimensionFinding(BaseModel):
    assessment: str = Field(description="What was found/concluded for this dimension, or an explicit 'unavailable' note")
    grounding: str = Field(description="'grounded' (cites a specific source) | 'inferred' (LLM reasoning, no direct source) | 'unavailable'")
    source: Optional[str] = Field(default=None, description="What the finding is grounded in, when grounding='grounded'")


class BusinessLLMAssessment(BaseModel):
    """
    The Business & Management agent's own structured output. Deliberately
    forces each of the 4 dimensions the user specified (promoter/management
    pedigree, business model & moat, operating trajectory, industry/macro
    headwinds) to be assessed individually and tagged grounded vs. inferred
    vs. unavailable — rather than one blended paragraph — so a downstream
    deterministic step can compute how much of this is actually sourced,
    not just LLM narrative.
    """
    raw_score: int = Field(ge=0, le=100, description="100 = excellent business quality & management, 0 = severe structural/governance concerns")
    summary: str
    promoter_management_pedigree: BusinessDimensionFinding
    business_model_moat: BusinessDimensionFinding
    operating_trajectory: BusinessDimensionFinding
    industry_macro_headwinds: BusinessDimensionFinding


class BusinessResult(BaseModel):
    entity_screened: str
    raw_score: int = Field(ge=0, le=100)
    summary: str
    promoter_management_pedigree: dict[str, Any]
    business_model_moat: dict[str, Any]
    operating_trajectory: dict[str, Any]
    industry_macro_headwinds: dict[str, Any]
    grounded_count: int
    inferred_count: int
    unavailable_count: int
    completeness_ratio: float = Field(ge=0.0, le=1.0, description="grounded_count / 4 — inferred/unavailable don't count as sourced")


# ---------------------------------------------------------------------------
# Banking specialist (Phase 4) — only invoked when the entity is classified
# as a bank (tools/bank_classifier.py). Same 4-dimension shape as Business &
# Management, banking-specific framing, plus RBI peer-group benchmarks
# (tools/banking_tools.py) as deterministic grounding context.
# ---------------------------------------------------------------------------

class BankingLLMAssessment(BaseModel):
    raw_score: int = Field(ge=0, le=100, description="100 = excellent banking-specific fundamentals & governance, 0 = severe concerns")
    summary: str
    promoter_management_pedigree: BusinessDimensionFinding
    business_model_moat: BusinessDimensionFinding
    operating_trajectory: BusinessDimensionFinding
    industry_macro_headwinds: BusinessDimensionFinding


class BankingResult(BaseModel):
    entity_screened: str
    bank_category: Optional[str] = None
    raw_score: int = Field(ge=0, le=100)
    summary: str
    promoter_management_pedigree: dict[str, Any]
    business_model_moat: dict[str, Any]
    operating_trajectory: dict[str, Any]
    industry_macro_headwinds: dict[str, Any]
    grounded_count: int
    inferred_count: int
    unavailable_count: int
    completeness_ratio: float = Field(ge=0.0, le=1.0)
    rbi_benchmark_available: bool = Field(description="Whether an RBI peer-group benchmark existed for this bank's category")
    rbi_benchmark: Optional[dict[str, Any]] = Field(
        default=None, description="The raw tools/banking_tools.py result — kept for the report to render as a real table, not just a boolean"
    )


# ---------------------------------------------------------------------------
# Score Engine (Phase 5) — fully deterministic, no LLM. See
# credit_report/scoring.py.
# ---------------------------------------------------------------------------

class AgentScoreBreakdown(BaseModel):
    agent: str = Field(description="'finance' | 'hygiene' | 'business' | 'banking'")
    raw_score: int = Field(ge=0, le=100, description="The agent's own qualitative judgment, unadjusted")
    completeness_ratio: float = Field(ge=0.0, le=1.0, description="Deterministically counted by the agent, not LLM-claimed")
    shrunk_score: float = Field(
        ge=0.0, le=100.0,
        description="raw_score pulled toward the neutral 50 baseline by (1 - completeness_ratio) — a score built on thin evidence can't stay near its raw extreme",
    )


class Scores(BaseModel):
    finance: AgentScoreBreakdown
    hygiene: AgentScoreBreakdown
    business: AgentScoreBreakdown
    banking: Optional[AgentScoreBreakdown] = Field(default=None, description="None for non-bank entities — excluded from the average, not scored as 0")
    final_score: float = Field(ge=0.0, le=100.0)
    agents_averaged: int = Field(description="3 for non-banks, 4 for banks")


# ---------------------------------------------------------------------------
# Final assembled report (Phase 6) — see credit_report/reporting/assembler.py.
# Deterministic: every field is copied from earlier state or computed by a
# fixed rule, no LLM call happens at this stage.
# ---------------------------------------------------------------------------

class CreditReport(BaseModel):
    generated_at: date = Field(default_factory=date.today)
    company_name: str
    company_reference: str
    ticker: Optional[str] = None
    has_ticker: bool
    is_bank: bool
    bank_category: Optional[str] = None

    scores: Scores
    finance_result: dict[str, Any]
    hygiene_result: dict[str, Any]
    business_result: dict[str, Any]
    banking_result: Optional[dict[str, Any]] = None

    recommendation: str = Field(description="'Recommended' | 'Recommended with Conditions' | 'Not Recommended' — fixed score thresholds, see reporting/assembler.py")
    recommendation_note: str

    issues: list[dict[str, Any]] = Field(default_factory=list, description="Every DataIssue accumulated across the whole pipeline run — surfaced, not hidden")

    disclaimer: str = Field(
        default=(
            "This credit assessment report is generated by an automated multi-agent "
            "pipeline using open, publicly available data sources for testing/demo "
            "purposes. It does not constitute a final credit decision and must not be "
            "used as a substitute for professional underwriting due diligence. Scores "
            "reflect a deterministic, confidence-weighted synthesis of each agent's own "
            "qualitative assessment, not independently verified ground truth."
        )
    )
