"""
Shared Pydantic schemas for data passed between pipeline stages.

Every value that crosses a stage boundary is validated against one of
these models before being trusted downstream. This is the concrete
enforcement mechanism behind the "no hallucinated numbers, all claims
cited" constraint: if a stage's output doesn't parse, the pipeline fails
loudly instead of silently passing bad data to the PDF.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1: Market Data
# Deterministic — populated directly from yfinance in tools/finance_tools.py.
# No LLM ever touches these fields, so there's no re-typing/paraphrasing
# step where a number could drift from its source.
# ---------------------------------------------------------------------------

class PricePoint(BaseModel):
    date: date
    close: float


class QuarterlyDataPoint(BaseModel):
    """One quarter's revenue and net income with computed growth rates."""
    quarter: str                            # e.g. "Q1 FY2025"
    revenue: Optional[float] = None         # in absolute units (as reported)
    revenue_formatted: Optional[str] = None # e.g. "Rs. 72,275 Cr"
    net_income: Optional[float] = None
    net_income_formatted: Optional[str] = None # e.g. "Rs. 13,349 Cr"
    revenue_growth_qoq: Optional[float] = None   # quarter-over-quarter % change
    revenue_growth_qoq_formatted: Optional[str] = None # e.g. "+2.23%"
    revenue_growth_yoy: Optional[float] = None   # year-over-year % change
    profit_growth_qoq: Optional[float] = None
    profit_growth_qoq_formatted: Optional[str] = None  # e.g. "-2.69%"
    profit_growth_yoy: Optional[float] = None
    data_gap_note: Optional[str] = None     # populated if a gap is detected between reporting periods


def format_currency_amount(amount: Optional[float], currency: Optional[str] = None) -> Optional[str]:
    """
    Format large numbers into standard financial scale strings.
    For INR: Lakh Cr, Cr, Lakhs (e.g. Rs. 17.73 Lakh Cr, Rs. 1,250.00 Cr).
    For USD/others: T, B, M, K (e.g. $3.15T, $500.00B, $45.20M).
    """
    if amount is None:
        return None
    curr = (currency or "").strip().upper()
    if curr in ("INR", "₹", "RS", "RUPEES"):
        prefix = "Rs. "
        abs_val = abs(amount)
        if abs_val >= 1e12:
            return f"{prefix}{amount / 1e12:.2f} Lakh Cr"
        elif abs_val >= 1e7:
            return f"{prefix}{amount / 1e7:,.2f} Cr"
        elif abs_val >= 1e5:
            return f"{prefix}{amount / 1e5:,.2f} Lakhs"
        else:
            return f"{prefix}{amount:,.2f}"
    else:
        prefix = "$" if curr in ("USD", "$", "") else f"{curr} "
        abs_val = abs(amount)
        if abs_val >= 1e12:
            return f"{prefix}{amount / 1e12:.2f}T"
        elif abs_val >= 1e9:
            return f"{prefix}{amount / 1e9:.2f}B"
        elif abs_val >= 1e6:
            return f"{prefix}{amount / 1e6:.2f}M"
        elif abs_val >= 1e3:
            return f"{prefix}{amount / 1e3:.2f}K"
        else:
            return f"{prefix}{amount:,.2f}"


def format_number_amount(amount: Optional[float], decimals: int = 2) -> Optional[str]:
    """Format number with thousands separators and fixed decimal precision (e.g. 2456.122 -> '2,456.12')."""
    if amount is None:
        return None
    return f"{amount:,.{decimals}f}"


def format_percent(val: Optional[float], include_sign: bool = False, decimals: int = 2) -> Optional[str]:
    """Format decimal fraction or percentage into string with % sign (e.g. 0.40389 -> '40.39%')."""
    if val is None:
        return None
    if abs(val) <= 1.0:
        pct = val * 100.0
    else:
        pct = val
    if include_sign:
        return f"{pct:+.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


class MarketMetrics(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    # --- Price & basic technicals ---
    current_price: Optional[float] = None
    current_price_formatted: Optional[str] = None
    fifty_day_ma: Optional[float] = None
    fifty_day_ma_formatted: Optional[str] = None
    two_hundred_day_ma: Optional[float] = None
    two_hundred_day_ma_formatted: Optional[str] = None

    # --- Extended technicals (computed from price history) ---
    rsi_14: Optional[float] = None                 # 14-day RSI
    macd_line: Optional[float] = None              # MACD line (12-26 EMA diff)
    macd_signal: Optional[float] = None            # Signal line (9-day EMA of MACD)
    macd_histogram: Optional[float] = None         # MACD - Signal
    volume_20d_avg: Optional[float] = None         # 20-day average daily volume
    volume_trend: Optional[str] = None             # "rising" | "falling" | "flat"
    support_level: Optional[float] = None          # derived from outlook-window price low
    resistance_level: Optional[float] = None       # derived from outlook-window price high

    # --- Valuation multiples ---
    market_cap: Optional[float] = None
    market_cap_formatted: Optional[str] = None # e.g. "Rs. 17.73 Lakh Cr" or "$3.12T"
    pe_ratio: Optional[float] = None          # trailing P/E
    pe_ratio_formatted: Optional[str] = None   # e.g. "17.15"
    forward_pe: Optional[float] = None        # forward P/E
    forward_pe_formatted: Optional[str] = None # e.g. "14.54"
    pb_ratio: Optional[float] = None          # price-to-book
    pb_ratio_formatted: Optional[str] = None   # e.g. "7.79"
    ps_ratio: Optional[float] = None          # price-to-sales (TTM)
    ps_ratio_formatted: Optional[str] = None   # e.g. "3.10"
    ev_ebitda: Optional[float] = None         # EV / EBITDA
    ev_ebitda_formatted: Optional[str] = None  # e.g. "11.40"
    dividend_yield: Optional[float] = None    # as a decimal, e.g. 0.012 = 1.2%
    dividend_yield_formatted: Optional[str] = None # e.g. "2.75%"

    # --- Earnings & revenue ---
    eps_ttm: Optional[float] = None           # EPS trailing twelve months
    eps_ttm_formatted: Optional[str] = None   # e.g. "137.65"
    revenue_ttm: Optional[float] = None       # total revenue TTM
    revenue_ttm_formatted: Optional[str] = None # e.g. "Rs. 9.50 Lakh Cr" or "$380.50B"
    gross_margin: Optional[float] = None      # gross margin as decimal
    gross_margin_formatted: Optional[str] = None # e.g. "40.39%"
    operating_margin: Optional[float] = None  # operating margin as decimal
    operating_margin_formatted: Optional[str] = None # e.g. "23.96%"

    # --- Extended fundamentals (from yfinance .info) ---
    debt_to_equity: Optional[float] = None    # total debt / total equity
    debt_to_equity_formatted: Optional[str] = None # e.g. "10.21"
    roe: Optional[float] = None               # return on equity (decimal)
    roe_formatted: Optional[str] = None       # e.g. "47.74%"
    roce: Optional[float] = None              # return on capital employed (decimal)
    roce_formatted: Optional[str] = None      # e.g. "54.93%"

    # --- Analyst ratings (from yfinance .info / recommendations / price targets) ---
    analyst_buy_count: Optional[int] = None
    analyst_hold_count: Optional[int] = None
    analyst_sell_count: Optional[int] = None
    analyst_target_mean: Optional[float] = None
    analyst_target_mean_formatted: Optional[str] = None # e.g. "2,456.12"
    analyst_target_high: Optional[float] = None
    analyst_target_high_formatted: Optional[str] = None # e.g. "3,480.00"
    analyst_target_low: Optional[float] = None
    analyst_target_low_formatted: Optional[str] = None  # e.g. "1,800.00"
    analyst_recommendation: Optional[str] = None  # e.g. "buy", "hold", "sell"

    # --- Ownership / holdings (from yfinance .major_holders) ---
    promoter_holding_pct: Optional[float] = None   # % held by promoters/insiders
    promoter_holding_pct_formatted: Optional[str] = None # e.g. "71.80%"
    fii_holding_pct: Optional[float] = None        # % held by foreign institutional investors
    institutional_holding_pct_formatted: Optional[str] = None # e.g. "17.45%"
    dii_holding_pct: Optional[float] = None        # % held by domestic institutional investors
    public_holding_pct: Optional[float] = None     # residual public float %
    public_holding_pct_formatted: Optional[str] = None # e.g. "10.75%"

    # --- Quarterly financials (last 4 quarters, computed in finance_tools) ---
    quarterly_financials: list[QuarterlyDataPoint] = Field(default_factory=list)

    # --- Configurable outlook window ---
    outlook_months: int = Field(default=6, description="Number of months the price window covers.")
    outlook_high: Optional[float] = None
    outlook_high_formatted: Optional[str] = None
    outlook_low: Optional[float] = None
    outlook_low_formatted: Optional[str] = None
    outlook_price_trend: list[PricePoint] = Field(default_factory=list)
    custom_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Ad-hoc deterministic metrics computed via Python calculation sandbox.",
    )

    unavailable_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Field names yfinance could not supply. Surfaced explicitly so "
            "the Chief Editor states 'unretrievable' rather than the gap "
            "being silently papered over."
        ),
    )

    fetched_at: date = Field(default_factory=date.today)


# ---------------------------------------------------------------------------
# AML / ABC screening (tools/aml_tools.py). A list record counts against the
# entity only when its name is exactly the entity's AND it carries one of the
# entity's identifiers (tools/name_match.py); a same-name record with no
# identifier goes to `analyst_review` and is never scored.
# ---------------------------------------------------------------------------

class AMLSeverity(str, Enum):
    NONE     = "None"
    WATCH    = "Watch"
    ELEVATED = "Elevated"
    HIGH     = "High"


class AMLFinding(BaseModel):
    """One screening hit — or an explicit 'no adverse finding' record."""
    entity_screened: str
    source_name: str          # e.g. "OFAC SDN List", "OpenSanctions", "SEC EDGAR FCPA"
    finding_summary: str      # plain-English description of what was (or wasn't) found
    severity: AMLSeverity
    source_url: str           # direct link to the result or source index
    analyst_review: list[str] = Field(
        default_factory=list,
        description=(
            "Exact-name records with no identifier proving they are this entity -- "
            "shown to the analyst, never scored and never shown to the scoring LLM."
        ),
    )


# ---------------------------------------------------------------------------
# Bank classification: deterministic Banking Router input (tools/bank_classifier.py).
# Gates whether banking-specific scoring/analysis runs for an entity — per RBI's
# own framing, an entity absent from the Second Schedule / SCB list is legally
# not a bank.
# ---------------------------------------------------------------------------

class BankClassification(BaseModel):
    entity_screened: str
    is_bank: bool
    bank_category: Optional[str] = Field(
        default=None,
        description=(
            "One of: 'Public Sector Bank', 'Private Sector Bank', 'Foreign Bank', "
            "'Small Finance Bank', 'Payments Bank', 'Regional Rural Bank' — or None "
            "if not a bank / classification unavailable."
        ),
    )
    matched_name: Optional[str] = Field(
        default=None, description="The exact RBI-listed entity name that matched, if any."
    )
    confidence: str = Field(
        description="'matched' | 'checked_not_found' | 'unavailable' (source could not be fetched/parsed this run)"
    )
    source_url: str
    note: Optional[str] = None
