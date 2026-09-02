"""
Bank-group-wise regulatory benchmarks from RBI's annual "Report on Trend and
Progress of Banking in India" — the closest freely-available, India-specific
source for CRAR, NPA ratios, leverage, liquidity, and spread.

Scope note, found the hard way — checked three candidates before settling
here, not assumed:
  - DBIE (data.rbi.org.in/DBIE): no clean bank-wise table found via public
    search; appears to be sector/aggregate time series.
  - "Statistical Tables Relating to Banks in India" at the commonly-cited
    URL (rbi.org.in/upload/Publications/PDFs/57202.pdf): fetched directly —
    returns a 4.7KB stub, not the actual publication.
  - The "Report on Trend and Progress of Banking in India" (this module's
    source): real, current (published 2025-12-29, data as of 2025-03-31),
    substantial (220 pages) — but breaks figures out by BANK GROUP (Public
    Sector / Private Sector / Foreign / Small Finance Banks), never by
    individual bank name. RBI does not publish e.g. "HDFC Bank's own CRAR"
    anywhere public; that level of detail lives in each bank's own
    quarterly results/investor presentations, not a unified RBI source.

So: this gives genuine, current, RBI-sourced PEER-GROUP benchmarks — what's
normal for a bank in this category — not the specific bank's own figures.
credit_report/agents/banking.py is expected to pair this with
search_web_news (a listed bank's own CRAR/NPA is routinely reported in
business press) to compare the specific bank against its peer-group
benchmark from here.

Snapshot, not a live fetch: the source is an ~8MB, 220-page annual PDF;
re-downloading and re-parsing it on every call would be wasteful for data
that only changes once a year. Same hardcoded-snapshot-with-staleness-check
pattern as tools/aml_tools.py's TI CPI / FATF lists. Figures below were
extracted directly from Tables IV.4, IV.5, IV.7, and IV.9 of the source PDF
(verified by downloading and reading it, not taken from a secondary source).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SNAPSHOT_AS_OF = date(2025, 3, 31)
_SNAPSHOT_PUBLISHED = date(2025, 12, 29)
_SOURCE_NAME = "RBI Report on Trend and Progress of Banking in India 2024-25"
_SOURCE_URL = (
    "https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/"
    "0RTP291220258C89B9E5F3F240AEB82AC25A1707A8C6.PDF"
)

# "Payments Bank" is deliberately absent — RBI's own tables here explicitly
# exclude PBs ("Data for SCBs exclude RRBs and PBs").
_BANK_GROUP_BENCHMARKS: dict[str, dict[str, float]] = {
    "Public Sector Bank": {
        "crar_pct": 16.1, "tier1_capital_ratio_pct": 13.8,
        "gross_npa_pct": 2.6, "net_npa_pct": 0.5,
        "leverage_ratio_pct": 6.3, "liquidity_coverage_ratio_pct": 132.1,
        "cost_of_funds_pct": 5.2, "return_on_funds_pct": 8.0, "spread_pct": 2.8,
    },
    "Private Sector Bank": {
        "crar_pct": 18.3, "tier1_capital_ratio_pct": 16.8,
        "gross_npa_pct": 1.8, "net_npa_pct": 0.5,
        "leverage_ratio_pct": 10.2, "liquidity_coverage_ratio_pct": 128.6,
        "cost_of_funds_pct": 5.5, "return_on_funds_pct": 9.2, "spread_pct": 3.7,
    },
    "Foreign Bank": {
        "crar_pct": 18.8, "tier1_capital_ratio_pct": 17.1,
        "gross_npa_pct": 0.9, "net_npa_pct": 0.1,
        "leverage_ratio_pct": 10.3, "liquidity_coverage_ratio_pct": 149.6,
        "cost_of_funds_pct": 4.2, "return_on_funds_pct": 7.4, "spread_pct": 3.2,
    },
    "Small Finance Bank": {
        "crar_pct": 21.5, "tier1_capital_ratio_pct": 18.8,
        "gross_npa_pct": 3.6, "net_npa_pct": 1.4,
        "leverage_ratio_pct": 10.0, "liquidity_coverage_ratio_pct": 153.7,
        "cost_of_funds_pct": 7.2, "return_on_funds_pct": 13.9, "spread_pct": 6.7,
    },
}

_SCB_AGGREGATE_BENCHMARK: dict[str, float] = {
    "crar_pct": 17.4, "tier1_capital_ratio_pct": 15.5,
    "gross_npa_pct": 2.2, "net_npa_pct": 0.5,
    "leverage_ratio_pct": 8.0, "liquidity_coverage_ratio_pct": 132.6,
    "cost_of_funds_pct": 5.3, "return_on_funds_pct": 8.5, "spread_pct": 3.2,
}


def get_banking_sector_benchmark(bank_category: Optional[str]) -> dict[str, Any]:
    """
    Return RBI-sourced peer-group regulatory benchmarks (CRAR, GNPA%, NNPA%,
    leverage, LCR, spread) for the given bank category — e.g. as classified
    by tools/bank_classifier.py:classify_bank(). Never fabricates a
    bank-specific figure; this is explicitly peer-group context, not the
    bank's own numbers.
    """
    today = date.today()
    staleness_note = None
    if (today.year - _SNAPSHOT_AS_OF.year) >= 2:
        staleness_note = (
            f"Benchmark snapshot is as of {_SNAPSHOT_AS_OF.isoformat()} — "
            "check for a newer RBI Trend & Progress report."
        )

    group = _BANK_GROUP_BENCHMARKS.get(bank_category or "")
    if not group:
        return {
            "status": "not_available",
            "bank_category": bank_category,
            "reason": (
                f"No RBI bank-group benchmark for category {bank_category!r} — "
                "covers Public Sector, Private Sector, Foreign, and Small Finance "
                "Banks only (RBI's source tables explicitly exclude Payments Banks "
                "and Regional Rural Banks)."
            ),
            "source": _SOURCE_NAME,
            "source_url": _SOURCE_URL,
        }

    return {
        "status": "ok",
        "bank_category": bank_category,
        "as_of": _SNAPSHOT_AS_OF.isoformat(),
        "benchmarks": group,
        "scb_aggregate_benchmarks": _SCB_AGGREGATE_BENCHMARK,
        "source": _SOURCE_NAME,
        "source_url": _SOURCE_URL,
        "staleness_note": staleness_note,
    }
