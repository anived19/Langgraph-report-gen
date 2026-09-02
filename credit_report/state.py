"""
Parent graph state schema.

`issues` uses the `Annotated[list, operator.add]` reducer because Finance,
Hygiene, and Business & Management all run in the same superstep (fan-out
from classify_company) and each may append its own issues — without a
reducer, LangGraph's default last-write-wins merge would silently drop all
but one node's issues.
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class CreditReportState(TypedDict, total=False):
    # Intake
    query: str
    company_reference: str

    # Company Identity Loader / Classification (deterministic)
    company_name: Optional[str]
    ticker: Optional[str]
    has_ticker: bool
    is_bank: bool
    bank_category: Optional[str]
    resolution_confidence: Optional[float]

    # Specialist results (each written by exactly one node)
    finance_result: Optional[dict]
    hygiene_result: Optional[dict]
    business_result: Optional[dict]
    banking_result: Optional[dict]

    # Deterministic downstream stages
    scores: Optional[dict]
    report: Optional[dict]

    # Accumulated across every stage — never blocks the pipeline
    issues: Annotated[list[dict], operator.add]

    # One entry per LLM-backed agent (Hygiene/Finance/Business/Banking) — the
    # full internal ReAct loop trace (every model turn, every tool call and
    # result, in order), not just the final structured output. See
    # credit_report/trace.py. Same accumulation reasoning as `issues`.
    traces: Annotated[list[dict], operator.add]
