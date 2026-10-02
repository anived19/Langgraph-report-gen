"""
Parent graph state schema.

`issues` and `traces` use the `Annotated[list, operator.add]` reducer so
every node's entries accumulate — without a reducer, LangGraph's default
last-write-wins merge would keep only the latest node's list (and would
silently drop entries if nodes ever run in the same superstep again).
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class CreditReportState(TypedDict, total=False):
    # Intake
    query: str
    company_reference: str
    cin: Optional[str]  # optional; unlocks the Directors node (Probe42 / MCA records)

    # Company Identity Loader / Classification (deterministic)
    company_name: Optional[str]
    ticker: Optional[str]
    has_ticker: bool
    is_bank: bool
    bank_category: Optional[str]
    resolution_confidence: Optional[float]

    # Directors node (deterministic): registry legal name and the entity's
    # identifiers (CIN, PAN) for exact-entity watchlist matching. The
    # identifiers are bound into tools in code and never put in a prompt.
    legal_name: Optional[str]
    entity_ids: list[str]
    directors_result: Optional[dict]

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

    # One entry per LLM-backed agent attempt (Hygiene/Finance/Business/
    # Banking; a rerun after failed output checks adds another) — the full
    # internal ReAct loop trace (every model turn, every tool call and
    # result, in order), not just the final structured output. See
    # credit_report/trace.py. Same accumulation reasoning as `issues`.
    traces: Annotated[list[dict], operator.add]
