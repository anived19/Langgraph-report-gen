"""
Deterministic conditional edges for the parent graph.

No model ever decides *whether* a specialist runs — these are plain
Python functions inspecting state, per the architecture doc's core rule
that routing is graph code, not model judgment.
"""
from __future__ import annotations

from typing import Any, Literal


def route_after_specialists(state: dict[str, Any]) -> Literal["banking_agent", "score_engine"]:
    """
    After Finance, Hygiene, and Business & Management have all joined
    (fan-in from the parallel fan-out at classify_company): run the
    Banking agent only if the entity was classified as a bank.
    """
    return "banking_agent" if state.get("is_bank") else "score_engine"
