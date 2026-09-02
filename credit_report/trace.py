"""
Persists the full execution trace of each agent's internal ReAct loop — the
LangGraph message-by-message history (every model turn's reasoning/tool
choice, every tool result), not just the final structured output.

Why this exists: an agent's `.invoke()` return value already contains this
(LangGraph's own `messages` state), but every agent function in
credit_report/agents/*.py only ever pulled out `structured_response` and let
the rest get garbage-collected. The trace was real and inspectable while
running (that's how every claim about conditional tool-calling in this
project was verified — instrumenting a live run), but nothing persisted it
afterward. This module is the fix: convert the raw LangChain message objects
into a JSON-safe structure and carry it through state to be written
alongside the report, so "why did Hygiene score this a 70" has the actual
tool-call sequence to point to, not just a claim.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)


def _serialize_message(m: Any) -> dict[str, Any]:
    cls = m.__class__.__name__

    if cls == "AIMessage":
        tool_calls = getattr(m, "tool_calls", None) or []
        return {
            "type": "ai",
            "content": str(m.content) if m.content else None,
            "tool_calls": [
                {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                for tc in tool_calls
            ],
        }
    if cls == "ToolMessage":
        content = m.content
        return {
            "type": "tool_result",
            "tool_name": getattr(m, "name", None),
            "tool_call_id": getattr(m, "tool_call_id", None),
            "content": content if isinstance(content, (str, int, float, bool, type(None))) else str(content),
        }
    if cls == "HumanMessage":
        return {"type": "human", "content": str(m.content)}
    if cls == "SystemMessage":
        return {"type": "system", "content": str(m.content)}
    return {"type": cls.lower(), "content": str(getattr(m, "content", ""))}


def serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert a LangGraph agent's raw `messages` state into a JSON-safe trace."""
    return [_serialize_message(m) for m in messages]


def build_agent_trace(agent_name: str, entity_name: str, messages: list[Any]) -> dict[str, Any]:
    """
    Build one trace entry for CreditReportState["traces"]. `messages` is the
    `agent_result["messages"]` list from that agent's own `agent.invoke(...)`
    call (the full internal loop history), not the outer graph's messages —
    the outer graph has none, it's a plain sequential DAG (see graph.py).
    """
    serialized = serialize_messages(messages)
    tool_call_rounds = sum(1 for e in serialized if e["type"] == "ai" and e["tool_calls"])
    tool_results = sum(1 for e in serialized if e["type"] == "tool_result")
    return {
        "agent": agent_name,
        "entity": entity_name,
        "graph_steps": len(serialized),
        "tool_call_rounds": tool_call_rounds,
        "tool_results": tool_results,
        "messages": serialized,
    }


def dump_traces(traces: list[dict[str, Any]], company_name: str, output_path: Optional[Path] = None) -> Path:
    """
    Write every agent's trace for one report run to a single JSON file,
    alongside the PDF — same naming convention as the equity pipeline's own
    harness/orchestrator.py:_dump_trace(). Never fails the run over this:
    a report that generated fine but couldn't write its trace file should
    still complete, not crash — caller should treat this as best-effort.
    """
    if output_path is None:
        safe_name = "".join(c if c.isalnum() else "_" for c in company_name)[:60] or "credit_report"
        output_path = settings.output_dir / f"{safe_name}_{date.today().isoformat()}_trace.json"
    output_path = Path(output_path)

    payload = {
        "company_name": company_name,
        "generated_at": date.today().isoformat(),
        "agents": traces,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Trace file written to %s", output_path)
    return output_path
