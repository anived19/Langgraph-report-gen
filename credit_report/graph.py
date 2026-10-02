"""
Parent graph: deterministic routing over Directors (no LLM) -> Finance ->
Hygiene -> Business & Management -> Banking (only if is_bank) -> Score
Engine -> Report Synthesizer — a strict sequential queue, not a fan-out, for
rate-limit pacing (see the note below and credit_report/model.py).

Directors runs first because Hygiene needs the entity's identifiers and
registry legal name from it (exact-entity watchlist matching) and Business
reasons over the board it returns. It makes no model calls.

Topology note #1 (bug found and fixed after Phase 7): Business & Management
is NOT independent of Finance/Hygiene — its prompt and _build_context()
(credit_report/agents/business.py) explicitly reason over their results.
LangGraph's execution is bulk-synchronous: nodes scheduled in the same
superstep (i.e. parallel fan-out siblings) all read the state as it existed
*before* that superstep — none of them see each other's writes until the
superstep completes. Business was originally included in the same fan-out
as Finance/Hygiene, so it always read finance_result/hygiene_result as
None, silently. Confirmed by instrumenting a real graph run. Fixed by
making Business a genuine downstream step, not a fan-out sibling. Banking
was already correctly downstream of the join in every version of this
graph, so it was never affected by this specific bug.

Topology note #2 (testing-phase choice): Finance and Hygiene ARE genuinely
independent of each other and could run in parallel. They're deliberately
queued sequentially anyway — not for correctness, but to keep model call
volume spread out under the shared rate limiter (credit_report/model.py,
20 RPM on the Foundry deployment) rather than bursting two agents' calls at
once. Each specialist
can also now call its own tools (search_web_news in particular) more
readily per the prompt tightening in this same change, which means more
model calls per report than before — reason enough to queue rather than
parallelize while validating this on a free-tier quota. Revisit (restore
the classify_company -> finance_agent / classify_company -> hygiene_agent
fan-out) once quota headroom stops being the binding constraint.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from credit_report.agents.banking import run_banking_agent
from credit_report.agents.business import run_business_agent
from credit_report.agents.directors import run_directors_check
from credit_report.agents.finance import run_finance_agent
from credit_report.agents.hygiene import run_hygiene_agent
from credit_report.company_identity import classify_company, load_company_identity
from credit_report.reporting.assembler import assemble_credit_report
from credit_report.routing import route_after_specialists
from credit_report.scoring import compute_scores
from credit_report.state import CreditReportState


def build_credit_report_graph() -> CompiledStateGraph:
    graph = StateGraph(CreditReportState)

    graph.add_node("load_company_identity", load_company_identity)
    graph.add_node("classify_company", classify_company)
    graph.add_node("directors", run_directors_check)
    graph.add_node("finance_agent", run_finance_agent)
    graph.add_node("hygiene_agent", run_hygiene_agent)
    graph.add_node("business_agent", run_business_agent)
    graph.add_node("banking_agent", run_banking_agent)
    graph.add_node("score_engine", compute_scores)
    graph.add_node("report_synthesizer", assemble_credit_report)

    graph.add_edge(START, "load_company_identity")
    graph.add_edge("load_company_identity", "classify_company")

    # Sequential queue (see module docstring): classify_company -> directors
    # -> finance -> hygiene -> business, one at a time, not a parallel fan-out.
    graph.add_edge("classify_company", "directors")
    graph.add_edge("directors", "finance_agent")
    graph.add_edge("finance_agent", "hygiene_agent")
    graph.add_edge("hygiene_agent", "business_agent")

    # Banking Router: Banking also reasons over Business's result, so it
    # comes after Business, not in parallel with it.
    graph.add_conditional_edges(
        "business_agent", route_after_specialists, ["banking_agent", "score_engine"]
    )

    graph.add_edge("banking_agent", "score_engine")
    graph.add_edge("score_engine", "report_synthesizer")
    graph.add_edge("report_synthesizer", END)

    return graph.compile()
