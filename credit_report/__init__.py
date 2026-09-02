"""
Credit assessment report generator — LangGraph orchestration.

Separate pipeline from harness/ (which drives the equity/valuation report
flow). Shares the data layer in tools/ (finance_tools, aml_tools,
ticker_resolver, bank_classifier) but has its own graph, agents, scoring,
and report format — see the build plan for the full design.
"""
