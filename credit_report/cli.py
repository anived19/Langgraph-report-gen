"""
CLI entry point for the credit-report pipeline.

Usage:
    python -m credit_report.cli "HDFC Bank"
    python -m credit_report.cli "TCS.NS"
    python -m credit_report.cli "JU Agri Sciences Private Limited" --cin U74899DL1986PTC024176

--cin is optional. With it the report adds a Directors section from MCA
records (Probe42 via the Finoscale Data API — a paid call, cached 7 days)
with a PEP screen per director, and watchlist hits can be confirmed against
the company's identifiers.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from credit_report.graph import build_credit_report_graph
from credit_report.reporting.pdf import compile_credit_report_pdf
from credit_report.schemas import CreditReport
from credit_report.trace import dump_traces

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("credit_report.cli")


def generate_credit_report(company_reference: str, cin: Optional[str] = None) -> None:
    print(f"\n> Credit report request: {company_reference}" + (f" (CIN {cin})" if cin else "") + "\n")

    graph = build_credit_report_graph()
    print("[1/2] Running Company Identity -> Directors -> Finance / Hygiene / Business (+ Banking) -> Score Engine...")
    state: dict = {"company_reference": company_reference}
    if cin:
        state["cin"] = cin.strip().upper()
    result = graph.invoke(state)

    report_dict = result["report"]
    print(
        f"      -> {report_dict['company_name']} | ticker={report_dict['ticker']} | "
        f"is_bank={report_dict['is_bank']} ({report_dict.get('bank_category') or 'n/a'})"
    )
    print(
        f"      -> Final score: {report_dict['scores']['final_score']}/100 "
        f"({report_dict['scores']['agents_averaged']} agents) -> {report_dict['recommendation']}"
    )
    if report_dict["issues"]:
        print(f"      -> {len(report_dict['issues'])} data-quality note(s) logged")
    if report_dict["analyst_review"]:
        print(f"      -> {len(report_dict['analyst_review'])} possible namesake(s) listed for analyst review (not scored)")

    print("[2/2] Rendering PDF...")
    report = CreditReport.model_validate(report_dict)
    pdf_path = compile_credit_report_pdf(report)

    try:
        trace_path = dump_traces(result.get("traces", []), report.company_name)
        print(f"      -> Full agent trace (every model turn, every tool call) saved to: {trace_path}")
    except Exception:
        logger.warning("Could not write trace file — report itself still completed fine.", exc_info=True)

    print(f"\nDone. Credit report saved to: {pdf_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Credit Report Generator")
    parser.add_argument(
        "query",
        type=str,
        help='Company reference, e.g. "HDFC Bank", "TCS.NS", "JU Agri Sciences Private Limited"',
    )
    parser.add_argument(
        "--cin",
        default=None,
        help="The company's CIN (optional) — adds directors from MCA records and a PEP screen per director",
    )
    args = parser.parse_args()

    try:
        generate_credit_report(args.query, args.cin)
    except Exception:
        logger.exception("Credit report generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
