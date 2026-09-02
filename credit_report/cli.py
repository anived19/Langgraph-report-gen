"""
CLI entry point for the credit-report pipeline.

Usage:
    python -m credit_report.cli "HDFC Bank"
    python -m credit_report.cli "TCS.NS"
    python -m credit_report.cli "JU Agri Sciences Private Limited"
"""
from __future__ import annotations

import argparse
import logging
import sys

from credit_report.graph import build_credit_report_graph
from credit_report.reporting.pdf import compile_credit_report_pdf
from credit_report.schemas import CreditReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("credit_report.cli")


def generate_credit_report(company_reference: str) -> None:
    print(f"\n> Credit report request: {company_reference}\n")

    graph = build_credit_report_graph()
    print("[1/2] Running Company Identity -> Finance / Hygiene / Business (+ Banking) -> Score Engine...")
    result = graph.invoke({"company_reference": company_reference})

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

    print("[2/2] Rendering PDF...")
    report = CreditReport.model_validate(report_dict)
    pdf_path = compile_credit_report_pdf(report)
    print(f"\nDone. Credit report saved to: {pdf_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Credit Report Generator")
    parser.add_argument(
        "query",
        type=str,
        help='Company reference, e.g. "HDFC Bank", "TCS.NS", "JU Agri Sciences Private Limited"',
    )
    args = parser.parse_args()

    try:
        generate_credit_report(args.query)
    except Exception:
        logger.exception("Credit report generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
