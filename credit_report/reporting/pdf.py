"""
Markdown-free HTML -> PDF compilation for credit reports. Reuses
tools/pdf_tools.py:write_pdf() (the WeasyPrint-first/xhtml2pdf-fallback
engine logic, shared with the equity-report pipeline) rather than
duplicating it — this module only owns its own Jinja2 template and its own
data model (CreditReport), not PDF-engine plumbing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import settings
from credit_report.schemas import CreditReport
from tools.pdf_tools import write_pdf

_TEMPLATES_DIR = Path(__file__).parent


def render_credit_report_html(report: CreditReport) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("template.html")
    return template.render(report=report)


def compile_credit_report_pdf(report: CreditReport, output_path: Optional[Path] = None) -> Path:
    if output_path is None:
        safe_name = "".join(c if c.isalnum() else "_" for c in report.company_name)[:60] or "credit_report"
        output_path = settings.output_dir / f"{safe_name}_{report.generated_at.isoformat()}_credit.pdf"
    output_path = Path(output_path)

    html = render_credit_report_html(report)
    return write_pdf(html, output_path, base_url=str(_TEMPLATES_DIR))
