"""
HTML -> PDF writing for credit reports (credit_report/reporting/pdf.py owns
the template and data model; this module only owns PDF-engine plumbing).

Engine selection: WeasyPrint gives the best CSS fidelity but depends on
system-level Pango/Cairo/GDK-PixBuf libraries that don't install cleanly
on every machine (Windows in particular). xhtml2pdf is pure-pip with weaker
CSS support but zero native-dependency risk. This module tries
`settings.pdf_engine` first and automatically falls back to the other one if
the primary engine fails to import or errors at render time, so a broken
WeasyPrint install on someone's laptop doesn't block report generation
entirely.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def _write_pdf_weasyprint(html: str, output_path: Path, base_url: str) -> None:
    # Imported lazily so a missing/broken system install only breaks this
    # function, not the whole module (and therefore not the xhtml2pdf path).
    from weasyprint import HTML

    HTML(string=html, base_url=base_url).write_pdf(str(output_path))


def _write_pdf_xhtml2pdf(html: str, output_path: Path, base_url: str) -> None:
    from xhtml2pdf import pisa

    with open(output_path, "wb") as f:
        result = pisa.CreatePDF(src=html, dest=f)
    if result.err:
        raise RuntimeError(f"xhtml2pdf reported {result.err} error(s) while rendering the PDF")


_ENGINES = {
    "weasyprint": _write_pdf_weasyprint,
    "xhtml2pdf": _write_pdf_xhtml2pdf,
}


def write_pdf(html: str, output_path: Path, base_url: str | None = None) -> Path:
    """
    Render `html` to a PDF file at `output_path` and return the path.

    Tries settings.pdf_engine first; on any failure (import error, render
    error), automatically retries with the other engine rather than failing
    over a rendering-backend issue. Has no knowledge of the report's data
    model, only an HTML string in and a PDF file out.
    """
    output_path = Path(output_path)
    resolved_base_url = base_url if base_url is not None else str(Path.cwd())

    primary = settings.pdf_engine if settings.pdf_engine in _ENGINES else "weasyprint"
    fallback = "xhtml2pdf" if primary == "weasyprint" else "weasyprint"

    try:
        _ENGINES[primary](html, output_path, resolved_base_url)
        logger.info("PDF rendered via %s -> %s", primary, output_path)
        return output_path
    except Exception as exc:
        logger.warning("PDF engine '%s' failed (%s) — falling back to '%s'", primary, exc, fallback)

    _ENGINES[fallback](html, output_path, resolved_base_url)
    logger.info("PDF rendered via fallback %s -> %s", fallback, output_path)
    return output_path
