"""
Deterministic bank classification against RBI's official "List of Scheduled
Commercial Banks (SCBs)" — the current, RBI-maintained categorization of every
entity licensed as a bank in India (Public Sector, Private Sector, Foreign,
Small Finance, Payments, and Regional Rural Banks).

Per the credit-report architecture's Banking Router / Company Classification
step: banking-specific scoring should only run for entities that are actually
banks. Per RBI's own framing of the Second Schedule, an entity absent from this
list is legally not a bank — this module fails closed (reports "not a bank")
when the source can't be reached, rather than guessing.

Source discovery note: RBI does not keep one static, stable-filename PDF for
this list — the underlying file is replaced (new hash-suffixed filename) every
time RBI updates it; the old, commonly-linked "Second Schedule" PDF found via
web search (rbidocs.../publications/pdfs/84656.pdf) is a stale ~2013-14 archival
copy that still lists merged/defunct banks (Dena Bank, Vijaya Bank, ING Vysya)
and predates Small Finance Banks and Payments Banks entirely — confirmed by
fetching it directly, not assumed. The current list was found by following RBI's
own site navigation (Regulated Entities -> "List of Scheduled Commercial Banks
(SCBs)"), landing on a stable page (`_RBI_SCB_LANDING_PAGE`) that itself always
points at whatever the current PDF is. So: always re-resolve the live PDF link
from that landing page at request time rather than hardcoding a PDF filename
that will rot — the same principle used for the World Bank AML source fix in
tools/aml_tools.py (don't hardcode a URL that goes stale; hardcode the page that
always points at the current one).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
import pypdf

from config import settings
from schemas import BankClassification
from tools.aml_tools import _name_matches
from utils.retry import retry_on_transient_error

logger = logging.getLogger(__name__)

# Stable RBI navigation page for the SCB list (Regulated Entities -> this link).
# Resolved live via RBI's own site nav on 2026-09-01 — the PDF it points to had
# been updated 2026-06-10, listing 12 PSBs / 21 private banks / 44 foreign banks /
# 11 SFBs / 5 payments banks / 28 RRBs, consistent with current market figures.
_RBI_SCB_LANDING_PAGE = "https://www.rbi.org.in/scripts/bs_viewcontent.aspx?Id=3657"

_CACHE_DIR = settings.cache_dir / "bank_classifier"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# RBI updates this list only when a bank is licensed, merged, or delicensed —
# an infrequent event — so a week-long cache is safe and avoids re-fetching an
# ~140KB PDF on every entity classified.
_CACHE_TTL_HOURS = 24 * 7

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,*/*;q=0.8",
}

_SECTION_TO_CATEGORY = {
    "sbi & nationalised banks": "Public Sector Bank",
    "domestic private sector banks": "Private Sector Bank",
    "payments banks": "Payments Bank",
    "regional rural banks": "Regional Rural Bank",
    "small finance banks": "Small Finance Bank",
    "foreign banks in india": "Foreign Bank",
}

_NOISE_LINE_PATTERNS = (
    re.compile(r"^sr\.?\s*no\.?$", re.I),
    re.compile(r"^sl\.?\s*no\.?$", re.I),
    re.compile(r"^name of (the )?bank$", re.I),
    re.compile(r"^name of the rrb$", re.I),
    re.compile(r"^list of scheduled commercial banks", re.I),
    re.compile(r"^\*"),  # footnote lines, e.g. "* Banking licence cancelled."
)

_NUMBERED_LINE = re.compile(r"^(\d+)\.\s*(.+)$")


# ---------------------------------------------------------------------------
# Small disk cache (mirrors the pattern in tools/aml_tools.py / scraper_tools.py)
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{h}.cache"


def _read_cache(key: str, ttl_hours: float) -> Optional[bytes]:
    path = _cache_path(key)
    if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < ttl_hours:
        try:
            return path.read_bytes()
        except Exception:
            return None
    return None


def _write_cache(key: str, data: bytes) -> None:
    try:
        _cache_path(key).write_bytes(data)
    except Exception as exc:
        logger.debug("bank_classifier cache write failed for %s: %s", key, exc)


@retry_on_transient_error(max_attempts=3)
def _get(url: str) -> httpx.Response:
    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20) as client:
        return client.get(url)


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def _resolve_current_pdf_url() -> Optional[str]:
    """Re-resolve the live PDF link from the stable landing page every cache window,
    rather than trusting a previously seen filename (RBI republishes under a new
    hash-suffixed filename each update)."""
    cache_key = f"landing:{_RBI_SCB_LANDING_PAGE}"
    cached = _read_cache(cache_key, ttl_hours=_CACHE_TTL_HOURS)
    if cached:
        html = cached.decode("utf-8", errors="replace")
    else:
        try:
            resp = _get(_RBI_SCB_LANDING_PAGE)
            resp.raise_for_status()
            html = resp.text
            _write_cache(cache_key, html.encode("utf-8"))
        except Exception as exc:
            logger.warning("Could not fetch RBI SCB landing page %s: %s", _RBI_SCB_LANDING_PAGE, exc)
            return None

    m = re.search(r'href=["\']([^"\']+\.PDF)["\']', html, re.I)
    if not m:
        logger.warning("Could not locate a .PDF link on RBI SCB landing page")
        return None

    url = m.group(1)
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://www.rbi.org.in" + url
    return url


def _fetch_scb_pdf_bytes() -> Optional[bytes]:
    pdf_url = _resolve_current_pdf_url()
    if not pdf_url:
        return None

    cache_key = f"pdf:{pdf_url}"
    cached = _read_cache(cache_key, ttl_hours=_CACHE_TTL_HOURS)
    if cached:
        return cached

    try:
        resp = _get(pdf_url)
        resp.raise_for_status()
        _write_cache(cache_key, resp.content)
        return resp.content
    except Exception as exc:
        logger.warning("Could not fetch RBI SCB PDF at %s: %s", pdf_url, exc)
        return None


def _parse_scb_pdf(pdf_bytes: bytes) -> dict[str, list[str]]:
    """Returns {category: [bank names]} parsed from the RBI SCB list PDF."""
    reader = pypdf.PdfReader(BytesIO(pdf_bytes))
    lines: list[str] = []
    for page in reader.pages:
        lines.extend((page.extract_text() or "").splitlines())

    categorized: dict[str, list[str]] = {c: [] for c in _SECTION_TO_CATEGORY.values()}
    current_category: Optional[str] = None
    pending_name: Optional[str] = None

    def _flush() -> None:
        nonlocal pending_name
        if pending_name and current_category:
            clean = re.sub(r"\*+\s*$", "", pending_name).strip()
            if clean:
                categorized[current_category].append(clean)
        pending_name = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        section_key = line.lower().rstrip(" *")
        if section_key in _SECTION_TO_CATEGORY:
            _flush()
            current_category = _SECTION_TO_CATEGORY[section_key]
            continue

        if any(p.match(line) for p in _NOISE_LINE_PATTERNS):
            continue

        m = _NUMBERED_LINE.match(line)
        if m:
            _flush()
            pending_name = m.group(2).strip()
        elif current_category and pending_name:
            # A bank name that wrapped onto the next PDF line (e.g. the Rabobank
            # entry under Foreign Banks) — glue it back onto the pending entry.
            pending_name = f"{pending_name} {line}".strip()

    _flush()
    return categorized


def _load_scb_registry() -> Optional[dict[str, list[str]]]:
    pdf_bytes = _fetch_scb_pdf_bytes()
    if not pdf_bytes:
        return None
    try:
        registry = _parse_scb_pdf(pdf_bytes)
        if not any(registry.values()):
            logger.warning("RBI SCB PDF parsed but produced zero entries — treating as unavailable")
            return None
        return registry
    except Exception as exc:
        logger.warning("Could not parse RBI SCB PDF: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_bank(entity_name: str) -> BankClassification:
    """
    Deterministically classify whether `entity_name` is a licensed bank in India,
    per RBI's current "List of Scheduled Commercial Banks (SCBs)". If the entity
    is not on this list, it is legally not a bank.
    """
    registry = _load_scb_registry()
    if not registry:
        return BankClassification(
            entity_screened=entity_name,
            is_bank=False,
            bank_category=None,
            matched_name=None,
            confidence="unavailable",
            source_url=_RBI_SCB_LANDING_PAGE,
            note=(
                "RBI's Scheduled Commercial Banks list could not be fetched/parsed this "
                "run. Classification defaults to non-bank (fail-closed) — verify manually "
                "before excluding banking-specific analysis for this entity."
            ),
        )

    for category, names in registry.items():
        for name in names:
            if _name_matches(entity_name, name):
                return BankClassification(
                    entity_screened=entity_name,
                    is_bank=True,
                    bank_category=category,
                    matched_name=name,
                    confidence="matched",
                    source_url=_RBI_SCB_LANDING_PAGE,
                    note=None,
                )

    return BankClassification(
        entity_screened=entity_name,
        is_bank=False,
        bank_category=None,
        matched_name=None,
        confidence="checked_not_found",
        source_url=_RBI_SCB_LANDING_PAGE,
        note="Not found on RBI's List of Scheduled Commercial Banks — not a licensed bank.",
    )
