"""
CRISIL rating-rationale search + fetch — the primary Finance data source for
companies with no resolvable ticker (private/unlisted companies).

Source discovery note: CRISIL's public rating-rationale page
(crisilratings.com/.../rating-rationale.html) is a JS-driven search widget,
but the actual data comes from a plain JSON endpoint with no auth/CAPTCHA —
found via a live Playwright network trace, same approach used for the World
Bank AML fix in tools/aml_tools.py. The rationale document itself is a plain,
unauthenticated HTML file at a predictable URL. Verified end-to-end against
the sample credit report's own company (JU Agri Sciences Private Limited),
which returned its actual "Crisil A/Stable" reaffirmation — matching what
the sample report itself cites.

GST/MCA fallback was researched and deliberately NOT built here: the public
GST ("Search Taxpayer") and MCA ("View Company/LLP Master Data") portals
both require solving a CAPTCHA — genuine anti-automation, not a
missing-header or dead-endpoint problem like the sources fixed elsewhere in
this project — and the most promising free aggregator checked (ZaubaCorp)
sits behind an active Cloudflare bot challenge. Neither is appropriate to
script around. The no-ticker Finance fallback relies on this module plus
tools/search_tools.py's search_web_news for source-cited facts when no
rating rationale exists (see credit_report/agents/finance.py).

CARE and ICRA rationale search were not integrated this phase (CRISIL alone
was already fully verified working and is a substantial, real data source);
worth adding as a follow-up for companies rated only by those agencies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from config import settings
from utils.retry import retry_on_transient_error

logger = logging.getLogger(__name__)

# Business & Management (Phase 3) reuses this module for the same entity
# Finance already looked up — cache so a second consumer doesn't re-hit
# CRISIL for data already fetched this run. Mirrors tools/aml_tools.py's
# cache pattern.
_CACHE_DIR = settings.cache_dir / "rating_agency"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_HOURS = 24


def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{h}.json"


def _read_cache(key: str, ttl_hours: float = _CACHE_TTL_HOURS) -> Optional[dict[str, Any]]:
    path = _cache_path(key)
    if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < ttl_hours:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(key: str, data: dict[str, Any]) -> None:
    try:
        _cache_path(key).write_text(json.dumps(data), encoding="utf-8")
    except Exception as exc:
        logger.debug("rating_agency cache write failed for %s: %s", key, exc)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
}

_CRISIL_SEARCH_URL = (
    "https://www.crisilratings.com/content/crisilratings/en/home/our-business/"
    "ratings/rating-rationale/_jcr_content/wrapper_100_par/ratingresultlisting.results.json"
)
_CRISIL_DOC_BASE_URL = "https://www.crisilratings.com/mnt/winshare/Ratings/RatingList/RatingDocs/"
_TIMEOUT = 20


@retry_on_transient_error(max_attempts=3)
def _get(url: str, params: Optional[dict[str, Any]] = None) -> httpx.Response:
    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=_TIMEOUT) as client:
        return client.get(url, params=params)


def search_crisil_rationale(company_name: str) -> Optional[dict[str, Any]]:
    """
    Search CRISIL's rating-rationale index by company name. Returns the
    best-matching entry's metadata (company_name, heading, rating_date,
    document_url), or None if nothing matched or the search itself failed.
    """
    try:
        resp = _get(_CRISIL_SEARCH_URL, params={
            "cmd": "RR",
            "start": 0,
            "limit": 10,
            "filters": json.dumps({"company_name": company_name}),
        })
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("CRISIL rationale search failed for %r: %s", company_name, exc)
        return None

    docs = data.get("docs") or []
    if not docs:
        return None

    # Exact (case-insensitive) name match wins over the API's own relevance
    # ordering, since a loose filter can return unrelated companies whose
    # name merely overlaps a token (see tools/aml_tools.py's _name_matches
    # for the same class of false-positive risk).
    exact = next(
        (d for d in docs if (d.get("companyName") or "").strip().lower() == company_name.strip().lower()),
        None,
    )
    best = exact or docs[0]

    file_name = best.get("ratingFileName")
    if not file_name:
        return None

    return {
        "company_name": best.get("companyName"),
        "heading": best.get("heading"),
        "rating_date": best.get("ratingDate"),
        "document_url": _CRISIL_DOC_BASE_URL + quote(file_name),
        "exact_match": exact is not None,
    }


def fetch_crisil_rationale_text(document_url: str) -> Optional[str]:
    """Fetch a CRISIL rationale document and return its plain text content
    (HTML tags stripped) — no auth required, plain public HTML file."""
    try:
        resp = _get(document_url)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Could not fetch CRISIL rationale document %s: %s", document_url, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def get_rating_rationale(company_name: str) -> dict[str, Any]:
    """
    Top-level entry point for the Finance agent's no-ticker fallback: find
    and fetch a company's most recent CRISIL rating rationale, if one
    exists. Never fabricates a rationale for an unrated/unfound company —
    returns an explicit status instead.
    """
    if not company_name or not company_name.strip():
        return {"status": "not_found", "source": "CRISIL Ratings", "company_name": company_name, "text": None}

    cache_key = f"rationale:{company_name.strip().lower()}"
    cached = _read_cache(cache_key)
    if cached:
        return cached

    match = search_crisil_rationale(company_name)
    if not match:
        result = {
            "status": "not_found",
            "source": "CRISIL Ratings",
            "company_name": company_name,
            "text": None,
            "document_url": None,
            "rating_date": None,
        }
        _write_cache(cache_key, result)
        return result

    text = fetch_crisil_rationale_text(match["document_url"])
    if not text:
        # Deliberately not cached — a fetch failure is more likely transient
        # (network blip) than "this company has no rationale", so the next
        # call should retry rather than replay a stale failure.
        return {
            "status": "fetch_failed",
            "source": "CRISIL Ratings",
            "company_name": match["company_name"],
            "text": None,
            "document_url": match["document_url"],
            "rating_date": match.get("rating_date"),
        }

    result = {
        "status": "ok",
        "source": "CRISIL Ratings",
        "company_name": match["company_name"],
        "heading": match.get("heading"),
        "rating_date": match.get("rating_date"),
        "document_url": match["document_url"],
        "text": text,
    }
    _write_cache(cache_key, result)
    return result
