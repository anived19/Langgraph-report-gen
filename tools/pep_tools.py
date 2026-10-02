"""
PEP (politically exposed person) screening for an entity's directors.

Two free, key-free India sources, ported from VDD-Automation's
vdd/aml/india_legal.py:

  * MyNeta / ADR — Election Commission candidate affidavits for every Lok
    Sabha, Rajya Sabha, State Assembly and Legislative Council candidate, i.e.
    the core of the Indian PEP population, with a declared-criminal-cases
    flag. REQUIRED: if it can't be reached the person is "unscreened".
  * Wikidata Query Service — office-holder records (appointed officials,
    judges, bureaucrats MyNeta misses). Best-effort: it throttles per IP, so a
    failure is disclosed as a coverage gap rather than voiding the screen.

OpenSanctions' PEP API needs a paid key and its bulk dump is CC-BY-NC, so
neither is used.

Matching is exact-name only (tools/name_match.py). No PEP source carries a
PAN or DIN, so an exact-name hit can never be confirmed as this person: it is
listed for the analyst to rule on and is never scored.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Optional

import requests

from config import settings
from tools.name_match import same_name

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "credit-report-agent/1.0 (Finoscale credit assessment; compliance screening)",
})
_CACHE_DIR = settings.cache_dir / "pep"
_CACHE_TTL_HOURS = 24
_TIMEOUT = 45

_MYNETA_URL = "https://www.myneta.info/search_myneta.php"
_MYNETA_LABEL = "MyNeta/ECI candidate affidavits"
_WDQS_URL = "https://query.wikidata.org/sparql"
_WDQS_LABEL = "Wikidata office-holders"
_WDQS_QUERY = """
SELECT ?pLabel ?posLabel ?start WHERE {
  SERVICE wikibase:mwapi {
    bd:serviceParam wikibase:api "EntitySearch" ; wikibase:endpoint "www.wikidata.org" ;
                    mwapi:search %s ; mwapi:language "en" .
    ?p wikibase:apiOutputItem mwapi:item .
  }
  ?p wdt:P31 wd:Q5 ; p:P39 ?st .
  ?st ps:P39 ?pos .
  OPTIONAL { ?st pq:P580 ?start }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
} LIMIT 25
"""
_WDQS_MIN_INTERVAL = 1.5  # seconds between WDQS queries — it 429s on back-to-back calls
_WDQS_ATTEMPTS = 2
_WDQS_TIMEOUT = 20
_wdqs_last_call = 0.0


def _cache_file(kind: str, name: str):
    h = hashlib.sha256(f"{kind}:{name.lower().strip()}".encode()).hexdigest()[:24]
    return _CACHE_DIR / f"{kind}_{h}.json"


def _cached(kind: str, name: str, fetch) -> list[dict]:
    path = _cache_file(kind, name)
    if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < _CACHE_TTL_HOURS:
        return json.loads(path.read_text(encoding="utf-8"))
    rows = fetch(name)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _myneta_search(name: str) -> list[dict]:
    """Server-rendered HTML table: Candidate | Party | Constituency | Election | Criminal(Y/N)."""
    resp = _SESSION.get(_MYNETA_URL, params={"q": name}, timeout=_TIMEOUT)
    resp.raise_for_status()
    html = resp.text
    if "results found" not in html and "Searched for" not in html:
        raise ValueError("MyNeta returned an unparseable page")
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = [re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S | re.I)]
        cells = [re.sub(r"\s+", " ", c).strip() for c in cells]
        if len(cells) >= 5 and cells[0]:
            rows.append({"name": cells[0], "party": cells[1], "constituency": cells[2],
                         "election": cells[3], "criminal": cells[4].upper().startswith("Y")})
    return rows


def _wikidata_officeholders(name: str) -> list[dict]:
    """P39 'position held' on a human, throttled and retried once — WDQS
    requires a descriptive User-Agent and 429s on back-to-back queries."""
    global _wdqs_last_call
    query = _WDQS_QUERY % json.dumps(name)
    headers = {"Accept": "application/sparql-results+json"}
    resp, last_exc = None, None
    for attempt in range(_WDQS_ATTEMPTS):
        wait = _WDQS_MIN_INTERVAL - (time.time() - _wdqs_last_call)
        if wait > 0:
            time.sleep(wait)
        _wdqs_last_call = time.time()
        try:
            resp = _SESSION.get(_WDQS_URL, params={"query": query, "format": "json"},
                                headers=headers, timeout=_WDQS_TIMEOUT)
        except requests.RequestException as exc:
            last_exc, resp = exc, None
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code != 429:
            break
        time.sleep(min(float(resp.headers.get("Retry-After") or (2 * (attempt + 1))), 10))
    if resp is None:
        raise last_exc if last_exc else RuntimeError("WDQS unreachable")
    resp.raise_for_status()
    bindings = resp.json().get("results", {}).get("bindings", [])
    return [{"name": b.get("pLabel", {}).get("value", ""),
             "position": b.get("posLabel", {}).get("value", ""),
             "since": b.get("start", {}).get("value", "")[:10]} for b in bindings]


def screen_person_pep(name: str) -> dict[str, Any]:
    """PEP screen for one natural person. Returns
    {name, status: 'no_match' | 'analyst_review' | 'unscreened', matches,
    sources_checked, sources_failed, note}. Never a score input."""
    matches: list[str] = []
    checked: list[str] = []
    failed: list[str] = []
    myneta_ok = False

    try:
        for r in _cached("myneta", name, _myneta_search):
            if same_name(name, r["name"]):
                matches.append(f"MyNeta/ECI: '{r['name']}' ({r['party']}, {r['constituency']}, {r['election']}"
                               + (", criminal cases declared" if r["criminal"] else "") + ")")
        checked.append(_MYNETA_LABEL)
        myneta_ok = True
    except Exception as exc:
        logger.warning("MyNeta PEP screen failed for %r: %s", name, exc)
        failed.append(f"{_MYNETA_LABEL} ({type(exc).__name__})")

    try:
        for r in _cached("wikidata", name, _wikidata_officeholders):
            if r["position"] and same_name(name, r["name"]):
                matches.append(f"Wikidata: '{r['name']}' holds/held '{r['position']}'"
                               + (f" since {r['since']}" if r["since"] else ""))
        checked.append(_WDQS_LABEL)
    except Exception as exc:
        # Best-effort tier: a throttled WDQS is a disclosed coverage gap, not a failed screen.
        logger.info("Wikidata PEP enrichment unavailable for %r (%s)", name, type(exc).__name__)
        failed.append(f"{_WDQS_LABEL} ({type(exc).__name__})")

    if matches:
        status = "analyst_review"
        note = ("Exact-name record(s) found; PEP sources carry no PAN/DIN, so the analyst must confirm whether "
                "this is the same person. Not scored.")
    elif not myneta_ok:
        status = "unscreened"
        note = f"Not screened — {', '.join(failed)} unavailable this run. Not counted as clean."
    else:
        status = "no_match"
        note = f"No exact-name record across {', '.join(checked)}."
        if failed:
            note += f" Coverage gap: {', '.join(failed)} did not run."
    return {"name": name, "status": status, "matches": list(dict.fromkeys(matches)),
            "sources_checked": checked, "sources_failed": failed, "note": note}
