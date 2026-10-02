"""
Directors node — deterministic, no LLM.

Needs the entity's CIN (passed with --cin; in production it comes from the
platform's company record). With it:

1. Probe42 (MCA records, via the Finoscale Data API — paid, 7-day cache):
   the company identity (legal name, PAN) and the directors page (current
   and ceased directors / KMP, and each director's other boards).
2. Per current director: designation, tenure, DIN status, other active
   boards, and linked companies under CIRP / struck off / in liquidation.
3. A PEP screen per person (tools/pep_tools.py). Exact name only, and never
   scored — PEP sources carry no PAN/DIN.

It also hands the entity's identifiers (CIN, PAN) to the Hygiene node for
exact-entity watchlist matching, and its legal name for screening. DIN/PAN
never enter an LLM prompt; only names, designations, dates and counts go
into the Business agent's context (see board_context below).

No web search, by design: director data comes from registry records only.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Optional

from credit_report.schemas import DirectorProfile, DirectorsResult
from tools.finoscale_api import probe42_company, probe42_directors
from tools.name_match import mask_din
from tools.pep_tools import screen_person_pep

logger = logging.getLogger(__name__)

CIN_RE = re.compile(r"^[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}$")
_MAX_PEP_SCREENS = 25
_FLAG_STATUS_WORDS = ("strike", "struck", "liquidat", "dormant", "dissolved", "insolven")


def _parse_date(s: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10]) if s else None
    except ValueError:
        return None


def _as_list(v: Any) -> list[dict]:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _network_by_din(network: list[dict]) -> dict[str, dict]:
    return {str(n.get("din")): n.get("network") or {} for n in network if n.get("din")}


def _associations(network: dict, own_cin: str) -> tuple[int, list[str]]:
    """(other boards this person currently sits on, flagged linked entities)."""
    active, flagged = 0, []
    entries = [(c, c.get("cin"), c.get("company_status")) for c in _as_list(network.get("companies"))]
    entries += [(l, l.get("llpin"), l.get("status")) for l in _as_list(network.get("llps"))]
    for rec, ident, status in entries:
        if ident and ident == own_cin:
            continue
        if not rec.get("date_of_cessation") and (status or "").upper() == "ACTIVE":
            active += 1
        status_l = (status or "").lower()
        reasons = []
        if rec.get("cirp_status"):
            reasons.append(f"CIRP: {rec['cirp_status']}")
        if any(w in status_l for w in _FLAG_STATUS_WORDS):
            reasons.append(f"status: {status}")
        if reasons:
            role = "former " if rec.get("date_of_cessation") else ""
            flagged.append(f"{rec.get('legal_name') or ident} ({role}{rec.get('designation') or 'director'}; "
                           f"{', '.join(reasons)})")
    return active, flagged


def build_directors_result(cin: str, company: dict, page: dict, today: Optional[date] = None) -> DirectorsResult:
    today = today or date.today()
    signatories = _as_list(page.get("authorized_signatories"))
    network = _network_by_din(_as_list(page.get("director_network")))

    current: list[DirectorProfile] = []
    ceased_recent: list[dict[str, Any]] = []
    ceased_total = 0
    for s in signatories:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        ceased_on = _parse_date(s.get("date_of_cessation"))
        if ceased_on:
            ceased_total += 1
            if (today - ceased_on).days <= 730:
                ceased_recent.append({"name": name, "designation": s.get("designation"),
                                      "ceased_on": ceased_on.isoformat()})
            continue
        appointed = _parse_date(s.get("date_of_appointment"))
        other_boards, flagged = _associations(network.get(str(s.get("din")), {}), cin)
        current.append(DirectorProfile(
            name=name,
            designation=s.get("designation"),
            appointed_on=appointed.isoformat() if appointed else None,
            tenure_years=round((today - appointed).days / 365.25, 1) if appointed else None,
            din_masked=mask_din(str(s.get("din") or "")) or None,
            din_status=s.get("din_status"),
            other_directorships=other_boards,
            flagged_associations=flagged,
        ))

    for d in current[:_MAX_PEP_SCREENS]:
        d.pep = screen_person_pep(d.name)
    for d in current[_MAX_PEP_SCREENS:]:
        d.pep = {"name": d.name, "status": "unscreened", "matches": [],
                 "note": f"Not screened — PEP screening is capped at {_MAX_PEP_SCREENS} people per report."}

    return DirectorsResult(
        status="ok",
        legal_name=company.get("legal_name"),
        current=current,
        ceased_last_24_months=sorted(ceased_recent, key=lambda c: c["ceased_on"], reverse=True),
        ceased_total=ceased_total,
        pep_analyst_review=sum(1 for d in current if d.pep.get("status") == "analyst_review"),
        pep_unscreened=sum(1 for d in current if d.pep.get("status") == "unscreened"),
    )


def run_directors_check(state: dict[str, Any]) -> dict[str, Any]:
    cin = (state.get("cin") or "").strip().upper()
    if not cin:
        result = DirectorsResult(status="not_available",
                                 reason="No CIN supplied — run with --cin to fetch directors from MCA records.")
        return {"directors_result": result.model_dump(), "entity_ids": [],
                "issues": [{"stage": "directors", "severity": "info", "message": result.reason}]}
    if not CIN_RE.match(cin):
        result = DirectorsResult(status="not_available", reason=f"{cin!r} is not a valid CIN.")
        return {"directors_result": result.model_dump(), "entity_ids": [],
                "issues": [{"stage": "directors", "severity": "warning", "message": result.reason}]}

    try:
        company = probe42_company(cin) or {}
        page = probe42_directors(cin) or {}
    except Exception as exc:
        logger.warning("Probe42 directors fetch failed for %s: %s", cin, exc)
        result = DirectorsResult(status="error", reason=f"Probe42 directors fetch failed: {exc}")
        return {"directors_result": result.model_dump(), "entity_ids": [cin],
                "issues": [{"stage": "directors", "severity": "warning", "message": result.reason}]}

    result = build_directors_result(cin, company, page)
    issues: list[dict[str, Any]] = []
    if not result.current:
        issues.append({"stage": "directors", "severity": "warning",
                       "message": f"Probe42 returned no current directors for {cin}."})
    if result.pep_unscreened:
        issues.append({"stage": "directors", "severity": "warning",
                       "message": f"PEP screen could not run for {result.pep_unscreened} director(s) — not counted as clean."})

    entity_ids = [i for i in (cin, company.get("pan")) if i]
    update: dict[str, Any] = {"directors_result": result.model_dump(), "entity_ids": entity_ids, "issues": issues}
    if result.legal_name:
        update["legal_name"] = result.legal_name
    return update


def board_context(directors_result: Optional[dict]) -> Optional[str]:
    """Registry facts for the Business agent's dimension 1 — names, roles,
    dates and counts only. No DIN/PAN, and no PEP results (those are never
    scored, so they must not reach the scoring LLM)."""
    if not directors_result or directors_result.get("status") != "ok":
        return None
    lines = []
    for d in directors_result.get("current", []):
        line = f"- {d['name']} — {d.get('designation') or 'Director'}"
        if d.get("appointed_on"):
            line += f", appointed {d['appointed_on']}"
        line += f"; {d.get('other_directorships', 0)} other active board seat(s)"
        if d.get("din_status") and d["din_status"].lower() != "approved":
            line += f"; DIN status: {d['din_status']}"
        if d.get("flagged_associations"):
            line += "; linked entities flagged in MCA records: " + "; ".join(d["flagged_associations"][:3])
        lines.append(line)
    ceased = directors_result.get("ceased_last_24_months") or []
    out = (f"Board and key managerial personnel ({directors_result.get('source')}), "
           f"{len(lines)} current:\n" + "\n".join(lines))
    if ceased:
        out += (f"\nCessations in the last 24 months ({len(ceased)}): "
                + "; ".join(f"{c['name']} ({c.get('designation') or 'director'}, {c['ceased_on']})" for c in ceased))
    else:
        out += "\nNo director/KMP cessations in the last 24 months."
    return out
