"""
Exact-entity matching for sanctions / watchlist / PEP records.

A list record counts against the entity only when it is exactly the entity:
the full name must be identical after formatting-only normalisation (no
fuzzy / partial-token matching — "B R TRADING CO" must not match every
"... TRADING" entry), and when the record carries an Indian identifier it
must be one of the entity's. A same-name record with no identifier is
reported for the analyst and never scored.

Ported from the platform's VDD port (fino-backend
src/apps/credit-risk/vdd-report/pipeline/aml/free-screens.ts: canonicalName /
identifierVerdict, TaskHub DB-186) so both implementations draw the same line.
"""
from __future__ import annotations

import re
from typing import Iterable, Literal

GENERIC_STOP_WORDS = {
    "bank", "corp", "corporation", "ltd", "limited", "inc", "incorporated",
    "group", "holdings", "holding", "company", "co", "services", "industries",
    "industry", "state", "national", "trust", "financial", "finance", "the",
    "of", "and", "&", "ltd.", "inc.", "plc", "sa", "gmbh", "pvt", "private",
    "public", "enterprises", "international", "global",
}

# Abbreviations that are the same word, not a different name.
_LEGAL_FORM = {
    "co": "company",
    "coy": "company",
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
}

# Words that mark a business name: never strip a leading "Mr"/"Ms" from these.
_BUSINESS_WORDS = GENERIC_STOP_WORDS | {
    "trading", "traders", "trader", "steel", "steels", "agency", "agencies",
    "llp", "firm", "store", "stores", "works", "mills", "exports", "imports",
    "impex", "associates", "brothers", "bros", "sons",
}

HONORIFICS = {
    "shri", "smt", "smt.", "sri", "kumari", "km", "dr", "dr.", "mr", "mrs",
    "ms", "prof", "adv", "late", "thiru", "shrimati", "sardar", "haji",
}


def canonical_name(name: str) -> str:
    """The name reduced to formatting-independent form: lower case,
    punctuation and "M/s" / "Messrs" dropped, legal-form abbreviations
    expanded, runs of initials joined ("B. R." = "B R" = "BR"), and — for a
    person's name only — leading honorifics dropped. Two names are the same
    name iff these are equal."""
    s = (name or "").lower().replace("&", " and ")
    s = re.sub(r"\bm\s*/\s*s\b", " ", s)
    s = re.sub(r"\bmessrs\b", " ", s)
    toks = [_LEGAL_FORM.get(t, t) for t in re.sub(r"[^a-z0-9\s]", " ", s).split()]
    is_business = any(t in _BUSINESS_WORDS for t in toks)
    if toks and toks[0] == "ms" and is_business:  # "Ms" = M/s
        toks = toks[1:]
    if not is_business:
        while len(toks) > 1 and toks[0] in HONORIFICS:
            toks = toks[1:]
    out: list[str] = []
    initials = ""
    for t in toks:
        if re.fullmatch(r"[a-z]", t):
            initials += t
            continue
        if initials:
            out.append(initials)
        initials = ""
        out.append(t)
    if initials:
        out.append(initials)
    return " ".join(out)


def same_name(a: str, b: str) -> bool:
    """Exactly the same name (see canonical_name); never a partial or fuzzy match."""
    x = canonical_name(a)
    return bool(x) and x == canonical_name(b)


# PAN, GSTIN, CIN or LLPIN anywhere in a record's text.
_INDIAN_ID = re.compile(
    r"\b(?:\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]|[A-Z]{5}\d{4}[A-Z]|"
    r"[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}|[A-Z]{3}-\d{4})\b"
)

RecordVerdict = Literal["confirmed", "same_name_only", "different"]


def identifier_verdict(record_text: str, entity_ids: Iterable[str]) -> RecordVerdict:
    """Whether a record whose name is exactly the entity's IS the entity:
    `confirmed` when it carries one of the entity's identifiers, `different`
    when it carries only other Indian identifiers, `same_name_only` when it
    carries none (a namesake the analyst has to rule on; never scored)."""
    ids = {re.sub(r"\s+", "", i).upper() for i in entity_ids if i}
    found = _INDIAN_ID.findall((record_text or "").upper())
    if any(i in ids for i in found):
        return "confirmed"
    # A GSTIN embeds the PAN: an entity PAN inside a record's GSTIN is the entity.
    if any(len(v) == 10 and v in i for i in found for v in ids):
        return "confirmed"
    return "different" if found else "same_name_only"


def mask_pan(pan: str) -> str:
    """Keep first 2 + last 1 (ABCDE1234F -> ABXXXXXXXF), as the platform's reports do."""
    p = (pan or "").strip()
    return f"{p[:2]}{'X' * (len(p) - 3)}{p[-1:]}" if len(p) > 3 else p


def mask_din(din: str) -> str:
    """Keep first 2 + last 2 digits (07549147 -> 07XXXX47)."""
    d = (din or "").strip()
    return f"{d[:2]}{'X' * (len(d) - 4)}{d[-2:]}" if len(d) > 4 else d
