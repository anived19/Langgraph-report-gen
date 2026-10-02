"""
Minimal client for the internal Finoscale Data API — only the Probe42 calls
the credit report needs (company identity and the directors page, both keyed
by CIN). Ported from VDD-Automation's vdd/finoscale_api/client.py.

These are paid vendor calls: every response is cached to disk and reused for
up to `settings.finoscale_cache_max_age_days` (7 by default), and no path
serves an answer older than that except when the API is down and the expired
copy is the only one there is (logged).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests

from config import settings

logger = logging.getLogger(__name__)

_CACHE_DIR = settings.cache_dir / "finoscale_api"
_TIMEOUT = 30.0
_RETRY_AFTER_429 = 5.0


class FinoscaleAPIError(Exception):
    def __init__(self, status_code: int, message: str, path: str = ""):
        self.status_code = status_code
        self.message = message
        self.path = path
        super().__init__(f"[{status_code}] {path}: {message}")


def _cache_path(cache_key: str) -> Path:
    h = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
    safe = "".join(c if c.isalnum() else "_" for c in cache_key)[:60]
    return _CACHE_DIR / f"{safe}_{h}.json"


def _send(url: str, params: dict) -> requests.Response:
    """One retry each for the two transient failures seen on the PPE API: a
    read timeout (retried with double the timeout) and a 429 (retried once
    after a short pause; the API sends no Retry-After)."""
    headers = {"X-Api-Key": settings.finoscale_api_key or "", "Content-Type": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
    except requests.ReadTimeout:
        resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT * 2)
    if resp.status_code == 429:
        time.sleep(_RETRY_AFTER_429)
        resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
    return resp


def _cached_get(path: str, params: dict) -> Any:
    cache_file = _cache_path(f"{path}?{sorted(params.items())}")
    stale: Optional[dict] = None
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        age_days = (time.time() - float(cached.get("_fetched_at") or 0)) / 86400
        if age_days <= settings.finoscale_cache_max_age_days:
            return cached["data"]
        stale = cached

    if not settings.finoscale_api_key:
        raise FinoscaleAPIError(0, "FINOSCALE_API_KEY is not set", path)

    url = settings.finoscale_api_base.rstrip("/") + path
    try:
        resp = _send(url, params)
    except requests.RequestException:
        if stale is not None:
            logger.warning("Finoscale API unreachable — serving EXPIRED cache for %s", path)
            return stale["data"]
        raise
    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text}
    if resp.status_code >= 400:
        if stale is not None and resp.status_code in (429, 500, 502, 503, 504):
            logger.warning("Finoscale API %d — serving EXPIRED cache for %s", resp.status_code, path)
            return stale["data"]
        msg = data.get("message") if isinstance(data, dict) else str(data)
        raise FinoscaleAPIError(resp.status_code, msg or "request failed", path)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"_fetched_at": time.time(), "_path": path, "_params": params,
                                      "data": data}, indent=2), encoding="utf-8")
    return data


def _unwrap(raw: Any) -> Any:
    # Probe42 wraps its payload in a `data` envelope (confirmed against real PPE responses).
    return raw["data"] if isinstance(raw, dict) and "data" in raw else raw


def probe42_company(cin: str) -> Optional[dict]:
    """Company identity object (legal_name, cin, pan, status, ...)."""
    return _unwrap(_cached_get("/api/v1/data/fetch-comprehensive-details-by-field",
                               {"id": cin, "field": "company"}))


def probe42_directors(cin: str) -> Optional[dict]:
    """The directors page: authorized_signatories, director_network, director_shareholdings."""
    return _unwrap(_cached_get("/api/v1/data/fetch-comprehensive-details-by-page",
                               {"id": cin, "page": "directors"}))
