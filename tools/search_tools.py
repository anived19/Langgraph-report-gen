"""
Tavily web search wrapper — the agents' only web tool.

Returns a list of dicts (title/url/content/score) rather than a Tavily-specific
object. Without TAVILY_API_KEY every search returns [] (logged once): each
agent's completeness ratio already counts a search that found nothing, so the
report shows the gap instead of the run failing at import.
"""
from __future__ import annotations

import logging

from tavily import TavilyClient

from config import settings
from utils.retry import retry_on_transient_error

logger = logging.getLogger(__name__)

_client = TavilyClient(api_key=settings.tavily_api_key) if settings.tavily_api_key else None
_warned_no_key = False


@retry_on_transient_error(max_attempts=3)
def search_web_news(
    query: str,
    ticker: str = "",
    depth: str = "basic",
    max_results: int = 5,
) -> list[dict]:
    """
    Search the live web for news/sentiment relevant to `query`.
    `depth` may be 'basic' (1 credit) or 'advanced' (2 credits).
    """
    global _warned_no_key
    if _client is None:
        if not _warned_no_key:
            logger.warning("TAVILY_API_KEY is not set — web search is disabled for this run.")
            _warned_no_key = True
        return []

    max_results = max(1, min(max_results, 10))
    search_depth = "advanced" if depth.lower() == "advanced" else "basic"

    full_query = query.strip()
    if ticker and ticker.upper() not in full_query.upper():
        full_query = f"{ticker} {full_query}"

    response = _client.search(
        query=full_query,
        search_depth=search_depth,
        topic="finance",
        max_results=max_results,
        include_answer=False,
    )

    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
            }
        )

    if not results:
        logger.warning("Tavily returned no results for query: %r", full_query)

    return results
