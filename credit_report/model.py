"""
Shared Gemini model + rate limiter for every credit_report agent.

Testing-phase pacing: ONE shared InMemoryRateLimiter caps requests at
12/minute across ALL agents combined, not per agent. Gemini's quota is
per-project/per-model, not per Python object — a separate limiter per agent
would undercount real API load the moment two agents run close together
(and Business/Banking's completeness now genuinely depends on them calling
search_web_news, per the prompt fix in this same change — more tool calls,
more Gemini calls, more reason to pace centrally). Same 12 RPM target the
old equity pipeline hand-rolls in harness/gemini_retry.py — implemented
here via LangChain's built-in primitive instead, per the architecture doc's
own suggestion (Section 13) that this retires the need for that custom
plumbing.

max_bucket_size=1 means no burst allowance: every call waits its turn at a
strict ~5-second minimum spacing, rather than allowing a burst of several
quick calls followed by a long wait. This mirrors the existing harness's
_pace_gemini_call() behavior exactly (5.0s fixed interval) rather than a
bursty token-bucket, which is the more conservative, quota-safe choice.
"""
from __future__ import annotations

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings

_rate_limiter = InMemoryRateLimiter(
    requests_per_second=12 / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def build_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        rate_limiter=_rate_limiter,
    )
