"""
Shared chat model + rate limiter for every LLM call in a credit-report run.

Provider: Microsoft Foundry (DeepSeek-V4-Pro through the deployment's
OpenAI-compatible endpoint — data stays inside our Azure subscription) or
Gemini. Selection mirrors VDD-Automation's vdd/review/model.py: LLM_PROVIDER
if set, else a configured Foundry endpoint outranks a Gemini key.

Pacing: ONE InMemoryRateLimiter for the whole process — every agent and the
adverse-media filter go through the same bucket, because the quota is per
deployment, not per Python object. max_bucket_size=1 means no burst: calls
are spaced evenly at 60/RPM seconds. 429s that still happen (quota shared
with another process) are retried by the HTTP client, which honours
Retry-After; that loop is separate from the output-validation reruns in
credit_report/validation.py.

Reasoning mode stays off: nothing here enables thinking, and FOUNDRY_EXTRA_BODY
is the only way to send model-specific fields.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from config import settings

logger = logging.getLogger(__name__)

_DEFAULT_RPM = {"foundry": 20.0, "gemini": 12.0}


def select_provider() -> Optional[str]:
    explicit = (settings.llm_provider or "").strip().lower()
    foundry_ok = bool(settings.foundry_endpoint and settings.foundry_api_key)
    if explicit in ("foundry", "azure"):
        return "foundry" if foundry_ok else None
    if explicit in ("gemini", "google", "google_genai"):
        return "gemini" if settings.gemini_api_key else None
    if explicit:
        raise ValueError(f"Unrecognized LLM_PROVIDER={explicit!r} — expected 'foundry' or 'gemini'")
    if foundry_ok:
        return "foundry"
    if settings.gemini_api_key:
        return "gemini"
    return None


def requests_per_minute(provider: Optional[str] = None) -> float:
    provider = provider or select_provider() or "foundry"
    return float(settings.llm_rpm or _DEFAULT_RPM.get(provider, 12.0))


_rate_limiter: Optional[InMemoryRateLimiter] = None


def shared_rate_limiter() -> InMemoryRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_minute() / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _rate_limiter


def build_model() -> BaseChatModel:
    provider = select_provider()
    if provider == "foundry":
        if not settings.foundry_deployment:
            raise ValueError("FOUNDRY_ENDPOINT/FOUNDRY_API_KEY are set but FOUNDRY_DEPLOYMENT is not — set it to "
                             "the deployment name from the Foundry portal (e.g. DeepSeek-V4-Pro).")
        key = settings.foundry_api_key
        # Azure authenticates with an `api-key` header; the OpenAI client sends
        # `Authorization: Bearer`. Sending both covers the /openai/v1 route
        # (accepts either) and the Model Inference route (api-key only).
        kwargs: dict[str, Any] = {
            "api_key": key,
            "base_url": settings.foundry_endpoint.rstrip("/"),
            "default_headers": {"api-key": key},
            "max_tokens": settings.llm_max_output_tokens,
            "max_retries": settings.llm_max_http_retries,
            "temperature": 0,
            "rate_limiter": shared_rate_limiter(),
        }
        if settings.foundry_api_version:
            kwargs["default_query"] = {"api-version": settings.foundry_api_version}
        if settings.foundry_extra_body:
            kwargs["extra_body"] = json.loads(settings.foundry_extra_body)
        return init_chat_model(settings.foundry_deployment, model_provider="openai", **kwargs)
    if provider == "gemini":
        return init_chat_model(
            settings.gemini_model,
            model_provider="google_genai",
            api_key=settings.gemini_api_key,
            rate_limiter=shared_rate_limiter(),
            max_retries=settings.llm_max_http_retries,
        )
    raise RuntimeError("No LLM configured — set FOUNDRY_ENDPOINT/FOUNDRY_API_KEY/FOUNDRY_DEPLOYMENT "
                       "(or GEMINI_API_KEY) in .env.")


class _FinishWithAnswer(AgentMiddleware):
    """Guarantees an agent run ends with its structured answer. With a
    ToolStrategy response_format every turn is bound with tool_choice="any",
    so the model can never answer in prose — calling the answer tool is the
    only exit. A model that keeps calling tools would otherwise never take it
    (seen on VDD runs with an open-weight model). On the last allowed call
    this withholds every other tool and appends a one-line nudge for that
    call only. Same mechanism as VDD-Automation's vdd/review/graph.py."""

    def __init__(self, model_calls: int):
        super().__init__()
        self.model_calls = model_calls
        self.calls = 0

    def wrap_model_call(self, request, handler):
        self.calls += 1
        if self.calls >= self.model_calls:
            logger.info("Model call %d/%d — forcing the structured answer (other tools withheld).",
                        self.calls, self.model_calls)
            nudge = HumanMessage(content="Your tool budget is used up. Give your final structured answer now, "
                                         "using only the evidence already in this conversation.")
            request = request.override(tools=[], messages=list(request.messages) + [nudge])
        return handler(request)


def build_agent(tools: list, system_prompt: str, response_format: type):
    """A ReAct agent with a hard cap on model calls whose last call is forced
    to be the structured answer. A fresh middleware per agent — it counts."""
    calls = max(2, settings.agent_max_model_calls)
    return create_agent(
        model=build_model(),
        tools=tools,
        system_prompt=system_prompt,
        response_format=ToolStrategy(response_format),
        middleware=[
            _FinishWithAnswer(calls),
            # Backstop only: reached if the forced answer itself fails schema
            # validation twice (ToolStrategy re-prompts on a bad payload).
            ModelCallLimitMiddleware(run_limit=calls + 2, exit_behavior="end"),
        ],
    )


def structured_call(schema: type, prompt: str):
    """One model call returning a `schema` instance, via function calling
    (supported by both providers; avoids depending on each deployment's
    JSON-mode support). Goes through the same shared rate limiter."""
    model = build_model().with_structured_output(schema, method="function_calling")
    return model.invoke(prompt)
