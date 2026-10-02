"""
Centralized configuration for the credit-report pipeline.

Every other module reads settings from here rather than calling
os.environ directly — one place to see the full config surface, and
pydantic-settings validates types/required fields at startup instead of
failing deep inside a pipeline run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider (credit_report/model.py) ---
    # Precedence when several are configured: LLM_PROVIDER if set, else
    # foundry > gemini — same rule as VDD-Automation's vdd/review/model.py.
    llm_provider: Optional[str] = None  # "foundry" | "gemini"

    # Microsoft Foundry: an open-weight model (DeepSeek-V4-Pro) hosted inside
    # our own Azure subscription, reached through its OpenAI-compatible
    # /openai/v1 endpoint. See VDD-Automation's .env.example for the
    # deployment checklist that keeps the data-residency guarantees true.
    foundry_endpoint: Optional[str] = None
    foundry_api_key: Optional[str] = None
    foundry_deployment: Optional[str] = None
    foundry_api_version: Optional[str] = None
    # JSON of model-specific request fields from the model card, e.g. a
    # thinking switch. Unset = the deployment's default. Reasoning mode stays
    # off for this pipeline (it produced empty and hallucinated sections on
    # VDD runs), so leave this unset unless deliberately re-testing it.
    foundry_extra_body: Optional[str] = None

    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.6-flash"

    # Requests per minute across EVERY model call in a run (all agents plus
    # the adverse-media filter). Unset = 20 for Foundry (the deployment's
    # quota), 12 for Gemini's free tier.
    llm_rpm: Optional[float] = None
    # Per-call output cap. A reasoning model's thinking counts against it.
    llm_max_output_tokens: int = 8192
    # HTTP-level retries for 429 / 5xx (the OpenAI client honours Retry-After).
    # Separate from the validation reruns below: "the API throttled me" and
    # "the output was bad" are different loops. The Foundry deployment also
    # caps tokens (20k/min): a long prompt can hit that before the RPM limit,
    # and these retries are what absorb it.
    llm_max_http_retries: int = 6
    # Model calls one agent may make in one attempt; the last one is forced to
    # be the structured answer (credit_report/model.py:build_agent).
    agent_max_model_calls: int = 8
    # Reruns of one agent after its output fails the code checks in
    # credit_report/validation.py. Clamped to 0..2.
    validation_max_retries: int = 1

    # --- Web search (Tavily). Optional: without it, search_web_news returns
    # nothing and the agents' completeness ratios show the gap. ---
    tavily_api_key: Optional[str] = None

    # --- Finoscale Data API (Probe42 MCA records for directors). Paid per
    # call: every response is cached for finoscale_cache_max_age_days. ---
    finoscale_api_key: Optional[str] = None
    finoscale_api_base: str = "https://api-ppe.finoscale.ai"
    finoscale_cache_max_age_days: float = 7.0

    # --- PDF ---
    pdf_engine: str = "weasyprint"  # "weasyprint" | "xhtml2pdf"

    # --- Paths ---
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("cache")

    # --- Cache ---
    cache_ttl_hours: int = 6

    # --- Outlook window ---
    # Number of months used for the price trend window and outlook section.
    # Changing this here is the only code change needed — it flows through
    # finance_tools and schemas automatically.
    outlook_months: int = 6

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
