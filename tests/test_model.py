import pytest

from config import settings
from credit_report import model


@pytest.fixture
def clean(monkeypatch):
    for k in ("llm_provider", "foundry_endpoint", "foundry_api_key", "foundry_deployment", "foundry_api_version",
              "foundry_extra_body", "gemini_api_key", "llm_rpm"):
        monkeypatch.setattr(settings, k, None)
    monkeypatch.setattr(model, "_rate_limiter", None)


def test_foundry_outranks_gemini(clean, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "g")
    assert model.select_provider() == "gemini"
    monkeypatch.setattr(settings, "foundry_endpoint", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setattr(settings, "foundry_api_key", "k")
    assert model.select_provider() == "foundry"
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    assert model.select_provider() == "gemini"


def test_rpm_defaults(clean, monkeypatch):
    assert model.requests_per_minute("foundry") == 20
    assert model.requests_per_minute("gemini") == 12
    monkeypatch.setattr(settings, "llm_rpm", 30)
    assert model.requests_per_minute("foundry") == 30


def test_foundry_model_config(clean, monkeypatch):
    monkeypatch.setattr(settings, "foundry_endpoint", "https://r.services.ai.azure.com/openai/v1/")
    monkeypatch.setattr(settings, "foundry_api_key", "secret")
    monkeypatch.setattr(settings, "foundry_deployment", "DeepSeek-V4-Pro")
    m = model.build_model()
    assert m.model_name == "DeepSeek-V4-Pro"
    assert str(m.openai_api_base) == "https://r.services.ai.azure.com/openai/v1"
    assert m.default_headers["api-key"] == "secret"
    assert m.temperature == 0 and m.max_tokens == settings.llm_max_output_tokens
    assert m.max_retries == settings.llm_max_http_retries
    # one limiter for the whole process, at 20 RPM
    assert m.rate_limiter is model.build_model().rate_limiter
    assert m.rate_limiter.requests_per_second == pytest.approx(20 / 60)
    assert not getattr(m, "extra_body", None)  # reasoning mode not switched on


def test_foundry_requires_deployment(clean, monkeypatch):
    monkeypatch.setattr(settings, "foundry_endpoint", "https://r/openai/v1")
    monkeypatch.setattr(settings, "foundry_api_key", "k")
    with pytest.raises(ValueError, match="FOUNDRY_DEPLOYMENT"):
        model.build_model()


def test_no_provider_raises(clean):
    with pytest.raises(RuntimeError, match="No LLM configured"):
        model.build_model()
