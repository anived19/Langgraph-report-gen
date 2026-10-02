"""End-to-end graph run with a scripted tool-calling model and stubbed data
sources: no network, no API cost. Exercises the real wiring — ToolStrategy
structured output, the model-call cap, a validation rerun with feedback,
identifiers bound in code, the namesake list kept away from the scoring
LLM, and the report template."""
import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import credit_report.company_identity as ci
from config import settings
from credit_report import model as model_mod
from credit_report.agents import business, directors, finance, hygiene
from credit_report.graph import build_credit_report_graph
from credit_report.reporting.pdf import render_credit_report_html
from credit_report.schemas import CreditReport
from schemas import AMLFinding, AMLSeverity, BankClassification

CIN = "U12345MH2001PLC123456"
GOOD_SUMMARY = "Clean across every watchlist screened; no adverse media found in this cycle."
DIM = {"assessment": "Board stable per MCA records (Probe42).", "grounding": "grounded", "source": "MCA records (Probe42)"}

ANSWERS = {
    "FinanceLLMAssessment": {"raw_score": 72, "summary": "P/E of 24.53 with ROE of 47.74% and debt/equity of 10.21.",
                             "six_month_outlook": "Price 3,456.78 against an analyst mean target of 3,800.00; RSI-14 at 61.2."},
    "HygieneLLMAssessment": {"raw_score": 90, "summary": GOOD_SUMMARY},
    "BusinessLLMAssessment": {"raw_score": 65, "summary": "Established operator with a stable board and no governance flags.",
                              "promoter_management_pedigree": DIM, "business_model_moat": DIM,
                              "operating_trajectory": DIM, "industry_macro_headwinds": DIM},
}
FIRST_TOOLS = ["get_price_snapshot", "get_valuation_multiples", "get_fundamentals", "get_quarterly_financials",
               "get_technicals", "get_ownership", "run_structured_aml_sweep"]


def _tool_name(t: Any) -> str:
    if isinstance(t, dict):
        return t.get("name") or t.get("function", {}).get("name") or t.get("title", "")
    return getattr(t, "name", None) or getattr(t, "__name__", "")


class ScriptedModel(BaseChatModel):
    """Calls every data tool it is bound to once, then answers. The first
    Hygiene answer is deliberately too short, to force a validation rerun."""
    tool_names: list[str] = []
    log: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tool_names": [_tool_name(t) for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        done = {m.name for m in messages if isinstance(m, ToolMessage)}
        for name in FIRST_TOOLS:
            if name in self.tool_names and name not in done:
                args = {} if name == "run_structured_aml_sweep" else {"ticker": "BETA.NS"}
                return self._call(name, args)
        answer = next(n for n in self.tool_names if n in ANSWERS)
        args = dict(ANSWERS[answer])
        if answer == "HygieneLLMAssessment":
            rejected = any("REJECTED" in str(m.content) for m in messages)
            args["summary"] = GOOD_SUMMARY if rejected else "Clean."
        self.log.append(answer)
        return self._call(answer, args)

    def _call(self, name, args):
        msg = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{len(self.log)}_{name}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _fake(name, result):
    def fn(ticker: str) -> dict:
        """fake data tool"""
        return result
    fn.__name__ = name
    return fn


@pytest.fixture
def offline(monkeypatch, tmp_path):
    scripted = ScriptedModel()
    monkeypatch.setattr(model_mod, "build_model", lambda: scripted)
    monkeypatch.setattr(settings, "validation_max_retries", 1)

    monkeypatch.setattr(ci, "resolve_entity", lambda q: [{"ticker": "BETA.NS", "name": "Beta Steel Limited", "confidence": 0.95}])
    monkeypatch.setattr(ci, "_listed_long_name", lambda t: "Beta Steel Limited")
    monkeypatch.setattr(ci, "classify_bank", lambda n: BankClassification(entity_screened=n, is_bank=False,
                                                                            confidence="checked_not_found", source_url=""))
    monkeypatch.setattr(directors, "probe42_company", lambda cin: {"legal_name": "BETA STEEL LIMITED", "pan": "AAACB1234C"})
    monkeypatch.setattr(directors, "probe42_directors", lambda cin: {"authorized_signatories": [
        {"name": "ARUN KUMAR MEHTA", "din": "01234567", "designation": "Managing Director",
         "din_status": "Approved", "date_of_appointment": "2010-04-01", "date_of_cessation": None}]})
    monkeypatch.setattr(directors, "screen_person_pep", lambda n: {"name": n, "status": "analyst_review",
                                                                   "matches": ["MyNeta/ECI: 'Arun Kumar Mehta' (XYZ, C1, 2019)"]})

    data = {"pe_ratio": 24.53, "roe": 0.4774, "debt_to_equity": 10.21, "current_price": 3456.78,
            "analyst_target_mean": 3800.0, "rsi_14": 61.2}
    monkeypatch.setattr(finance, "_BASE_TOOLS", tuple(_fake(n, data) for n in FIRST_TOOLS[:6]))

    sweep_calls = []

    def sweep(entity_name, ticker="", entity_ids=None):
        sweep_calls.append((entity_name, entity_ids))
        return [AMLFinding(entity_screened=entity_name, source_name="OFAC SDN List",
                           finding_summary="No record of this entity on OFAC SDN List.", severity=AMLSeverity.NONE,
                           source_url="", analyst_review=["OFAC SDN List: same-name record 'BETA STEEL LTD' — analyst to rule out"]
                           ).model_dump()]
    monkeypatch.setattr(hygiene, "run_structured_aml_sweep", sweep)

    def search_web_news(query: str) -> list:
        """fake search"""
        return []
    monkeypatch.setattr(business, "_BASE_TOOLS", (search_web_news,))
    return scripted, sweep_calls


def test_full_graph_offline(offline):
    scripted, sweep_calls = offline
    result = build_credit_report_graph().invoke({"company_reference": "Beta Steel", "cin": CIN})
    report = CreditReport.model_validate(result["report"])

    # Hygiene was rerun once with feedback, then passed.
    assert scripted.log.count("HygieneLLMAssessment") == 2
    assert [t["attempt"] for t in result["traces"] if t["agent"] == "hygiene"] == [1, 2]
    assert any("Passed output checks on attempt 2" in i["message"] for i in report.issues)
    assert report.hygiene_result["summary"] == GOOD_SUMMARY

    # Identifiers bound in code; the registry legal name is what gets screened.
    assert sweep_calls[0] == ("BETA STEEL LIMITED", [CIN, "AAACB1234C"])
    # The scoring LLM never saw the namesake note, the report does.
    hygiene_tool_msgs = [m for t in result["traces"] if t["agent"] == "hygiene"
                         for m in t["messages"] if m["type"] == "tool_result"]
    assert hygiene_tool_msgs and all("analyst_review" not in str(m["content"]) for m in hygiene_tool_msgs)
    assert any("BETA STEEL LTD" in a for a in report.analyst_review)
    assert any("PEP — director ARUN KUMAR MEHTA" in a for a in report.analyst_review)

    # No identifier ever reached an LLM prompt.
    prompts = json.dumps(result["traces"])
    for secret in (CIN, "AAACB1234C", "01234567"):
        assert secret not in prompts

    # Finance figures were all grounded: no validation issues for finance/business.
    assert not [i for i in report.issues if i["stage"] in ("finance_agent", "business_agent")
                and "failed checks" in i["message"]]
    assert report.scores.agents_averaged == 3

    html = render_credit_report_html(report)
    assert "Directors &amp; Key Managerial Personnel" in html and "01XXXX67" in html
    assert "Analyst Review" in html
