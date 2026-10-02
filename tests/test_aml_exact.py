"""Watchlist screens: exact name AND identifier to count; same-name without an
identifier goes to analyst review; unreachable is unscreened, never clean."""
import json

import pytest

import tools.aml_tools as aml
from schemas import AMLSeverity

CIN = "U74899DL1986PTC024176"

OFAC_XML = """<?xml version="1.0"?>
<sdnList xmlns="http://tempuri.org/sdnList.xsd">
  <sdnEntry><uid>1</uid><lastName>ACME TRADING COMPANY</lastName><sdnType>Entity</sdnType>
    <akaList><aka><lastName>ACME TRADERS FZE</lastName></aka></akaList></sdnEntry>
  <sdnEntry><uid>2</uid><lastName>ACME TRADING COMPANY LLC</lastName><sdnType>Entity</sdnType></sdnEntry>
  <sdnEntry><uid>3</uid><lastName>BETA STEEL PRIVATE LIMITED</lastName><sdnType>Entity</sdnType>
    <idList><id><idType>Registration ID</idType><idNumber>U74899DL1986PTC024176</idNumber></id></idList></sdnEntry>
  <sdnEntry><uid>4</uid><lastName>GAMMA LIMITED</lastName><sdnType>Entity</sdnType>
    <idList><id><idType>Registration ID</idType><idNumber>U11111MH2000PLC111111</idNumber></id></idList></sdnEntry>
</sdnList>"""


@pytest.fixture
def ofac(monkeypatch):
    monkeypatch.setattr(aml, "_cached_get", lambda url, **kw: OFAC_XML)


def test_near_name_never_matches(ofac):
    f = aml.screen_ofac_sdn("ACME TRADING CO PVT LTD", [CIN])
    assert f.severity == AMLSeverity.NONE
    assert f.analyst_review == []


def test_exact_name_without_identifier_is_review_only(ofac):
    f = aml.screen_ofac_sdn("Acme Trading Co.", [CIN])
    assert f.severity == AMLSeverity.NONE
    assert len(f.analyst_review) == 1 and "ACME TRADING COMPANY" in f.analyst_review[0]


def test_exact_name_and_identifier_counts(ofac):
    f = aml.screen_ofac_sdn("Beta Steel Pvt Ltd", [CIN])
    assert f.severity == AMLSeverity.HIGH
    assert "Exact name AND identifier" in f.finding_summary


def test_exact_name_with_other_identifier_is_a_different_entity(ofac):
    f = aml.screen_ofac_sdn("Gamma Ltd", [CIN])
    assert f.severity == AMLSeverity.NONE
    assert f.analyst_review == []
    assert "other identifiers" in f.finding_summary


def test_unreachable_source_is_unscreened(monkeypatch):
    monkeypatch.setattr(aml, "_cached_get", lambda url, **kw: None)
    f = aml.screen_ofac_sdn("Beta Steel Pvt Ltd", [CIN])
    assert f.severity == AMLSeverity.WATCH
    assert "unscreened" in f.finding_summary


def test_world_bank_exact_only(monkeypatch):
    payload = {"response": {"ZPROCSUPP": [{"SUPP_NAME": "Delta Infra Limited", "COUNTRY_NAME": "India"},
                                          {"SUPP_NAME": "Delta Infra Projects Limited", "COUNTRY_NAME": "India"}]}}
    monkeypatch.setattr(aml, "_cached_get", lambda url, **kw: json.dumps(payload))
    assert aml.screen_world_bank_debarred("Delta Infra Ltd").analyst_review
    assert not aml.screen_world_bank_debarred("Delta Infra").analyst_review


def test_opensanctions_without_key_is_unscreened(monkeypatch):
    class R:
        status_code = 401

    monkeypatch.delenv("OPENSANCTIONS_API_KEY", raising=False)
    monkeypatch.setattr(aml._SESSION, "get", lambda *a, **k: R())
    f = aml.screen_opensanctions("Beta Steel Pvt Ltd")
    assert f.severity == AMLSeverity.WATCH and "unscreened" in f.finding_summary


def test_sweep_passes_identifiers_to_every_screener(monkeypatch):
    seen = []

    def fake(name, ids=None):
        seen.append(ids)
        return aml.AMLFinding(entity_screened=name, source_name="x", finding_summary="ok",
                              severity=AMLSeverity.NONE, source_url="")

    for fn in ("screen_ofac_sdn", "screen_opensanctions", "screen_world_bank_debarred",
               "screen_un_sanctions", "screen_eu_sanctions", "screen_sec_fcpa"):
        monkeypatch.setattr(aml, fn, fake)
    out = aml.run_structured_aml_sweep("Beta Steel", "", entity_ids=[CIN])
    assert seen == [[CIN]] * 6
    assert len(out) == 8  # 6 screens + TI CPI + FATF


def test_adverse_media_drops_urls_the_model_invented(monkeypatch):
    raw = [{"title": "SEBI order", "url": "https://news.example/a", "content": "SEBI penalises Beta Steel"}]
    result = aml._MediaFilterResult(findings=[
        aml._MediaFinding(source_url="https://news.example/a", finding_summary="SEBI penalty", severity="Elevated"),
        aml._MediaFinding(source_url="https://made-up.example/b", finding_summary="Fraud", severity="High"),
    ])
    monkeypatch.setattr("credit_report.model.structured_call", lambda schema, prompt: result)
    findings = aml._filter_adverse_media_with_llm("Beta Steel", raw)
    assert [f.source_url for f in findings] == ["https://news.example/a"]
