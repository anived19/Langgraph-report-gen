"""PEP screen (exact name, never scored) and the Directors node, on a
synthetic Probe42 payload with the real response shape — no real people."""
from datetime import date

import pytest

import tools.pep_tools as pep
from credit_report.agents import directors as dmod

CIN = "U12345MH2001PTC123456"


@pytest.fixture(autouse=True)
def no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pep, "_CACHE_DIR", tmp_path)


def test_pep_exact_name_hit_is_analyst_review(monkeypatch):
    monkeypatch.setattr(pep, "_myneta_search", lambda n: [
        {"name": "Shri Arun Kumar Mehta", "party": "XYZ", "constituency": "C1", "election": "2019", "criminal": False},
        {"name": "Arun Mehta", "party": "XYZ", "constituency": "C2", "election": "2019", "criminal": True},
    ])
    monkeypatch.setattr(pep, "_wikidata_officeholders", lambda n: [])
    r = pep.screen_person_pep("ARUN KUMAR MEHTA")
    assert r["status"] == "analyst_review"
    assert len(r["matches"]) == 1 and "C1" in r["matches"][0]  # "Arun Mehta" is a different name


def test_pep_no_match(monkeypatch):
    monkeypatch.setattr(pep, "_myneta_search", lambda n: [])
    monkeypatch.setattr(pep, "_wikidata_officeholders", lambda n: [])
    assert pep.screen_person_pep("Arun Kumar Mehta")["status"] == "no_match"


def test_pep_myneta_down_is_unscreened(monkeypatch):
    def boom(n):
        raise ConnectionError("down")
    monkeypatch.setattr(pep, "_myneta_search", boom)
    monkeypatch.setattr(pep, "_wikidata_officeholders", lambda n: [])
    assert pep.screen_person_pep("Arun Kumar Mehta")["status"] == "unscreened"


def test_pep_wikidata_down_is_a_disclosed_gap(monkeypatch):
    def boom(n):
        raise TimeoutError("429")
    monkeypatch.setattr(pep, "_myneta_search", lambda n: [])
    monkeypatch.setattr(pep, "_wikidata_officeholders", boom)
    r = pep.screen_person_pep("Arun Kumar Mehta")
    assert r["status"] == "no_match" and "Coverage gap" in r["note"]


PAGE = {
    "authorized_signatories": [
        {"name": "ARUN KUMAR MEHTA", "din": "01234567", "pan": "ABCPM1234Q", "designation": "Managing Director",
         "din_status": "Approved", "date_of_appointment": "2010-04-01", "date_of_cessation": None},
        {"name": "NEHA RAO", "din": "07654321", "pan": "BCDPR2345R", "designation": "Director",
         "din_status": "Disqualified", "date_of_appointment": "2020-01-15", "date_of_cessation": None},
        {"name": "VIKRAM SINGH", "din": "05555555", "designation": "Director",
         "date_of_appointment": "2015-01-01", "date_of_cessation": "2025-06-30"},
        {"name": "OLD DIRECTOR", "din": "04444444", "designation": "Director",
         "date_of_appointment": "2005-01-01", "date_of_cessation": "2019-03-31"},
    ],
    "director_network": [
        {"din": "01234567", "name": "ARUN KUMAR MEHTA", "network": {
            "companies": [
                {"cin": CIN, "legal_name": "SELF CO", "company_status": "ACTIVE", "designation": "Managing Director"},
                {"cin": "U1", "legal_name": "ALPHA PVT LTD", "company_status": "ACTIVE", "designation": "Director",
                 "date_of_cessation": None, "cirp_status": None},
                {"cin": "U2", "legal_name": "BETA PVT LTD", "company_status": "STRIKE OFF", "designation": "Director",
                 "date_of_cessation": "2018-01-01", "cirp_status": None},
                {"cin": "U3", "legal_name": "GAMMA PVT LTD", "company_status": "ACTIVE", "designation": "Director",
                 "date_of_cessation": None, "cirp_status": "Under CIRP"},
            ],
            "llps": [{"llpin": "AAA-0001", "legal_name": "DELTA LLP", "status": "ACTIVE",
                      "designation": "Designated Partner", "date_of_cessation": None}],
        }},
    ],
}


def test_build_directors_result(monkeypatch):
    monkeypatch.setattr(dmod, "screen_person_pep", lambda n: {"name": n, "status": "no_match", "matches": []})
    r = dmod.build_directors_result(CIN, {"legal_name": "SELF CO PRIVATE LIMITED", "pan": "AAACS1234A"}, PAGE,
                                    today=date(2026, 10, 2))
    assert [d.name for d in r.current] == ["ARUN KUMAR MEHTA", "NEHA RAO"]
    arun = r.current[0]
    assert arun.other_directorships == 3  # ALPHA, GAMMA, DELTA LLP; not self, not the struck-off one
    assert any("BETA" in f and "STRIKE OFF" in f for f in arun.flagged_associations)
    assert any("GAMMA" in f and "CIRP" in f for f in arun.flagged_associations)
    assert arun.din_masked == "01XXXX67" and arun.tenure_years == 16.5
    assert r.ceased_total == 2
    assert [c["name"] for c in r.ceased_last_24_months] == ["VIKRAM SINGH"]


def test_board_context_has_no_identifiers_or_pep(monkeypatch):
    monkeypatch.setattr(dmod, "screen_person_pep",
                        lambda n: {"name": n, "status": "analyst_review", "matches": ["MyNeta/ECI: 'X'"]})
    r = dmod.build_directors_result(CIN, {"legal_name": "SELF CO"}, PAGE, today=date(2026, 10, 2))
    text = dmod.board_context(r.model_dump())
    for secret in ("01234567", "ABCPM1234Q", "07654321", "BCDPR2345R", "01XXXX67", "MyNeta", "PEP"):
        assert secret not in text
    assert "ARUN KUMAR MEHTA" in text and "DIN status: Disqualified" in text
    assert "VIKRAM SINGH" in text


def test_directors_node_without_cin_is_not_available():
    out = dmod.run_directors_check({})
    assert out["directors_result"]["status"] == "not_available" and out["entity_ids"] == []


def test_directors_node_passes_identifiers_and_legal_name(monkeypatch):
    monkeypatch.setattr(dmod, "probe42_company", lambda cin: {"legal_name": "SELF CO PRIVATE LIMITED", "pan": "AAACS1234A"})
    monkeypatch.setattr(dmod, "probe42_directors", lambda cin: PAGE)
    monkeypatch.setattr(dmod, "screen_person_pep", lambda n: {"name": n, "status": "unscreened", "matches": []})
    out = dmod.run_directors_check({"cin": CIN.lower()})
    assert out["entity_ids"] == [CIN, "AAACS1234A"]
    assert out["legal_name"] == "SELF CO PRIVATE LIMITED"
    assert out["directors_result"]["pep_unscreened"] == 2
    assert any("PEP screen could not run" in i["message"] for i in out["issues"])
