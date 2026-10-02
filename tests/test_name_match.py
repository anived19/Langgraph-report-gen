import pytest

from tools.name_match import canonical_name, identifier_verdict, mask_din, mask_pan, same_name


@pytest.mark.parametrize("a,b,want", [
    ("SKANDAN PLASTRIX PRIVATE LIMITED", "Skandan Plastrix Pvt. Ltd.", True),
    ("B R TRADING CO", "B.R. Trading Company", True),
    ("M/s B R Trading Co", "BR TRADING COMPANY", True),
    ("Tata Steel Limited", "Tata Steel Ltd", True),
    # near names never match
    ("B R TRADING CO", "MR TRADING CO", False),
    ("B R TRADING CO", "BAR TRADING CO", False),
    ("Tata Steel Limited", "Tata Steel Europe Limited", False),
    ("Reliance Industries Limited", "Reliance Industries", False),
    ("", "", False),
])
def test_same_name_is_exact_only(a, b, want):
    assert same_name(a, b) is want


def test_canonical_name_matches_platform_port():
    # Same expectations as fino-backend free-screens.spec.ts
    assert canonical_name("MR TRADING CO") == "mr trading company"
    assert canonical_name("Ms B R TRADING COMPANY") == "br trading company"
    assert canonical_name("Mrs Priya Sharma") == "priya sharma"
    assert canonical_name("Shri Dr. Ramesh Chand Sharma") == "ramesh chand sharma"


@pytest.mark.parametrize("text,ids,want", [
    ("Entity X, CIN U74899DL1986PTC024176", ["U74899DL1986PTC024176"], "confirmed"),
    ("GSTIN 07AAACJ0096L1ZQ", ["AAACJ0096L"], "confirmed"),  # PAN embedded in a GSTIN
    ("PAN ABCDE1234F", ["AAACJ0096L"], "different"),
    ("listed 2019, no identifiers", ["AAACJ0096L"], "same_name_only"),
    ("listed 2019, no identifiers", [], "same_name_only"),
])
def test_identifier_verdict(text, ids, want):
    assert identifier_verdict(text, ids) == want


def test_masks():
    assert mask_pan("ABCDE1234F") == "ABXXXXXXXF"
    assert mask_din("07549147") == "07XXXX47"
