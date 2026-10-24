import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from quant_scanner.edgar import parse_submissions, parse_form4_xml
from quant_scanner.altdata import insider_cluster_state, flag_trades, validate_signal

SUB = {"filings": {"recent": {"form": ["8-K", "4", "10-Q"], "filingDate": ["2026-10-19", "2026-10-18", "2026-10-01"],
        "acceptanceDateTime": ["2026-10-19T21:05:11.000Z", "2026-10-18T14:00:00.000Z", "2026-10-01T20:00:00.000Z"],
        "items": ["2.02,9.01", "", ""], "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-26-000003"],
        "primaryDocument": ["a.htm", "xslF345X05/form4.xml", "q.htm"]}}}
F4 = """<ownershipDocument><issuer><issuerTradingSymbol>XYZ</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerName>DOE JOHN</rptOwnerName></reportingOwnerId>
<reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle></reportingOwnerRelationship></reportingOwner>
<nonDerivativeTable><nonDerivativeTransaction><transactionDate><value>2026-10-16</value></transactionDate>
<transactionCoding><transactionCode>P</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>25.5</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction>
<nonDerivativeTransaction><transactionDate><value>2026-10-16</value></transactionDate><transactionCoding><transactionCode>S</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>25</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""

def test_parsers():
    df = parse_submissions(SUB)
    assert list(df["form"]) == ["8-K", "4", "10-Q"] and df.loc[0, "items"] == ("2.02", "9.01") and df["accepted"].dt.tz is not None
    assert parse_submissions({}).empty
    rows = parse_form4_xml(F4)
    assert len(rows) == 2 and rows[0]["code"] == "P" and rows[0]["value_usd"] == 25500.0 and rows[0]["is_officer"] and rows[0]["title"] == "CEO"
    assert parse_form4_xml("no xml") == []

def _purchases():
    return pd.DataFrame({"ticker": ["AAA"] * 4 + ["BBB"], "filed": pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-20", "2026-03-01", "2026-01-05"], utc=True),
                         "insider": ["a", "b", "c", "a", "z"], "value_usd": [30000, 25000, 40000, 20000, 90000]})

def test_cluster_and_pit_flag():
    st = insider_cluster_state(_purchases())
    a = st[st["ticker"] == "AAA"].set_index("asof")
    assert a.loc[pd.Timestamp("2026-01-20"), "cluster"] and not a.loc[pd.Timestamp("2026-01-12"), "cluster"]
    assert not st[st["ticker"] == "BBB"]["cluster"].any()                     # un solo insider no es cluster
    tr = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "BBB"],
                       "signal_date": pd.to_datetime(["2026-01-20", "2026-01-21", "2026-03-15", "2026-01-21"])})
    f = flag_trades(tr, st)
    assert list(f) == [False, True, False, False]                             # el mismo día no cuenta; 54 días después tampoco

def test_validate_signal():
    rng = np.random.default_rng(1)
    n = 600
    flag = pd.Series(rng.random(n) < 0.15)
    base = rng.normal(-0.05, 1.0, n)
    tr = pd.DataFrame({"signal_date": pd.date_range("2016-01-01", periods=n, freq="5D"), "R_net": base + np.where(flag, 0.8, 0.0)})
    ok = validate_signal(tr, flag)
    assert ok["decision"] == "graduar al modelo", ok
    noise = pd.DataFrame({"signal_date": tr["signal_date"], "R_net": base})
    bad = validate_signal(noise, flag)
    assert bad["decision"] == "solo informativo" and bad["reasons"]
    few = validate_signal(tr, pd.Series(np.arange(n) < 5))
    assert few["decision"] == "solo informativo"

if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
