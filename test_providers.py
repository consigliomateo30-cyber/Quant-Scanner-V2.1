import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import quant_scanner.providers as P
from quant_scanner.fred import parse_observations

TIINGO = [{"date": f"2026-10-{d:02d}T00:00:00.000Z", "adjOpen": 100.0 + d, "adjHigh": 102.0 + d, "adjLow": 99.0 + d, "adjClose": 101.0 + d, "adjVolume": 1e6} for d in (14, 15, 16, 19, 20)]
ALPACA_MULTI = {"bars": {"SPY": [{"t": f"2026-10-{d:02d}T04:00:00Z", "o": 100.0 + d, "h": 102.0 + d, "l": 99.0 + d, "c": 101.0 + d, "v": 5e4} for d in (14, 15, 16, 19, 20)]}, "next_page_token": None}

def test_parsers():
    t = P.parse_tiingo(TIINGO)
    assert list(t.columns) == P.OHLCV and len(t) == 5 and t.index[-1] == pd.Timestamp("2026-10-20") and t["Close"].iloc[-1] == 121.0
    assert P.parse_tiingo([]) is None and P.parse_tiingo([{"x": 1}]) is None
    a = P.parse_alpaca(ALPACA_MULTI, "SPY")
    assert len(a) == 5 and a["Volume"].iloc[0] == 5e4
    assert len(P.parse_alpaca({"bars": ALPACA_MULTI["bars"]["SPY"]}, "SPY")) == 5 and P.parse_alpaca({"bars": {}}, "SPY") is None
    q = P.parse_finnhub_quote({"c": 101.5, "dp": 0.8, "h": 102, "l": 100, "o": 100.5, "pc": 100.7, "t": 1792500000})
    assert q["precio"] == 101.5 and q["hora_utc"].tzinfo is not None and P.parse_finnhub_quote({"c": 0}) is None
    s = parse_observations({"observations": [{"date": "2026-10-01", "value": "4.12"}, {"date": "2026-10-02", "value": "."}, {"date": "2026-10-05", "value": "4.20"}]})
    assert list(s.values) == [4.12, 4.20]

def test_reconcile():
    a, b = P.parse_tiingo(TIINGO), P.parse_alpaca(ALPACA_MULTI, "SPY")
    assert P.reconcile({"tiingo": a, "alpaca": b})["estado"] == "ok"
    bad = b.copy(); bad.iloc[-1, bad.columns.get_loc("Close")] *= 1.03                  # un tick malo de +3%
    r = P.reconcile({"tiingo": a, "alpaca": bad})
    assert r["estado"] == "conflicto" and r["cuarentena"] and "2.9" in r["detalle"][0]
    stale = b.iloc[:-1]
    assert P.reconcile({"tiingo": a, "alpaca": stale})["estado"] == "advertencia"
    assert P.reconcile({"tiingo": a})["estado"] == "una sola fuente"

def test_http_backoff_and_no_secret_leak():
    calls = []
    class R:
        def __init__(s, code, js=None): s.status_code, s._js, s.ok = code, js, 200 <= code < 300
        def json(s): return s._js
    seq = [R(429), R(503), R(200, {"ok": 1})]
    orig = P.requests.get
    P.requests.get = lambda *a, **k: (calls.append(k), seq.pop(0))[1]
    try:
        waits = []
        assert P.http_get("https://x.test/a", {"api_key": "SECRETO123"}, retries=3, sleep=waits.append) == {"ok": 1} and waits == [1, 2]
        P.requests.get = lambda *a, **k: R(401)
        try: P.http_get("https://x.test/a", {"api_key": "SECRETO123"}, {"X-Token": "SECRETO123"}); raise AssertionError()
        except P.ProviderError as e: assert "SECRETO123" not in str(e) and "credenciales" in str(e)
        P.requests.get = lambda *a, **k: R(500)
        try: P.http_get("https://x.test/a?k=SECRETO123", retries=2, sleep=lambda s: None); raise AssertionError()
        except P.ProviderError as e: assert "SECRETO123" not in str(e) and "500" in str(e)
    finally:
        P.requests.get = orig

def test_fetch_validated_tolerates_failures():
    P.FETCHERS = {"tiingo": lambda t, s: (_ for _ in ()).throw(P.ProviderError("falta TIINGO_TOKEN")),
                  "yfinance": lambda t, s: P.parse_tiingo(TIINGO), "alpaca": lambda t, s: P.parse_alpaca(ALPACA_MULTI, t)}
    df, rep = P.fetch_validated("SPY", "2026-01-01")
    assert df is not None and rep["estado"] == "ok" and rep["errores"] == {"tiingo": "falta TIINGO_TOKEN"}
    df, rep = P.fetch_validated("BTC-USD", "2026-01-01")                                # cripto: solo yfinance
    assert rep["fuentes"] == ["yfinance"]
    P.FETCHERS = {k: (lambda t, s: (_ for _ in ()).throw(P.ProviderError("x"))) for k in ("tiingo", "yfinance", "alpaca")}
    df, rep = P.fetch_validated("SPY", "2026-01-01")
    assert df is None and rep["cuarentena"] and rep["estado"] == "sin datos"

if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
