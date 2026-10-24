import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from datetime import datetime, timedelta, timezone
import pandas as pd
from quant_scanner.config import AppConfig
from quant_scanner.context import evaluate_context
from quant_scanner.data import drop_partial_bar, freshness_report
from quant_scanner.events import all_events
from quant_scanner.news import NewsItem

def _df(last):
    idx = pd.bdate_range(end=last, periods=5)
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)

def test_partial_bar():
    d = _df("2026-10-20")                                                    # martes
    assert len(drop_partial_bar(d, "SPY", pd.Timestamp("2026-10-20 15:00", tz="UTC"))) == 4   # 11:00 ET: rueda en curso
    assert len(drop_partial_bar(d, "SPY", pd.Timestamp("2026-10-20 21:30", tz="UTC"))) == 5   # 17:30 ET: cerrada
    assert len(drop_partial_bar(_df("2026-10-16"), "SPY", pd.Timestamp("2026-10-17 15:00", tz="UTC"))) == 5   # sábado
    assert len(drop_partial_bar(d, "BTC-USD", pd.Timestamp("2026-10-20 21:30", tz="UTC"))) == 4  # cripto: hoy incompleto
    assert len(drop_partial_bar(_df("2026-10-19"), "BTC-USD", pd.Timestamp("2026-10-20 21:30", tz="UTC"))) == 5

def test_freshness():
    px = {"SPY": _df("2026-10-20"), "AAA": _df("2026-10-19"), "BBB": _df("2026-10-20"), "BTC-USD": _df("2026-10-19"), "^VIX": _df("2026-10-20")}
    fr = freshness_report(px, now=pd.Timestamp("2026-10-20 22:00", tz="UTC")).set_index("ticker")
    assert not fr.loc["AAA", "al_dia"] and fr.loc["AAA", "ruedas_de_atraso"] == 1 and fr.loc["BBB", "al_dia"]
    assert fr.loc["BTC-USD", "al_dia"] and "^VIX" not in fr.index

def test_context_overlay():
    cfg, ev = AppConfig(), all_events()
    now = datetime(2026, 10, 27, 22, 0, tzinfo=timezone.utc)
    bad = [NewsItem("XYZ 8-K", "sec.gov", "https://sec.gov/a", now - timedelta(hours=3), ("XYZ",), form="8-K", items=("4.02",))]
    r = evaluate_context(bad, pd.Timestamp("2026-10-27"), ev, cfg, 15, now=now)
    assert r["decision"] == "VETO" and r["risk_mult"] == 0.5 and any("FOMC" in n for n in r["notes"])   # señal el martes, FOMC el miércoles
    r = evaluate_context([], pd.Timestamp("2026-10-20"), ev, cfg, 15, earnings=pd.Timestamp("2026-10-30"), fetch_errors=["yfinance: Timeout"], now=now)
    assert r["decision"] == "OK" and r["risk_mult"] == 0.5 and any("Balance" in n for n in r["notes"]) and any("No pude verificar" in n for n in r["notes"])
    r = evaluate_context([], pd.Timestamp("2026-10-20"), ev, cfg, 10, earnings=pd.Timestamp("2026-12-20"), now=now)
    assert r["risk_mult"] == 1.0 and not r["notes"]

if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
