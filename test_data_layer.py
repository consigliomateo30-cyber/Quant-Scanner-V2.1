"""Capa de datos con un yfinance SIMULADO que replica el formato real (columnas MultiIndex)."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["QS_CACHE_DIR"] = tempfile.mkdtemp()
import numpy as np, pandas as pd
from synth import make_prices

PX = make_prices(4, 600, seed=3)
CALLS = {"n": 0, "fail": False}


def _frame(tk):
    d = PX["SPY"] if tk == "SPY" else PX["A0"]
    return d[["Open", "High", "Low", "Close", "Volume"]].copy()


def fake_download(tickers, start=None, auto_adjust=True, group_by="ticker", progress=False, threads=True):
    CALLS["n"] += 1
    if CALLS["fail"]:
        raise ConnectionError("sin red")
    if isinstance(tickers, str):
        tickers = [tickers]
    parts = {}
    for tk in tickers:
        f = _frame(tk)
        if tk == "BAD":                                    # activo con filas rotas: NaN, ceros y duplicados
            f = f.copy()
            f.iloc[5, :] = np.nan
            f.iloc[7, f.columns.get_loc("Close")] = 0.0
        parts[tk] = f
    if len(parts) == 1:                                    # yfinance reciente: (Price, Ticker) aun con un solo ticker
        tk, f = next(iter(parts.items()))
        f.columns = pd.MultiIndex.from_product([f.columns, [tk]])
        return f
    return pd.concat(parts, axis=1)                        # (Ticker, Price) con group_by='ticker'


sys.modules["yfinance"] = types.SimpleNamespace(download=fake_download)
import quant_scanner.data as qd


def test_clean_handles_real_formats():
    f = _frame("SPY"); f.columns = pd.MultiIndex.from_product([f.columns, ["SPY"]])
    c = qd._clean(f)
    assert list(c.columns) == qd.OHLCV and c.index.is_monotonic_increasing
    assert qd._clean(pd.DataFrame({"x": [1]})) is None and qd._clean(pd.DataFrame()) is None and qd._clean(None) is None
    bad = _frame("SPY"); bad = pd.concat([bad, bad.iloc[-2:]]); bad.iloc[3, :] = np.nan; bad.iloc[4, bad.columns.get_loc("Low")] = -1
    c = qd._clean(bad)
    assert c.index.is_unique and (c[["Open", "High", "Low", "Close"]] > 0).all().all() and len(c) == len(PX["SPY"]) - 2


def test_download_multi_single_cache_and_fallback():
    out = qd.download_prices(["SPY", "A0", "BAD"], years=3, use_cache=False)
    assert set(out) == {"SPY", "A0", "BAD"} and out["BAD"].index.is_unique and (out["BAD"]["Close"] > 0).all()
    n = CALLS["n"]
    again = qd.download_prices(["SPY", "A0"], years=3)                    # sirve desde caché (< 1 hora)
    assert CALLS["n"] == n and set(again) == {"SPY", "A0"}
    one = qd.download_prices(["SPY"], years=3, use_cache=False)           # un solo ticker: formato (Price, Ticker)
    assert set(one) == {"SPY"} and len(one["SPY"]) >= len(PX["SPY"]) - 1
    CALLS["fail"] = True                                                   # se cae la red: usa la caché vieja en vez de nada
    stale = qd.download_prices(["SPY", "A0"], years=3, use_cache=True, max_cache_age_hours=0.0)
    assert set(stale) == {"SPY", "A0"}
    none = qd.download_prices(["ZZZ"], years=3, use_cache=False)
    assert none == {}                                                      # sin red y sin caché: vacío, no excepción
    CALLS["fail"] = False


def test_partial_bar_dropped_from_downloads():
    d = PX["SPY"].copy()
    today_et = pd.Timestamp.now(tz="America/New_York")
    if today_et.weekday() < 5 and (today_et.hour * 60 + today_et.minute) < 16 * 60 + 20:
        d.index = d.index + (today_et.tz_localize(None).normalize() - d.index[-1])      # última barra = hoy, rueda abierta
        assert len(qd.drop_partial_bar(d, "SPY")) == len(d) - 1
    else:
        assert len(qd.drop_partial_bar(d, "SPY")) == len(d)


if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
