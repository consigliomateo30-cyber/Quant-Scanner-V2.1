import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from quant_scanner import indicators as ind
from synth import make_prices

def test_all_indicators_causal():
    d = make_prices(2, 700, seed=2)["A0"]
    for name, (fn, fam) in ind.REGISTRY.items():
        full = fn(d)
        for k in (350, 520, 690):
            part = fn(d.iloc[:k]).iloc[-1]
            a = full.iloc[k - 1]
            assert (pd.isna(a) and pd.isna(part)) or abs(float(a) - float(part)) < 1e-9, (name, k, a, part)

def test_known_values():
    d = pd.DataFrame({"Open": [1, 1, 1], "High": [2, 3, 2], "Low": [0, 1, 1], "Close": [1, 3, 1.5], "Volume": [1, 1, 1]},
                     index=pd.bdate_range("2024-01-01", periods=3))
    assert list(ind.ibs(d).round(3)) == [0.5, 1.0, 0.5]
    assert list(ind.consec_down(d)) == [0, 0, 1]
    c = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [10, 11, 12, 11, 10, 9.0], "Volume": 1.0}, index=pd.bdate_range("2024-01-01", periods=6))
    assert list(ind.consec_down(c)) == [0, 0, 0, 1, 2, 3]
    assert abs(ind.roc(c, 1).iloc[1] - 0.1) < 1e-12 and ind.rsi(c, 2).iloc[2] == 100.0

def test_weekly_resample_drops_partial():
    idx = pd.bdate_range("2026-10-05", "2026-10-21")                       # termina un miércoles
    d = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10.0}, index=idx)
    w = ind.resample_ohlcv(d)
    assert w.index[-1] == pd.Timestamp("2026-10-16") and w["Volume"].iloc[0] == 50.0
    full = ind.resample_ohlcv(d.loc[:"2026-10-16"])
    assert full.index[-1] == pd.Timestamp("2026-10-16")

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("OK ", n)
