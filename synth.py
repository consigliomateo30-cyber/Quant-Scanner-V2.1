"""Datos sintéticos con regímenes de tendencia, para probar la lógica sin internet.
NO sirven para medir rentabilidad real."""
import numpy as np
import pandas as pd


def make_prices(n_assets=12, n_days=2500, seed=1, trend_edge=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    out = {}
    names = ["SPY"] + [f"A{i}" for i in range(n_assets - 1)]
    mkt_regime = np.zeros(n_days)
    state = 1
    for t in range(n_days):
        if rng.random() < 1 / 150:
            state *= -1
        mkt_regime[t] = state
    for k, nm in enumerate(names):
        drift = np.zeros(n_days)
        s = 1
        for t in range(n_days):
            if rng.random() < 1 / 60:
                s = rng.choice([-1, 0, 1])
            drift[t] = trend_edge * s + 0.0002 * mkt_regime[t]
        ret = drift + rng.normal(0, 0.012 if nm != "SPY" else 0.008, n_days)
        close = 100 * np.exp(np.cumsum(ret))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        vol = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        out[nm] = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
    return out
