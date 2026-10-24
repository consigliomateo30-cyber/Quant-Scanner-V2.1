"""Familias de estrategia parametrizables. Cada una declara entrada y salida con indicadores de la biblioteca.
Se limita la cantidad de variantes a propósito: cada una es una 'prueba' que se paga en la corrección por
comparaciones múltiples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass(frozen=True)
class Spec:
    family: str
    params: tuple
    stop_atr: float
    max_hold: int
    timeframe: str = "1d"

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def name(self) -> str:
        body = ",".join(f"{k}={v:g}" for k, v in self.params)
        return f"{self.family}[{body}] stop={self.stop_atr:g}ATR hold={self.max_hold}|{self.timeframe}"


def make_spec(family: str, timeframe: str, stop_atr: float, max_hold: int, **params) -> Spec:
    return Spec(family, tuple(sorted(params.items())), stop_atr, max_hold, timeframe)


def _arr(x) -> np.ndarray:
    return pd.Series(x).fillna(False).to_numpy(dtype=bool)


def signals_for(d: pd.DataFrame, spec: Spec, regime_ok: pd.Series | None = None, bench: pd.Series | None = None):
    """Devuelve (entrada, salida) como arrays booleanos alineados con d. Todo causal."""
    p, c = spec.p, d["Close"]
    reg = (regime_ok.reindex(d.index, method="ffill").fillna(False).astype(bool) if regime_ok is not None
           else pd.Series(True, index=d.index))
    trend_ok = c > ind.sma(d, int(p["trend_n"])) if "trend_n" in p else pd.Series(True, index=d.index)
    f = spec.family
    if f == "donchian":
        brk = c > ind.donchian_high(d, int(p["entry_n"]))
        entry = brk & ~brk.shift(1, fill_value=False) & reg
        if "rs_n" in p:
            entry = entry & (ind.rel_strength(d, bench, int(p["rs_n"])) > 0) if bench is not None else entry & False
        exit_ = c < ind.donchian_low(d, int(p["exit_n"]))
    elif f == "rsi_pullback":
        entry = (ind.rsi(d, int(p.get("rsi_n", 2))) < p["thr"]) & trend_ok & reg
        exit_ = c > ind.sma(d, int(p["exit_n"]))
    elif f == "ibs_reversion":
        entry = (ind.ibs(d) < p["thr"]) & trend_ok & reg
        exit_ = (ind.ibs(d) > 0.7) | (c > d["High"].shift(1))
    elif f == "bb_reversion":
        entry = (ind.bollinger_z(d, int(p["n"])) < -p["k"]) & trend_ok & reg
        exit_ = c > ind.sma(d, int(p["n"]))
    elif f == "mom_pullback":
        entry = (ind.mom_skip(d, int(p["mom_n"]), int(p["skip_n"])) > 0) & (ind.rsi(d, 2) < p["thr"]) & trend_ok & reg
        exit_ = c > ind.sma(d, int(p["exit_n"]))
    elif f == "high52":
        near = ind.dist_high(d, int(p["n"])) > -p["near"]
        entry = near & ~near.shift(1, fill_value=False) & (c > ind.sma(d, int(p["trend_n"]))) & reg
        exit_ = c < ind.sma(d, int(p["exit_n"]))
    elif f == "ema_adx":
        fast, slow = ind.ema(d, int(p["fast"])), ind.ema(d, int(p["slow"]))
        cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        entry = cross & (ind.adx(d, 14) > p["adx_min"]) & trend_ok & reg
        exit_ = fast < slow
    elif f == "squeeze":
        sq = ind.squeeze_on(d, int(p["n"]))
        entry = sq.shift(1, fill_value=False) & ~sq & (c > ind.sma(d, int(p["n"]))) & (ind.macd_hist(d) > 0) & reg
        exit_ = c < ind.sma(d, int(p["exit_n"]))
    else:
        raise ValueError(f"familia desconocida: {f}")
    return _arr(entry), _arr(exit_)


def default_specs(timeframe: str = "1d") -> list[Spec]:
    if timeframe == "1d":
        T = dict(trend_n=200)
        return [
            make_spec("donchian", "1d", 2.0, 15, entry_n=10, exit_n=5),
            make_spec("donchian", "1d", 2.5, 30, entry_n=20, exit_n=10),
            make_spec("donchian", "1d", 3.0, 60, entry_n=55, exit_n=20),
            make_spec("donchian", "1d", 2.5, 30, entry_n=20, exit_n=10, rs_n=63),
            make_spec("rsi_pullback", "1d", 3.0, 10, thr=5, exit_n=5, **T),
            make_spec("rsi_pullback", "1d", 3.0, 10, thr=10, exit_n=5, **T),
            make_spec("rsi_pullback", "1d", 3.0, 10, thr=15, exit_n=5, **T),
            make_spec("ibs_reversion", "1d", 3.0, 7, thr=0.2, **T),
            make_spec("bb_reversion", "1d", 3.0, 10, n=20, k=2.0, **T),
            make_spec("bb_reversion", "1d", 3.0, 10, n=20, k=2.5, **T),
            make_spec("mom_pullback", "1d", 3.0, 10, mom_n=252, skip_n=21, thr=10, exit_n=5, **T),
            make_spec("mom_pullback", "1d", 3.0, 10, mom_n=252, skip_n=21, thr=20, exit_n=5, **T),
            make_spec("high52", "1d", 2.5, 40, n=252, near=0.03, trend_n=50, exit_n=20),
            make_spec("ema_adx", "1d", 3.0, 60, fast=20, slow=50, adx_min=20, **T),
            make_spec("squeeze", "1d", 2.5, 20, n=20, exit_n=10),
        ]
    if timeframe == "1w":
        T = dict(trend_n=40)
        return [
            make_spec("donchian", "1w", 2.5, 8, entry_n=8, exit_n=4),
            make_spec("donchian", "1w", 3.0, 13, entry_n=13, exit_n=5),
            make_spec("rsi_pullback", "1w", 3.0, 4, thr=10, exit_n=2, **T),
            make_spec("rsi_pullback", "1w", 3.0, 4, thr=15, exit_n=2, **T),
            make_spec("ibs_reversion", "1w", 3.0, 3, thr=0.2, **T),
            make_spec("mom_pullback", "1w", 3.0, 4, mom_n=52, skip_n=4, thr=10, exit_n=2, **T),
            make_spec("high52", "1w", 3.0, 13, n=52, near=0.03, trend_n=10, exit_n=10),
            make_spec("ema_adx", "1w", 3.0, 26, fast=10, slow=30, adx_min=20, **T),
        ]
    raise ValueError(timeframe)
