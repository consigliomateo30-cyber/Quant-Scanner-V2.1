"""Biblioteca de indicadores. Todos son CAUSALES: el valor en la barra t usa solo datos hasta t
(un test lo verifica truncando la serie). Cada uno declara su familia; no se agregan indicadores
"porque sí": solo los que tienen sustento en la literatura o en la práctica institucional."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def sma(d, n=20):
    return d["Close"].rolling(n).mean()


def ema(d, n=20):
    return d["Close"].ewm(span=n, adjust=False, min_periods=n).mean()


def atr(d, n=14):
    pc = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(d, n=14):
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna(), np.nan)


def ibs(d):
    """Internal Bar Strength: dónde cerró dentro del rango del día (0 = mínimo, 1 = máximo)."""
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    return ((d["Close"] - d["Low"]) / rng).fillna(0.5)


def donchian_high(d, n=20):
    return d["High"].shift(1).rolling(n).max()


def donchian_low(d, n=10):
    return d["Low"].shift(1).rolling(n).min()


def bollinger_z(d, n=20):
    m, s = d["Close"].rolling(n).mean(), d["Close"].rolling(n).std()
    return (d["Close"] - m) / s.replace(0, np.nan)


def roc(d, n=10):
    return d["Close"].pct_change(n)


def mom_skip(d, n=252, skip=21):
    """Momentum 12-1 (n barras menos las últimas `skip`): evita el reversal de corto plazo."""
    return d["Close"].shift(skip) / d["Close"].shift(n) - 1


def dist_high(d, n=252):
    """Distancia al máximo de n barras (George-Hwang): cerca del máximo anual = continuación."""
    return d["Close"] / d["High"].rolling(n).max() - 1


def macd_hist(d, fast=12, slow=26, sig=9):
    line = d["Close"].ewm(span=fast, adjust=False, min_periods=slow).mean() - d["Close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    return line - line.ewm(span=sig, adjust=False, min_periods=sig).mean()


def adx(d, n=14):
    up, dn = d["High"].diff(), -d["Low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(d, n)
    pdi = 100 * pd.Series(plus, index=d.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * pd.Series(minus, index=d.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stoch_k(d, n=14):
    lo, hi = d["Low"].rolling(n).min(), d["High"].rolling(n).max()
    return 100 * (d["Close"] - lo) / (hi - lo).replace(0, np.nan)


def cci(d, n=20):
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def squeeze_on(d, n=20, bb_k=2.0, kc_k=1.5):
    """Bandas de Bollinger dentro de los canales de Keltner = volatilidad comprimida."""
    m, s = d["Close"].rolling(n).mean(), d["Close"].rolling(n).std()
    a = atr(d, n)
    return ((m + bb_k * s) < (m + kc_k * a)) & ((m - bb_k * s) > (m - kc_k * a))


def realized_vol(d, n=20):
    return d["Close"].pct_change().rolling(n).std() * np.sqrt(252)


def vol_rank(d, n=20, window=252):
    """Percentil de la volatilidad actual dentro del último año (régimen de volatilidad)."""
    return realized_vol(d, n).rolling(window).rank(pct=True)


def volume_z(d, n=20):
    v = d["Volume"]
    return (v - v.rolling(n).mean()) / v.rolling(n).std().replace(0, np.nan)


def obv_slope(d, n=20):
    obv = (np.sign(d["Close"].diff().fillna(0)) * d["Volume"]).cumsum()
    return obv.diff(n) / d["Volume"].rolling(n).sum().replace(0, np.nan)


def gap_pct(d):
    return d["Open"] / d["Close"].shift(1) - 1


def consec_down(d):
    """Cantidad de cierres consecutivos a la baja (Connors)."""
    down = (d["Close"].diff() < 0).astype(int)
    grp = (down == 0).cumsum()
    return down.groupby(grp).cumsum()


def sma_slope(d, n=50, lag=10):
    m = d["Close"].rolling(n).mean()
    return m / m.shift(lag) - 1


def rel_strength(d, bench: pd.Series, n=63):
    """Retorno propio menos retorno del benchmark en n barras (fuerza relativa)."""
    b = bench.reindex(d.index, method="ffill")
    return d["Close"].pct_change(n) - b.pct_change(n)


# nombre -> (función, familia)
REGISTRY: dict[str, tuple[Callable, str]] = {
    "sma": (sma, "tendencia"), "ema": (ema, "tendencia"), "macd_hist": (macd_hist, "tendencia"), "adx": (adx, "tendencia"),
    "sma_slope": (sma_slope, "tendencia"), "donchian_high": (donchian_high, "ruptura"), "donchian_low": (donchian_low, "ruptura"),
    "roc": (roc, "momentum"), "mom_skip": (mom_skip, "momentum"), "dist_high": (dist_high, "momentum"),
    "rsi": (rsi, "reversión"), "ibs": (ibs, "reversión"), "bollinger_z": (bollinger_z, "reversión"), "stoch_k": (stoch_k, "reversión"),
    "cci": (cci, "reversión"), "consec_down": (consec_down, "reversión"),
    "atr": (atr, "volatilidad"), "squeeze_on": (squeeze_on, "volatilidad"), "realized_vol": (realized_vol, "volatilidad"),
    "vol_rank": (vol_rank, "volatilidad"), "volume_z": (volume_z, "volumen"), "obv_slope": (obv_slope, "volumen"),
    "gap_pct": (gap_pct, "microestructura"),
}


def resample_ohlcv(d: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    """Pasa a barras semanales (u otra regla) descartando la última barra si está incompleta."""
    r = d.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    if len(r) and rule.startswith("W-FRI") and r.index[-1] > d.index[-1] and d.index[-1].weekday() < 4:
        r = r.iloc[:-1]
    return r
