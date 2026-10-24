"""Indicadores, señales y simulación de trades (sin look-ahead).

Regla de ejecución: la señal se confirma al CIERRE de la barra i y se ejecuta en la
APERTURA de la barra i+1. Es exactamente como se opera una alerta de fin de día."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, StrategyConfig

_ENTRY_COL = {"trend": "entry_signal", "pullback": "entry_pullback"}
_EXIT_COL = {"trend": "exit_trend", "pullback": "exit_pullback"}


def regime_series(spy: pd.DataFrame | None, sma: int = 200) -> pd.Series:
    """True cuando SPY cierra por encima de su SMA. Sin datos: serie vacía (el consumidor falla seguro)."""
    if spy is None or len(spy) < sma:
        return pd.Series(dtype=bool)
    return (spy["Close"] > spy["Close"].rolling(sma).mean()).fillna(False)


def vix_rank_series(vix: pd.DataFrame | None) -> pd.Series | None:
    """Percentil del VIX en el último año (0-1). Usa solo datos pasados."""
    if vix is None or len(vix) < 260:
        return None
    return vix["Close"].rolling(252).rank(pct=True)


def _wilder_rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna(), np.nan)


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig, regime: pd.Series | None = None,
                   spy_ret20: pd.Series | None = None, vix_rank: pd.Series | None = None) -> pd.DataFrame:
    d = df.copy()
    prev_close = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / cfg.atr_window, adjust=False, min_periods=cfg.atr_window).mean()
    d["don_high"] = d["High"].shift(1).rolling(cfg.entry_window).max()
    d["don_low"] = d["Low"].shift(1).rolling(cfg.exit_window).min()

    ret = d["Close"].pct_change()
    d["rsi14"] = _wilder_rsi(d["Close"], 14)
    d["rsi2"] = _wilder_rsi(d["Close"], 2)
    d["vol_20d"] = ret.rolling(20).std()
    d["atr_norm"] = d["atr"] / d["Close"]
    d["dist_donchian"] = (d["Close"] - d["don_high"]) / d["atr"]
    vol5, vol20 = d["Volume"].rolling(5).mean(), d["Volume"].rolling(20).mean()
    d["vol_ratio"] = (vol5 / vol20.replace(0, np.nan)).fillna(1.0)
    d["ret_5d"] = d["Close"].pct_change(5)
    d["ret_20d"] = d["Close"].pct_change(20)
    sma200, sma50 = d["Close"].rolling(200).mean(), d["Close"].rolling(50).mean()
    d["dist_sma200"] = d["Close"] / sma200 - 1
    d["dist_sma50"] = d["Close"] / sma50 - 1
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    d["ibs"] = ((d["Close"] - d["Low"]) / rng).fillna(0.5)          # posición del cierre en el rango del día
    d["rs_20d"] = d["ret_20d"] - spy_ret20.reindex(d.index, method="ffill") if spy_ret20 is not None else np.nan
    d["vix_rank"] = vix_rank.reindex(d.index, method="ffill") if vix_rank is not None else np.nan

    if regime is not None and len(regime):
        d["mkt_regime"] = regime.reindex(d.index, method="ffill").fillna(False).astype(float)
    else:   # sin datos de SPY: si el filtro está activo se bloquea todo (falla seguro)
        d["mkt_regime"] = 0.0 if cfg.use_regime else 1.0
    ok_regime = (d["mkt_regime"] > 0) if cfg.use_regime else True

    # --- sistema de tendencia
    d["breakout"] = (d["Close"] > d["don_high"]) & d["don_high"].notna()
    d["entry_signal"] = d["breakout"] & ~d["breakout"].shift(1, fill_value=False) & ok_regime
    d["exit_trend"] = (d["Close"] < d["don_low"]) & d["don_low"].notna()
    # --- sistema de pullback
    trend_sma = d["Close"].rolling(cfg.pb_trend_sma).mean()
    d["entry_pullback"] = (d["rsi2"] < cfg.pb_rsi_entry) & (d["Close"] > trend_sma) & trend_sma.notna() & ok_regime
    exit_sma = d["Close"].rolling(cfg.pb_exit_sma).mean()
    d["exit_pullback"] = (d["Close"] > exit_sma) & exit_sma.notna()
    d["is_pullback"] = 0.0
    return d


def run_trade_loop(o, h, l, c, atr, entry_sig, exit_sig, stop_mult: float, max_hold: int, cooldown: int = 0,
                   partial_r: float | None = None, partial_frac: float = 0.5):
    """Núcleo ÚNICO de simulación (lo usan el backtest de producción y la matriz de investigación).
    Señal al cierre de i -> entrada en la apertura de i+1. Por barra: stop (gap-aware) -> parcial -> tiempo -> salida
    por señal (a la apertura siguiente). Devuelve tuplas (i, e, k, entry, stop, precio_salida_mezclado, motivo)."""
    n = len(o)
    out, i = [], 0
    while i < n - 1:
        if not entry_sig[i] or np.isnan(atr[i]):
            i += 1
            continue
        e = i + 1
        entry = o[e]
        stop = entry - stop_mult * atr[i]
        if stop <= 0 or entry <= stop:
            i += 1
            continue
        risk = entry - stop
        target = entry + partial_r * risk if partial_r is not None else np.inf
        partial_px, exit_px, exit_k, reason = None, None, None, None
        for j in range(e, n):
            if l[j] <= stop:
                exit_px, exit_k, reason = (o[j] if o[j] < stop else stop), j, "stop"
                break
            if partial_r is not None and partial_px is None and h[j] >= target:
                partial_px = max(target, o[j]) if o[j] > target else target
            if j - e + 1 >= max_hold:
                exit_px, exit_k, reason = c[j], j, "tiempo"
                break
            if exit_sig[j] and j + 1 < n:
                exit_px, exit_k, reason = o[j + 1], j + 1, "salida"
                break
        if exit_px is None:
            break
        blended = exit_px
        if partial_px is not None:
            blended = partial_frac * partial_px + (1 - partial_frac) * exit_px
            reason += "+parcial"
        out.append((i, e, exit_k, entry, stop, blended, reason))
        i = exit_k + (cooldown if reason.startswith("stop") else 0)
    return out


def simulate_trades(d: pd.DataFrame, cfg: StrategyConfig, ticker: str = "", system: str = "trend") -> pd.DataFrame:
    """Trades de un sistema con la regla REAL de salida. Devuelve el R bruto y los features de la barra de señal."""
    idx = d.index
    use_partial = cfg.partial_r is not None and system == "trend"
    raw = run_trade_loop(d["Open"].to_numpy(), d["High"].to_numpy(), d["Low"].to_numpy(), d["Close"].to_numpy(),
                         d["atr"].to_numpy(), d[_ENTRY_COL[system]].to_numpy(), d[_EXIT_COL[system]].to_numpy(),
                         cfg.stop_for(system), cfg.hold_for(system), cfg.cooldown_after_stop,
                         cfg.partial_r if use_partial else None, cfg.partial_frac)
    feat_arr = {k: d[k].to_numpy() for k in FEATURE_COLS}
    rows = []
    for i, e, k, entry, stop, blended, reason in raw:
        feats = {f: feat_arr[f][i] for f in FEATURE_COLS}
        feats["is_pullback"] = 1.0 if system == "pullback" else 0.0
        rows.append({"ticker": ticker, "system": system, "signal_date": idx[i], "entry_date": idx[e], "exit_date": idx[k],
                     "entry_price": entry, "stop": stop, "exit_price": blended, "exit_reason": reason,
                     "bars": k - e + 1, "R_gross": (blended - entry) / (entry - stop), **feats})
    return pd.DataFrame(rows)


def latest_state(d: pd.DataFrame, cfg: StrategyConfig, system: str = "trend") -> dict:
    """Estado de la ÚLTIMA barra (hoy) para un sistema."""
    last = d.iloc[-1]
    ok = not np.isnan(last["atr"]) and not np.isnan(last["don_high"])
    feats = {k: float(last[k]) if not np.isnan(last[k]) else np.nan for k in FEATURE_COLS}
    feats["is_pullback"] = 1.0 if system == "pullback" else 0.0
    in_trade_zone = bool(last["breakout"]) if system == "trend" else bool(last["rsi2"] < cfg.pb_rsi_entry)
    return {
        "date": d.index[-1], "close": float(last["Close"]), "atr": float(last["atr"]) if ok else np.nan,
        "entry_signal": bool(last[_ENTRY_COL[system]]) and ok,
        "in_trend": in_trade_zone,
        "don_low": float(last["don_low"]) if not np.isnan(last["don_low"]) else np.nan,
        "stop": float(last["Close"] - cfg.stop_for(system) * last["atr"]) if ok else np.nan,
        "exit_signal": bool(last[_EXIT_COL[system]]),
        "features": feats,
    }
