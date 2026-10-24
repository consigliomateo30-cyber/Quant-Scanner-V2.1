"""Comparador de variantes. Cada variante que probás es una 'prueba' contra el azar: el resultado
reporta el nº de pruebas para que el Deflated Sharpe sea honesto."""
from __future__ import annotations

import dataclasses

import pandas as pd

from .backtest import run_backtest
from .config import AppConfig
from .metrics import deflated_sharpe, probabilistic_sharpe


def default_variants(base: AppConfig) -> dict[str, AppConfig]:
    s = base.strategy
    rep = dataclasses.replace
    return {
        "swing base (tendencia + pullback)": base,
        "solo tendencia": rep(base, strategy=rep(s, pb_enabled=False)),
        "tendencia + parcial 1.5R": rep(base, strategy=rep(s, partial_r=1.5)),
        "stop tendencia 2.5 ATR": rep(base, strategy=rep(s, stop_atr=2.5)),
        "pullback RSI2<5": rep(base, strategy=rep(s, pb_rsi_entry=5.0)),
        "tendencia 20/10 (más lenta)": rep(base, strategy=rep(s, entry_window=20, exit_window=10, max_hold=30)),
    }


def compare_variants(prices, variants: dict[str, AppConfig], capital: float = 5000.0) -> pd.DataFrame:
    n = len(variants)
    rows = {}
    for name, cfg in variants.items():
        out = run_backtest(prices, cfg, capital)
        if "error" in out:
            rows[name] = {"error": out["error"]}
            continue
        m, tm = out["modelo"]["metrics"], out["modelo"]["trade_metrics"]
        r = out["modelo"]["equity"].pct_change().dropna()
        rows[name] = {"CAGR": m.get("CAGR"), "MaxDD": m.get("MaxDD"), "Sharpe": m.get("Sharpe"),
                      "Trades": tm.get("Trades"), "Acierto": tm.get("Tasa de acierto"),
                      "R medio": tm.get("R medio (expectancy)"), "Profit factor": tm.get("Profit factor"),
                      "PSR": probabilistic_sharpe(r), f"DSR ({n} pruebas)": deflated_sharpe(r, n)}
    return pd.DataFrame(rows).T
