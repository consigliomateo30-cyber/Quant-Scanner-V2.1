"""Escaneo del día: señales de cada sistema sobre la última barra + probabilidad del meta-modelo."""
from __future__ import annotations

import pandas as pd

from .backtest import build_trades, market_context, tradable
from .config import AppConfig, FEATURE_COLS
from .data import freshness_report
from .model import MetaModel
from .signals import add_indicators, latest_state
from .sizing import kelly_fraction, risk_pct_for

LABEL = {"trend": "Tendencia", "pullback": "Pullback"}
SCAN_COLUMNS = ["ticker", "sistema", "fecha_dato", "dias_de_atraso", "cierre", "entrada", "en_zona", "salida", "p_ganar", "payoff",
                "edge", "kelly", "riesgo_pct", "stop_sugerido", "min_donchian", "atr"]


def scan_today(prices: dict[str, pd.DataFrame], cfg: AppConfig) -> tuple[pd.DataFrame, MetaModel, dict]:
    trades = build_trades(prices, cfg)
    model = MetaModel(cfg.model).fit(trades) if not trades.empty else MetaModel(cfg.model)
    regime, spy_ret20, vix_rank = market_context(prices, cfg)
    regime_ok = bool(regime.iloc[-1]) if regime is not None and len(regime) else (not cfg.strategy.use_regime)
    rows, frames = [], {}
    today = pd.Timestamp.now().normalize()
    for tk in tradable(prices):
        df = prices[tk]
        if len(df) < 260:
            continue
        d = add_indicators(df, cfg.strategy, regime, spy_ret20, vix_rank)
        frames[tk] = d
        for system in cfg.systems():
            st = latest_state(d, cfg.strategy, system)
            feats = pd.DataFrame([st["features"]])[FEATURE_COLS]
            p = float(model.predict(feats)[0])
            b = model.payoff_for(system)
            rows.append({
                "ticker": tk, "sistema": system, "fecha_dato": st["date"], "dias_de_atraso": int((today - st["date"]).days),
                "cierre": st["close"], "entrada": st["entry_signal"], "en_zona": st["in_trend"],
                "salida": st["exit_signal"], "p_ganar": p, "payoff": b, "edge": p * b - (1 - p),
                "kelly": kelly_fraction(p, b), "riesgo_pct": risk_pct_for(p, b, cfg.risk),
                "stop_sugerido": st["stop"], "min_donchian": st["don_low"], "atr": st["atr"],
            })
    out = pd.DataFrame(rows, columns=SCAN_COLUMNS)
    fr = freshness_report(prices)
    out["al_dia"] = out["ticker"].map(fr.set_index("ticker")["al_dia"]).fillna(False).astype(bool)
    info = {"regime_ok": regime_ok, "model_mode": model.mode, "n_train": model.n_train,
            "prior": model.prior, "payoff": model.payoff, "payoff_sys": dict(model.payoff_sys)}
    return out, model, {**info, "frames": frames}
