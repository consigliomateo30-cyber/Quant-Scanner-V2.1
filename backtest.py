"""Backtest walk-forward a nivel CARTERA: costos, tope de posiciones, sin apalancamiento, freno por
drawdown y meta-modelo entrenado SOLO con trades ya cerrados en cada fecha de decisión."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig, CostConfig, MARKET_PROXY, ModelConfig, RiskConfig, VIX_TICKER
from .metrics import equity_metrics, trade_metrics
from .model import MetaModel
from .signals import add_indicators, regime_series, simulate_trades, vix_rank_series
from .sizing import dd_scale, position_units, risk_pct_for


def market_context(prices: dict[str, pd.DataFrame], cfg: AppConfig):
    """Series de contexto compartidas: régimen, retorno 20d de SPY y percentil del VIX."""
    regime = regime_series(prices.get(MARKET_PROXY), cfg.strategy.regime_sma) if cfg.strategy.use_regime else None
    spy = prices.get(MARKET_PROXY)
    spy_ret20 = spy["Close"].pct_change(20) if spy is not None else None
    return regime, spy_ret20, vix_rank_series(prices.get(VIX_TICKER))


def tradable(prices: dict[str, pd.DataFrame]) -> list[str]:
    return [tk for tk in prices if not tk.startswith("^")]


def build_trades(prices: dict[str, pd.DataFrame], cfg: AppConfig) -> pd.DataFrame:
    """Todos los trades históricos (todos los activos y sistemas) con R neto de costos."""
    regime, spy_ret20, vix_rank = market_context(prices, cfg)
    frames = []
    for tk in tradable(prices):
        df = prices[tk]
        if len(df) < 260:
            continue
        d = add_indicators(df, cfg.strategy, regime, spy_ret20, vix_rank)
        for system in cfg.systems():
            t = simulate_trades(d, cfg.strategy, tk, system)
            if not t.empty:
                frames.append(t)
    if not frames:
        return pd.DataFrame()
    t = pd.concat(frames, ignore_index=True)
    c = cfg.costs.per_side
    risk = (t["entry_price"] - t["stop"])
    t["R_net"] = (t["exit_price"] * (1 - c) - t["entry_price"] * (1 + c)) / risk
    return t.sort_values("signal_date").reset_index(drop=True)


def walk_forward_scores(trades: pd.DataFrame, mcfg: ModelConfig) -> pd.DataFrame:
    """Agrega p (prob. calibrada), b (payoff del sistema) y edge = p*b-(1-p) a cada trade, usando
    únicamente trades cuyo exit_date es anterior a la fecha de decisión. Sin fuga de información."""
    t = trades.sort_values("signal_date").reset_index(drop=True).copy()
    t["p"], t["b"] = np.nan, np.nan
    if t.empty:
        return t
    step = pd.Timedelta(days=int(mcfg.retrain_every_days * 365 / 252))
    cur, end = t["signal_date"].min(), t["signal_date"].max()
    while cur <= end:
        nxt = cur + step
        mask = (t["signal_date"] >= cur) & (t["signal_date"] < nxt)
        train = t[t["exit_date"] < cur]
        if mask.any() and len(train) >= mcfg.min_train_trades:
            m = MetaModel(mcfg).fit(train)
            t.loc[mask, "p"] = m.predict(t.loc[mask])
            t.loc[mask, "b"] = [m.payoff_for(s) for s in t.loc[mask, "system"]]
        cur = nxt
    t["edge"] = t["p"] * t["b"] - (1 - t["p"])
    return t


def run_portfolio(trades: pd.DataFrame, close: pd.DataFrame, capital: float, rc: RiskConfig,
                  costs: CostConfig, use_model: bool) -> tuple[pd.Series, pd.DataFrame]:
    cps = costs.per_side
    cs = close.ffill()
    t = trades[trades["p"].notna()] if use_model else trades
    by_day = {d: g for d, g in t.groupby("entry_date")}
    cash, positions, executed, eq = float(capital), [], [], []
    last_equity, peak, min_cash = float(capital), float(capital), float(capital)

    def do_exits(day):
        nonlocal cash
        keep = []
        for p in positions:
            if p["exit_date"] <= day:
                proceeds = p["units"] * p["exit_price"] * (1 - cps)
                cash += proceeds
                pnl = proceeds - p["cost"]
                executed.append({"ticker": p["ticker"], "system": p["system"], "entry_date": p["entry_date"],
                                 "exit_date": p["exit_date"], "units": p["units"], "exit_reason": p["exit_reason"],
                                 "bars": p["bars"], "entry_price": p["entry_price"], "exit_price": p["exit_price"],
                                 "pnl": pnl, "R_net": pnl / p["risk_usd"] if p["risk_usd"] > 0 else np.nan})
            else:
                keep.append(p)
        positions[:] = keep

    for day in cs.index:
        do_exits(day)
        g = by_day.get(day)
        if g is not None:
            scale = dd_scale(1.0 - last_equity / peak, rc)
            if use_model:
                g = g.sort_values("edge", ascending=False)
            for _, tr in g.iterrows():
                if scale <= 0 or len(positions) >= rc.max_positions or any(p["ticker"] == tr["ticker"] for p in positions):
                    continue
                rp = risk_pct_for(tr["p"] if use_model else None, tr["b"] if use_model else None, rc) * scale
                units = position_units(last_equity, tr["entry_price"], tr["stop"], rp, rc, cash, cps)
                if units <= 0:
                    continue
                cost = units * tr["entry_price"] * (1 + cps)
                cash -= cost
                min_cash = min(min_cash, cash)
                positions.append({"ticker": tr["ticker"], "system": tr.get("system", "trend"), "units": units, "cost": cost,
                                  "entry_date": day, "exit_date": tr["exit_date"],
                                  "entry_price": tr["entry_price"], "exit_price": tr["exit_price"],
                                  "exit_reason": tr["exit_reason"], "bars": tr["bars"],
                                  "risk_usd": units * (tr["entry_price"] * (1 + cps) - tr["stop"])})
            do_exits(day)
        px = cs.loc[day]
        last_equity = cash + sum(p["units"] * px[p["ticker"]] for p in positions)
        peak = max(peak, last_equity)
        eq.append(last_equity)
    out = pd.Series(eq, index=cs.index, name="equity")
    out.attrs["min_cash"] = min_cash
    return out, pd.DataFrame(executed)


def edge_curve(scored: pd.DataFrame, thresholds=(-9.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """Resultado REAL (fuera de muestra) de los trades según el edge que estimó el modelo.
    Permite elegir el umbral con evidencia: cuánto sube el acierto/R medio y cuántos trades se pierden."""
    s = scored.dropna(subset=["edge", "R_net"])
    rows = []
    for thr in thresholds:
        g = s[s["edge"] >= thr]
        if len(g) == 0:
            continue
        r = g["R_net"]
        gl = abs(r[r <= 0].sum())
        rows.append({"edge_min": thr if thr > -9 else None, "trades": len(g), "acierto": float((r > 0).mean()),
                     "R_medio": float(r.mean()), "profit_factor": float(r[r > 0].sum() / gl) if gl > 0 else float("inf")})
    return pd.DataFrame(rows)


def calibration(scored: pd.DataFrame, bins: int = 8) -> dict:
    """Brier score y tabla de confiabilidad: ¿cuando el modelo dice 60%, gana ~60%?"""
    s = scored.dropna(subset=["p", "R_net"]).copy()
    if len(s) < 30:
        return {}
    y = (s["R_net"] > 0).astype(float)
    brier = float(((s["p"] - y) ** 2).mean())
    base = float(((y.mean() - y) ** 2).mean())
    s["bin"] = pd.qcut(s["p"], q=min(bins, s["p"].nunique()), duplicates="drop")
    tab = s.groupby("bin", observed=True).apply(lambda g: pd.Series({"p_media": g["p"].mean(), "acierto_real": (g["R_net"] > 0).mean(), "n": len(g)}), include_groups=False)
    return {"brier": brier, "brier_base": base, "tabla": tab.reset_index(drop=True)}


def by_system(executed: pd.DataFrame) -> pd.DataFrame:
    if executed is None or executed.empty or "system" not in executed:
        return pd.DataFrame()
    rows = {}
    for sname, g in executed.groupby("system"):
        rows[sname] = trade_metrics(g)
    return pd.DataFrame(rows).T


def run_backtest(prices: dict[str, pd.DataFrame], cfg: AppConfig, capital: float = 5000.0) -> dict:
    """Corre línea base (sin meta-modelo) y con meta-modelo sobre la MISMA ventana, más SPY."""
    trades = build_trades(prices, cfg)
    if trades.empty:
        return {"error": "No se generaron trades con estos datos/parámetros."}
    scored = walk_forward_scores(trades, cfg.model)
    if scored["p"].notna().sum() == 0:
        return {"error": "Pocos trades para entrenar el meta-modelo: ampliá el universo o los años de historia.",
                "trades": scored}
    close = pd.DataFrame({tk: prices[tk]["Close"] for tk in tradable(prices)}).sort_index()
    start = scored.loc[scored["p"].notna(), "signal_date"].min()

    out = {"trades_all": scored, "eval_start": start, "edge_curve": edge_curve(scored), "calibration": calibration(scored)}
    for name, use_model in (("base", False), ("modelo", True)):
        eq, ex = run_portfolio(scored, close, capital, cfg.risk, cfg.costs, use_model)
        eq = eq.loc[start:]
        eq = eq / eq.iloc[0] * capital
        out[name] = {"equity": eq, "executed": ex, "metrics": equity_metrics(eq),
                     "trade_metrics": trade_metrics(ex), "by_system": by_system(ex)}
    if MARKET_PROXY in prices:
        spy = prices[MARKET_PROXY]["Close"].loc[start:]
        out["spy"] = {"equity": spy / spy.iloc[0] * capital, "metrics": equity_metrics(spy / spy.iloc[0] * capital)}
    return out
