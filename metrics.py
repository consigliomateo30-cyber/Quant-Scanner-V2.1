"""Métricas de performance y de riesgo, incluyendo PSR y DSR (Bailey & López de Prado)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649


def periods_per_year(index: pd.DatetimeIndex) -> float:
    years = max((index[-1] - index[0]).days / 365.25, 1e-9)
    return (len(index) - 1) / years


def equity_metrics(equity: pd.Series, rf_annual: float = 0.0) -> dict:
    eq = equity.dropna()
    if len(eq) < 3:
        return {}
    ret = eq.pct_change().dropna()
    ppy = periods_per_year(eq.index)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * math.sqrt(ppy)
    ex = ret - rf_annual / ppy
    sharpe = ex.mean() / ret.std() * math.sqrt(ppy) if ret.std() > 0 else 0.0
    dn = ret[ret < 0].std()
    sortino = ex.mean() / dn * math.sqrt(ppy) if dn and dn > 0 else 0.0
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    return {
        "CAGR": cagr, "Volatilidad": vol, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDD": max_dd, "Calmar": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
        "Retorno total": eq.iloc[-1] / eq.iloc[0] - 1, "Años": years,
    }


def trade_metrics(executed: pd.DataFrame) -> dict:
    if executed is None or executed.empty:
        return {"Trades": 0}
    r = executed["R_net"]
    wins, losses = r[r > 0], r[r <= 0]
    gp, gl = wins.sum(), abs(losses.sum())
    return {
        "Trades": int(len(r)), "Tasa de acierto": float((r > 0).mean()),
        "R medio (expectancy)": float(r.mean()),
        "Ganancia media (R)": float(wins.mean()) if len(wins) else 0.0,
        "Pérdida media (R)": float(losses.mean()) if len(losses) else 0.0,
        "Profit factor": float(gp / gl) if gl > 0 else float("inf"),
        "Duración media (barras)": float(executed["bars"].mean()),
    }


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: P(Sharpe verdadero > benchmark) corrigiendo por asimetría, curtosis y largo de la muestra.
    Todo en frecuencia de la serie (NO anualizado)."""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    g3 = r.skew()
    g4 = r.kurtosis() + 3.0            # pandas devuelve curtosis en exceso; la fórmula usa la cruda
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_trials_var: float | None = None) -> float:
    """DSR: PSR contra el Sharpe máximo esperable por azar tras probar n_trials variantes.
    n_trials = cuántas configuraciones/estrategias probaste (sé honesto con ese número)."""
    r = returns.dropna()
    if len(r) < 10 or n_trials < 1:
        return float("nan")
    if n_trials == 1:
        return probabilistic_sharpe(r, 0.0)
    var = sr_trials_var if sr_trials_var is not None else (1.0 / max(len(r) - 1, 1))
    sr_star = math.sqrt(var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
                                + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e)))
    return probabilistic_sharpe(r, sr_star)


def monte_carlo_trades(r_multiples: np.ndarray, risk_pct: float, n_trades: int = 100,
                       n_sims: int = 5000, seed: int = 7) -> dict:
    """Bootstrap sobre los R reales de los trades (no sobre retornos diarios): simula
    secuencias de n_trades con riesgo fijo por trade y mide drawdown y probabilidad de pérdida."""
    r = np.asarray(r_multiples, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n_sims, n_trades), replace=True)
    curve = np.cumprod(1.0 + risk_pct / 100.0 * draws, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), curve], axis=1), axis=1)[:, 1:]
    dd = (curve / peak - 1.0).min(axis=1)
    final = curve[:, -1] - 1.0
    return {
        "P(pérdida al final)": float((final < 0).mean()),
        "Retorno mediano": float(np.median(final)),
        "Retorno P5": float(np.percentile(final, 5)),
        "Retorno P95": float(np.percentile(final, 95)),
        "Drawdown mediano": float(np.median(dd)),
        "Drawdown P95 (peor 5%)": float(np.percentile(dd, 5)),
        "P(drawdown > 20%)": float((dd < -0.20).mean()),
        "P(drawdown > 30%)": float((dd < -0.30).mean()),
    }
