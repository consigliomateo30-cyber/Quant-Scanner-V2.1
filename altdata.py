"""Datos alternativos con RIGOR. Una señal alternativa solo puede entrar al modelo si pasa `validate_signal`:
muestra suficiente, diferencia de R medio con intervalo de confianza por encima de 0, y efecto estable en
las dos mitades del tiempo. Si no pasa, queda como información (se muestra, no pondera).

Señal implementada: compras agrupadas de insiders (Form 4, código P). Sustento: la literatura (Lakonishok y Lee,
Jeng-Metrick-Zeckhauser, Cohen-Malloy-Pomorski) encuentra retornos anormales tras compras de insiders, más
fuertes cuando compran varios a la vez y en empresas chicas. Ojo: nuestro universo son grandes empresas, donde
el efecto es más débil y las compras son raras: puede que nunca llegue a validarse, y está bien."""
from __future__ import annotations

import numpy as np
import pandas as pd


def insider_cluster_state(purchases: pd.DataFrame, min_insiders: int = 3, window_days: int = 30,
                          min_total_usd: float = 50_000.0) -> pd.DataFrame:
    """Estado del cluster en cada fecha de presentación (punto en el tiempo). `purchases` necesita:
    ticker, filed (timestamp), insider, value_usd. Solo compras abiertas (P)."""
    if purchases is None or purchases.empty:
        return pd.DataFrame(columns=["ticker", "asof", "n_insiders", "total_usd", "cluster"])
    p = purchases.copy()
    p["filed"] = pd.to_datetime(p["filed"], utc=True).dt.tz_localize(None)
    rows = []
    for tk, g in p.sort_values("filed").groupby("ticker"):
        for t in g["filed"].unique():
            w = g[(g["filed"] <= t) & (g["filed"] > t - pd.Timedelta(days=window_days))]
            n, tot = w["insider"].nunique(), float(w["value_usd"].sum())
            rows.append({"ticker": tk, "asof": pd.Timestamp(t).normalize(), "n_insiders": n, "total_usd": tot,
                         "cluster": bool(n >= min_insiders and tot >= min_total_usd)})
    return pd.DataFrame(rows)


def flag_trades(trades: pd.DataFrame, state: pd.DataFrame, lookback_days: int = 30) -> pd.Series:
    """True si, antes del día de la señal (estrictamente), hubo un cluster vigente. Sin look-ahead."""
    t = trades[["ticker", "signal_date"]].copy()
    t["_i"] = np.arange(len(t))
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    s = state[state["cluster"]].copy() if len(state) else state
    if s is None or s.empty:
        return pd.Series(False, index=trades.index)
    s = s[["ticker", "asof"]].sort_values("asof")
    m = pd.merge_asof(t.sort_values("signal_date"), s, left_on="signal_date", right_on="asof", by="ticker",
                      direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(days=lookback_days))
    flag = m.sort_values("_i")["asof"].notna().to_numpy()
    return pd.Series(flag, index=trades.index)


def validate_signal(trades: pd.DataFrame, flag: pd.Series, min_flagged: int = 30, n_boot: int = 3000, seed: int = 0) -> dict:
    """¿Los trades con la señal rinden distinto (en R neto) que los demás, de forma estable?"""
    d = trades.assign(_f=flag.to_numpy()).dropna(subset=["R_net"]).sort_values("signal_date")
    a, b = d.loc[d["_f"], "R_net"].to_numpy(), d.loc[~d["_f"], "R_net"].to_numpy()
    out = {"n_con_señal": len(a), "n_sin_señal": len(b), "reasons": []}
    if len(a) < 3 or len(b) < 3:
        out.update(decision="solo informativo", reasons=["muy pocos trades con la señal"])
        return out
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])
    diff = a.mean() - b.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    half = len(d) // 2
    first, second = d.iloc[:half], d.iloc[half:]
    def eff(x):
        fa, fb = x.loc[x["_f"], "R_net"], x.loc[~x["_f"], "R_net"]
        return (fa.mean() - fb.mean()) if len(fa) >= 5 and len(fb) >= 5 else np.nan
    e1, e2 = eff(first), eff(second)
    out.update(R_con=float(a.mean()), R_sin=float(b.mean()), diferencia=float(diff), ic95=(float(lo), float(hi)),
               efecto_mitad_1=None if np.isnan(e1) else float(e1), efecto_mitad_2=None if np.isnan(e2) else float(e2))
    if len(a) < min_flagged:
        out["reasons"].append(f"muestra chica ({len(a)} < {min_flagged})")
    if lo <= 0:
        out["reasons"].append("el intervalo de confianza incluye 0")
    if not (e1 > 0 and e2 > 0):
        out["reasons"].append("el efecto no es positivo y estable en ambas mitades del período")
    out["decision"] = "graduar al modelo" if not out["reasons"] else "solo informativo"
    return out
