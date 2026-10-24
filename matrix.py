"""Matriz de investigación: estrategia x clase de activo/industria x temporalidad.

Embudo profesional (para no autoengañarse al probar cientos de combinaciones):
  1) Screening a nivel de trades, SOLO con datos de desarrollo (hasta dev_end), con costos.
  2) Corrección por comparaciones múltiples (Benjamini-Hochberg, FDR) sobre TODAS las celdas probadas.
  3) Intervalo de confianza por bootstrap por mes (los trades de un mismo mes no son independientes). Se testea el
     EXCESO sobre la deriva pasiva del activo (estar invertido gana la deriva sin timing; además mitiga el sesgo de supervivencia).
  4) Las celdas que pasan se miran UNA vez en el período reservado (holdout, después de dev_end).
  5) Margen de costo: cuánto costo por lado aguanta cada celda antes de dejar de ganar (break-even).
Solo las celdas con nivel A/B son candidatas a pasar al backtest de cartera con meta-modelo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, resample_ohlcv
from .signals import run_trade_loop
from .specs import Spec, default_specs, signals_for
from .universe import COST_SCENARIOS_SIDE_PCT, GROUPS, GROUP_COST_SIDE_PCT, MARKET_PROXY

MIN_TRADES_TEST = 30      # mínimo para entrar al test
MIN_TRADES_PASS = 50      # mínimo para poder "pasar"
Q_MAX = 0.10
MIN_OOS_TRADES = 20


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def cluster_bootstrap(values, clusters, n_boot: int = 1000, seed: int = 0) -> tuple[float, tuple[float, float]]:
    """p-valor unilateral (H1: media > 0) y IC95 de la media, remuestreando clusters (meses) completos."""
    v = np.asarray(values, dtype=float)
    codes, _ = pd.factorize(pd.Series(clusters))
    sums, cnts = np.bincount(codes, weights=v), np.bincount(codes)
    m = len(sums)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    p = (np.sum(means <= 0) + 1) / (n_boot + 1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(p), (float(lo), float(hi))


def frames_for(prices: dict[str, pd.DataFrame], tickers: list[str], timeframe: str) -> dict[str, pd.DataFrame]:
    out = {}
    for tk in tickers:
        df = prices.get(tk)
        if df is None or len(df) < 300:
            continue
        out[tk] = df if timeframe == "1d" else resample_ohlcv(df)
        if len(out[tk]) < 80:
            out.pop(tk)
    return out


def regime_for(prices: dict[str, pd.DataFrame], timeframe: str):
    spy = prices.get(MARKET_PROXY)
    if spy is None:
        return None, None
    d = spy if timeframe == "1d" else resample_ohlcv(spy)
    n = 200 if timeframe == "1d" else 40
    return d["Close"] > d["Close"].rolling(n).mean(), d["Close"]


def trades_for(d: pd.DataFrame, spec: Spec, regime, bench, ticker: str, cost_side: float) -> pd.DataFrame:
    entry, exit_ = signals_for(d, spec, regime, bench)
    raw = run_trade_loop(d["Open"].to_numpy(), d["High"].to_numpy(), d["Low"].to_numpy(), d["Close"].to_numpy(),
                         atr(d, 14).to_numpy(), entry, exit_, spec.stop_atr, spec.max_hold,
                         cooldown=5 if spec.timeframe == "1d" else 1)
    if not raw:
        return pd.DataFrame()
    idx = d.index
    mu = float(d["Close"].pct_change().mean())          # deriva pasiva por barra: control, NO se usa para operar
    rows = []
    for i, e, k, en, st, ex, reason in raw:
        bars = k - e + 1
        r_net = (ex * (1 - cost_side) - en * (1 + cost_side)) / (en - st)
        rows.append({"ticker": ticker, "entry_date": idx[e], "exit_date": idx[k], "bars": bars,
                     "ret_gross": ex / en - 1, "ret_net": ex * (1 - cost_side) / (en * (1 + cost_side)) - 1,
                     "R_net": r_net, "R_exc": r_net - mu * bars * en / (en - st),
                     "ret_exc": ex / en - 1 - mu * bars, "reason": reason})
    return pd.DataFrame(rows)


def _basic(tr: pd.DataFrame) -> dict:
    if tr is None or len(tr) == 0:
        return {"n": 0, "wr": np.nan, "R": np.nan, "Rexc": np.nan, "PF": np.nan, "gross": np.nan, "gexc": np.nan, "net": np.nan, "bars": np.nan}
    r = tr["R_net"]
    gl = abs(r[r <= 0].sum())
    return {"n": len(tr), "wr": float((r > 0).mean()), "R": float(r.mean()), "Rexc": float(tr["R_exc"].mean()),
            "PF": float(r[r > 0].sum() / gl) if gl > 0 else float("inf"),
            "gross": float(tr["ret_gross"].mean() * 100), "gexc": float(tr["ret_exc"].mean() * 100),
            "net": float(tr["ret_net"].mean() * 100), "bars": float(tr["bars"].mean())}


def cell_row(group: str, tf: str, spec: Spec, tr: pd.DataFrame, dev_end: pd.Timestamp, n_tickers: int,
             n_boot: int, assumed_cost_pct: float) -> dict:
    dev = tr[tr["entry_date"] <= dev_end] if len(tr) else tr
    hold = tr[tr["entry_date"] > dev_end] if len(tr) else tr
    a, b = _basic(dev), _basic(hold)
    p, ci = (1.0, (np.nan, np.nan))
    if a["n"] >= MIN_TRADES_TEST:
        p, ci = cluster_bootstrap(dev["R_exc"].to_numpy(), dev["entry_date"].dt.to_period("M").astype(str), n_boot)
    years = max((dev_end - dev["entry_date"].min()).days / 365.25, 0.5) if a["n"] else np.nan
    return {"grupo": group, "tf": tf, "familia": spec.family, "spec": spec.name, "n_tickers": n_tickers,
            "n_dev": a["n"], "acierto_dev": a["wr"], "R_dev": a["R"], "R_exc_dev": a["Rexc"], "PF_dev": a["PF"], "bruto_dev_pct": a["gross"],
            "neto_dev_pct": a["net"], "barras": a["bars"], "p_dev": p, "ic_bajo": ci[0],
            "trades_x_activo_año": a["n"] / (n_tickers * years) if a["n"] else np.nan,
            "break_even_lado_pct": a["gexc"] / 2 if a["n"] else np.nan,
            "costo_supuesto_lado_pct": assumed_cost_pct,
            "n_hold": b["n"], "acierto_hold": b["wr"], "R_hold": b["R"], "R_exc_hold": b["Rexc"], "bruto_hold_pct": b["gexc"]}


def add_significance(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    testable = d["n_dev"] >= MIN_TRADES_TEST
    d["q_dev"] = 1.0
    if testable.any():
        d.loc[testable, "q_dev"] = benjamini_hochberg(d.loc[testable, "p_dev"].to_numpy())
    d["pasa_dev"] = ((d["n_dev"] >= MIN_TRADES_PASS) & (d["R_dev"] > 0) & (d["R_exc_dev"] > 0) & (d["q_dev"] <= Q_MAX)
                     & (d["PF_dev"] > 1.1) & (d["ic_bajo"] > 0))
    d["margen_costo"] = d["break_even_lado_pct"] / d["costo_supuesto_lado_pct"]
    def oos(r):
        if not r["pasa_dev"]:
            return "—"
        if r["n_hold"] < MIN_OOS_TRADES:
            return "OOS insuficiente"
        return "confirmado OOS" if (r["R_hold"] > 0 and r["R_exc_hold"] > 0 and r["bruto_hold_pct"] / 2 > r["costo_supuesto_lado_pct"]) else "falló OOS"
    d["oos"] = d.apply(oos, axis=1)
    def nivel(r):
        if r["oos"] == "confirmado OOS" and r["q_dev"] <= 0.05 and r["margen_costo"] >= 3:
            return "A"
        if r["oos"] == "confirmado OOS" and r["margen_costo"] >= 2:
            return "B"
        return "C" if r["pasa_dev"] else "D"
    d["nivel"] = d.apply(nivel, axis=1)
    for label, c in COST_SCENARIOS_SIDE_PCT.items():
        d[f"aguanta {label}"] = d["break_even_lado_pct"] >= c
    return d


def run_matrix(prices: dict[str, pd.DataFrame], groups: dict[str, list[str]] | None = None,
               specs_by_tf: dict[str, list[Spec]] | None = None, dev_end: str = "2021-12-31",
               n_boot: int = 1000) -> pd.DataFrame:
    groups = groups or GROUPS
    specs_by_tf = specs_by_tf or {"1d": default_specs("1d"), "1w": default_specs("1w")}
    dev_ts = pd.Timestamp(dev_end)
    rows = []
    for tf, specs in specs_by_tf.items():
        reg, bench = regime_for(prices, tf)
        for g, tks in groups.items():
            frames = frames_for(prices, tks, tf)
            if not frames:
                continue
            cost = GROUP_COST_SIDE_PCT.get(g, 0.10)
            for spec in specs:
                parts = [trades_for(d, spec, reg, bench, tk, cost / 100) for tk, d in frames.items()]
                parts = [x for x in parts if len(x)]
                tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                rows.append(cell_row(g, tf, spec, tr, dev_ts, len(frames), n_boot, cost))
    if not rows:
        return pd.DataFrame()
    return add_significance(pd.DataFrame(rows)).sort_values(["nivel", "R_dev"], ascending=[True, False]).reset_index(drop=True)


def operability_map(df: pd.DataFrame) -> pd.DataFrame:
    """Mejor celda por (grupo, temporalidad) entre las que no son D."""
    ok = df[df["nivel"].isin(["A", "B", "C"])]
    if ok.empty:
        return ok
    return ok.sort_values(["nivel", "R_dev"], ascending=[True, False]).groupby(["grupo", "tf"], as_index=False).first()


def render_report(df: pd.DataFrame, dev_end: str) -> str:
    n = len(df)
    lv = df["nivel"].value_counts().to_dict()
    lines = [f"# Matriz de investigación (desarrollo hasta {dev_end}; fuera de muestra después)", "",
             f"- Celdas probadas: **{n}** (cada una es una prueba; se corrige con FDR de Benjamini-Hochberg, q ≤ {Q_MAX}).",
             f"- Niveles: " + ", ".join(f"{k}: {lv.get(k, 0)}" for k in "ABCD"), "",
             "Escala: **A** confirmada fuera de muestra, q ≤ 0.05 y aguanta ≥ 3× el costo supuesto · **B** confirmada OOS y aguanta ≥ 2× · "
             "**C** pasa desarrollo pero sin confirmar OOS o con poco margen de costo · **D** no pasa.", ""]
    top = df[df["nivel"].isin(["A", "B", "C"])].head(30)
    if top.empty:
        lines += ["**Ninguna celda superó los filtros.** Es un resultado válido: significa que no hay evidencia suficiente para operar "
                  "ninguna de estas combinaciones con estos datos y costos. No bajes el listón para que aparezca algo.", ""]
    else:
        cols = ["nivel", "grupo", "tf", "spec", "n_dev", "acierto_dev", "R_dev", "R_exc_dev", "PF_dev", "q_dev", "break_even_lado_pct", "oos", "n_hold", "R_exc_hold"]
        lines += ["## Celdas candidatas", "", top[cols].round(3).to_markdown(index=False) if hasattr(top, "to_markdown") else top[cols].round(3).to_string(index=False), ""]
    lines += ["## Advertencias", "",
              "- Las acciones individuales tienen sesgo de supervivencia (empresas que hoy existen y son grandes): los ETFs son la prueba más limpia.",
              "- `break_even_lado_pct` es el costo por lado al que la celda deja de ganar. Comparalo con TU costo real (CEDEAR: spread + comisión + diferencia de CCL).",
              "- Resultados pasados no garantizan resultados futuros. Antes de usar plata real: backtest de cartera con meta-modelo y 2-3 meses de paper trading."]
    return "\n".join(lines)
