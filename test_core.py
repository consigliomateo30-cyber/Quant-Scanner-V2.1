import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from quant_scanner.config import AppConfig, StrategyConfig, RiskConfig, CostConfig
from quant_scanner.signals import add_indicators, simulate_trades, latest_state
from quant_scanner.sizing import position_units, risk_pct_for, kelly_fraction, dd_scale
from quant_scanner.model import MetaModel
from quant_scanner.backtest import edge_curve, calibration
from quant_scanner.backtest import build_trades, walk_forward_scores, run_portfolio, run_backtest
from quant_scanner.metrics import probabilistic_sharpe, deflated_sharpe, monte_carlo_trades
from synth import make_prices

CFG = AppConfig(strategy=StrategyConfig(use_regime=False))


def test_no_lookahead():
    px = make_prices(3, 900)["A0"]
    full = add_indicators(px, CFG.strategy)
    for k in (400, 600, 850):
        part = add_indicators(px.iloc[:k], CFG.strategy)
        for col in ("atr", "don_high", "don_low", "rsi14", "vol_20d", "entry_signal"):
            a, b = full[col].iloc[k - 1], part[col].iloc[-1]
            assert (pd.isna(a) and pd.isna(b)) or a == b, (col, k, a, b)


def test_entry_is_next_open_and_gap_stop():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = np.r_[np.full(40, 100.0), 110.0, 111.0, 112.0, np.full(17, 112.0)]
    df = pd.DataFrame({"Open": close.copy(), "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1e6}, index=idx)
    df.iloc[41, df.columns.get_loc("Open")] = 110.5             # apertura de la barra de ejecución
    df.iloc[42, df.columns.get_loc("Open")] = 80.0              # gap bajista violento
    df.iloc[42, df.columns.get_loc("Low")] = 79.0
    d = add_indicators(df, CFG.strategy)
    t = simulate_trades(d, CFG.strategy, "X")
    assert len(t) == 1
    r = t.iloc[0]
    assert r["entry_price"] == 110.5 and r["entry_date"] == idx[41] and r["signal_date"] == idx[40]
    assert r["exit_reason"] == "stop" and r["exit_price"] == 80.0   # gap: sale en la apertura, peor que el stop
    assert r["R_gross"] < -1.0


def test_latest_state_uses_last_bar():
    px = make_prices(2, 500)["A0"].copy()
    px.iloc[-1, px.columns.get_loc("Close")] = px["High"].iloc[-21:-1].max() * 1.05
    px.iloc[-1, px.columns.get_loc("High")] = px["Close"].iloc[-1]
    st = latest_state(add_indicators(px, CFG.strategy), CFG.strategy)
    assert st["entry_signal"] and st["date"] == px.index[-1] and st["close"] == px["Close"].iloc[-1]


def test_sizing_caps():
    rc = RiskConfig(max_pos_pct=25, risk_flat_pct=1, min_trade_usd=0)
    u = position_units(10_000, 100, 99.5, 1.0, rc)              # stop muy corto: el tope del 25% manda
    assert u * 100 <= 2_500 + 1e-9
    u = position_units(10_000, 100, 90, 1.0, rc, cash_available=300)
    assert u * 100 <= 300
    assert position_units(100, 500, 480, 1.0, rc) == 0           # capital chico: no alcanza ni 1 unidad
    assert risk_pct_for(0.30, 1.0, rc) == 0.0                    # edge negativo -> no opera
    assert 0 < risk_pct_for(0.55, 2.0, rc) <= rc.risk_max_pct
    assert math.isclose(kelly_fraction(0.5, 2.0), 0.25)


def test_walk_forward_no_leakage():
    px = make_prices(14, 2500, seed=3)
    tr = build_trades(px, CFG)
    s1 = walk_forward_scores(tr, CFG.model)
    assert s1["p"].notna().sum() > 0
    # cambiar el resultado de los últimos trades (futuro) no puede cambiar los scores anteriores
    tr2 = tr.copy()
    cutoff = tr2["signal_date"].quantile(0.8)
    tr2.loc[tr2["signal_date"] >= cutoff, "R_net"] = tr2.loc[tr2["signal_date"] >= cutoff, "R_net"] * -5
    s2 = walk_forward_scores(tr2, CFG.model)
    early = s1["signal_date"] < cutoff - pd.Timedelta(days=200)
    a, b = s1.loc[early, "p"].to_numpy(), s2.loc[early, "p"].to_numpy()
    assert np.allclose(a, b, equal_nan=True)
    # y nunca hay score si no hay suficientes trades cerrados
    first = s1.loc[s1["p"].notna(), "signal_date"].min()
    closed = (tr["exit_date"] < first).sum()
    assert closed >= CFG.model.min_train_trades


def test_portfolio_constraints():
    px = make_prices(14, 2500, seed=5)
    tr = walk_forward_scores(build_trades(px, CFG), CFG.model)
    close = pd.DataFrame({k: v["Close"] for k, v in px.items()})
    rc = RiskConfig(max_positions=4, allow_fractional=True, min_trade_usd=1)
    for use_model in (False, True):
        eq, ex = run_portfolio(tr, close, 5000, rc, CFG.costs, use_model)
        assert (eq > 0).all() and np.isfinite(eq).all()
        if not ex.empty:
            ev = pd.concat([pd.Series(1, index=ex["entry_date"]), pd.Series(-1, index=ex["exit_date"])]).groupby(level=0).sum()
            ev = ev.sort_index()
            assert ev.cumsum().max() <= rc.max_positions


def test_metrics():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1500))
    psr = probabilistic_sharpe(r, 0.0)
    assert 0.9 < psr <= 1.0
    assert deflated_sharpe(r, 100) < psr                          # probar muchas variantes penaliza
    noise = pd.Series(rng.normal(0, 0.01, 1500))
    assert probabilistic_sharpe(noise, 0.0) < 0.99
    mc = monte_carlo_trades(rng.normal(0.2, 1.3, 300), 1.0, 100, 2000)
    assert 0 <= mc["P(drawdown > 20%)"] <= 1 and mc["Drawdown P95 (peor 5%)"] <= 0


def _frame(close, open_=None, high=None, low=None):
    idx = pd.bdate_range("2019-01-01", periods=len(close))
    close = np.asarray(close, float)
    open_ = close.copy() if open_ is None else np.asarray(open_, float)
    high = np.maximum(open_, close) + 0.3 if high is None else np.asarray(high, float)
    low = np.minimum(open_, close) - 0.3 if low is None else np.asarray(low, float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)


def test_pullback_signal_causality_and_trade():
    close = np.r_[np.linspace(50, 150, 260), 147, 144, 141, 143, 147, 152, 153, 154, 155, 156, 157, 158]
    df = _frame(close)
    cfg = StrategyConfig(use_regime=False)
    d = add_indicators(df, cfg)
    k = 262                                              # tercera caída: RSI(2) muy bajo dentro de tendencia alcista
    assert d["entry_pullback"].iloc[k] and d["rsi2"].iloc[k] < cfg.pb_rsi_entry
    part = add_indicators(df.iloc[: k + 1], cfg)          # sin look-ahead: mismo resultado con datos cortados
    assert part["entry_pullback"].iloc[-1] and abs(part["rsi2"].iloc[-1] - d["rsi2"].iloc[k]) < 1e-12
    t = simulate_trades(d, cfg, "X", "pullback")
    assert len(t) >= 1
    r = t.iloc[0]
    pos = df.index.get_loc(r["signal_date"])                      # la señal pudo darse un día antes: se valida la regla general
    assert d["entry_pullback"].iloc[pos] and pos <= k
    assert r["system"] == "pullback" and r["entry_price"] == df["Open"].iloc[pos + 1] and r["entry_date"] == df.index[pos + 1]
    assert r["exit_reason"] in ("salida", "tiempo", "stop") and np.isfinite(r["R_gross"]) and r["is_pullback"] == 1.0


def test_partial_fill_is_exactly_zero_r_when_stopped_after():
    close = np.r_[np.full(40, 100.0), 110.0, 113.0, 95.0, np.full(6, 95.0)]
    open_ = np.r_[np.full(40, 100.0), 100.0, 110.0, 112.0, np.full(6, 95.0)]
    high = np.r_[np.full(40, 100.5), 110.2, 114.0, 112.5, np.full(6, 95.5)]
    low = np.r_[np.full(40, 99.5), 99.9, 109.5, 90.0, np.full(6, 94.5)]
    d = add_indicators(_frame(close, open_, high, low), StrategyConfig(use_regime=False, partial_r=1.0, partial_frac=0.5))
    t = simulate_trades(d, StrategyConfig(use_regime=False, partial_r=1.0, partial_frac=0.5), "X", "trend")
    assert len(t) == 1 and t.iloc[0]["exit_reason"] == "stop+parcial"
    assert abs(t.iloc[0]["R_gross"]) < 1e-9                       # +1R en la mitad, -1R en la otra mitad


def test_dd_scale():
    rc = RiskConfig(max_drawdown_pct=20.0)
    assert dd_scale(0.0, rc) == 1.0 and dd_scale(0.07, rc) == 1.0 and dd_scale(0.16, rc) == 0.0
    vals = [dd_scale(x / 100, rc) for x in range(0, 21)]
    assert all(a >= b for a, b in zip(vals, vals[1:])) and 0.25 <= dd_scale(0.11, rc) <= 1.0
    assert dd_scale(0.5, RiskConfig(dd_throttle=False)) == 1.0


def test_systems_and_per_system_payoff():
    px = make_prices(14, 2500, seed=3)
    both = build_trades(px, CFG)
    only = build_trades(px, AppConfig(strategy=StrategyConfig(use_regime=False, pb_enabled=False)))
    assert set(both["system"]) == {"trend", "pullback"} and set(only["system"]) == {"trend"}
    m = MetaModel(CFG.model).fit(both)
    assert set(m.payoff_sys) == {"trend", "pullback"} and m.payoff_for("nada") == m.payoff


def test_edge_curve_and_calibration():
    px = make_prices(14, 2500, seed=3)
    sc = walk_forward_scores(build_trades(px, CFG), CFG.model)
    ec = edge_curve(sc)
    assert len(ec) >= 2 and (ec["trades"].diff().dropna() <= 0).all()   # más exigente -> menos trades
    cal = calibration(sc)
    assert 0 <= cal["brier"] <= 1 and len(cal["tabla"]) >= 2


def test_throttle_reduces_drawdown():
    px = make_prices(14, 2500, seed=5)
    tr = walk_forward_scores(build_trades(px, CFG), CFG.model)
    close = pd.DataFrame({k: v["Close"] for k, v in px.items()})
    dds = {}
    for flag in (True, False):
        rc = RiskConfig(max_drawdown_pct=10.0, dd_throttle=flag, allow_fractional=True, min_trade_usd=1, risk_flat_pct=2.0)
        eq, _ = run_portfolio(tr, close, 5000, rc, CFG.costs, False)
        dds[flag] = (eq / eq.cummax() - 1).min()
    assert dds[True] >= dds[False] - 1e-9, dds


def test_purged_folds_have_no_overlap():
    from quant_scanner.model import purged_folds
    rng = np.random.default_rng(0)
    st = np.sort(rng.integers(0, 3000, 400)).astype("datetime64[D]").astype("datetime64[ns]")
    en = st + (rng.integers(1, 40, 400)).astype("timedelta64[D]")
    seen = 0
    for tr, te in purged_folds(st, en, 4, 5):
        t0, t1 = st[te].min(), en[te].max()
        emb = np.timedelta64(5, "D")
        assert len(np.intersect1d(tr, te)) == 0
        assert ((en[tr] < t0 - emb) | (st[tr] > t1 + emb)).all()          # ningún trade de entrenamiento toca el bloque de prueba
        seen += len(te)
    assert seen == 400


def test_model_fit_modes():
    px = make_prices(14, 2500, seed=3)
    tr = build_trades(px, CFG)
    m = MetaModel(CFG.model).fit(tr)
    assert m.mode == "rf+platt" and 0 < m.platt.coef_[0][0] < 10 and np.isfinite(m.predict(tr.head(5))).all()
    m2 = MetaModel(CFG.model).fit(tr.drop(columns=["signal_date", "exit_date"]))      # sin fechas: cae al respaldo OOB
    assert m2.mode == "rf+platt"


def test_config_validation():
    for bad in (lambda: StrategyConfig(stop_atr=0), lambda: StrategyConfig(entry_window=0), lambda: StrategyConfig(partial_r=0.1),
                lambda: RiskConfig(risk_max_pct=50), lambda: RiskConfig(max_drawdown_pct=1), lambda: CostConfig(commission_pct=-1),
                lambda: RiskConfig(kelly_mult=2)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("debió rechazar una configuración inválida")
    AppConfig()   # los valores por defecto son válidos


def test_end_to_end_runs():
    px = make_prices(14, 2500, seed=9)
    out = run_backtest(px, CFG, capital=5000)
    assert "error" not in out, out.get("error")
    for k in ("base", "modelo", "spy"):
        assert out[k]["equity"].iloc[0] == 5000 or abs(out[k]["equity"].iloc[0] - 5000) < 1e-6
    print({k: {m: round(v, 3) for m, v in out[k]["metrics"].items() if m in ("CAGR", "MaxDD", "Sharpe")} for k in ("base", "modelo", "spy")})
    print({k: {m: round(v, 3) for m, v in out[k]["trade_metrics"].items() if m in ("Trades", "Tasa de acierto", "R medio (expectancy)")} for k in ("base", "modelo")})


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("OK  ", name)
            except Exception as e:
                import traceback; fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
