"""Casos límite e invariantes. Un sistema que maneja dinero tiene que ser aburrido ante datos raros."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from quant_scanner.config import AppConfig, StrategyConfig, RiskConfig
from quant_scanner.signals import add_indicators, simulate_trades
from quant_scanner.sizing import position_units
from quant_scanner.backtest import build_trades, walk_forward_scores, run_backtest, run_portfolio
from quant_scanner.metrics import equity_metrics, trade_metrics, monte_carlo_trades, probabilistic_sharpe
from quant_scanner.report import build_report
from quant_scanner.model import MetaModel
from quant_scanner.scanner import scan_today
from synth import make_prices

CFG = AppConfig(strategy=StrategyConfig(use_regime=False))


def test_short_and_degenerate_series():
    px = make_prices(2, 700, seed=1)
    d = px["A0"]
    for n in (1, 5, 30, 120):
        out = add_indicators(d.iloc[:n], CFG.strategy)
        assert len(out) == n
        assert simulate_trades(out, CFG.strategy, "X", "trend").empty or True      # no debe lanzar excepción
    flat = d.copy(); flat[["Open", "High", "Low", "Close"]] = 100.0
    t = simulate_trades(add_indicators(flat, CFG.strategy), CFG.strategy, "F")
    assert t.empty                                                                   # precio plano: ni señales ni división por cero
    assert build_trades({"A0": d.iloc[:100]}, CFG).empty                              # historia insuficiente
    assert walk_forward_scores(pd.DataFrame(columns=["signal_date"]), CFG.model).empty


def test_non_finite_never_becomes_an_order():
    rc = RiskConfig(min_trade_usd=0)
    for args in ((np.nan, 100, 95, 1.0), (1000, np.nan, 95, 1.0), (1000, 100, np.nan, 1.0), (1000, 100, 95, np.nan), (np.inf, 100, 95, 1.0)):
        assert position_units(*args, rc) == 0.0
    assert position_units(1000, 100, 95, 1.0, rc, cash_available=-50) == 0.0          # caja negativa: no compra
    assert position_units(1000, 100, 110, 1.0, rc) == 0.0                              # stop por encima del precio


def test_trade_invariants_many_seeds():
    for seed in range(6):
        px = make_prices(5, 1200, seed=seed)
        t = build_trades(px, CFG)
        if t.empty:
            continue
        assert (t["signal_date"] < t["entry_date"]).all() and (t["entry_date"] <= t["exit_date"]).all()
        assert (t["stop"] < t["entry_price"]).all() and (t["bars"] >= 1).all() and (t["exit_price"] > 0).all()
        assert np.isfinite(t["R_net"]).all() and t["R_net"].min() > -50                  # un gap enorme no puede dar R absurdos
        for tk, g in t.groupby(["ticker", "system"]):                                      # un mismo activo y sistema nunca se solapan
            g = g.sort_values("entry_date")
            assert (g["entry_date"].iloc[1:].to_numpy() > g["exit_date"].iloc[:-1].to_numpy()).all() or len(g) < 2


def test_portfolio_never_uses_leverage():
    px = make_prices(10, 2500, seed=12)
    tr = walk_forward_scores(build_trades(px, CFG), CFG.model)
    close = pd.DataFrame({k: v["Close"] for k, v in px.items()})
    for rc in (RiskConfig(), RiskConfig(max_positions=3, allow_fractional=True, min_trade_usd=1, risk_flat_pct=3.0, max_pos_pct=60.0)):
        for use_model in (False, True):
            eq, ex = run_portfolio(tr, close, 2000, rc, CFG.costs, use_model)
            assert eq.attrs["min_cash"] >= -1e-6, eq.attrs["min_cash"]                  # la caja nunca baja de cero
            assert (eq > 0).all()


def test_backtest_is_deterministic_and_small_universe_ok():
    px = make_prices(4, 2400, seed=21)
    a = run_backtest(px, CFG, 5000)
    b = run_backtest(px, CFG, 5000)
    if "error" in a:                                                                    # con poco universo puede no haber trades para entrenar
        assert "error" in b and isinstance(a["error"], str)
        return
    assert np.allclose(a["modelo"]["equity"].to_numpy(), b["modelo"]["equity"].to_numpy())


def test_metrics_with_tiny_inputs():
    assert equity_metrics(pd.Series([100.0, 101.0], index=pd.bdate_range("2024-01-01", periods=2))) == {}
    assert trade_metrics(pd.DataFrame()) == {"Trades": 0}
    assert monte_carlo_trades(np.array([1.0, -1.0]), 1.0) == {}
    assert math.isnan(probabilistic_sharpe(pd.Series([0.01] * 5)))
    assert math.isnan(probabilistic_sharpe(pd.Series([0.0] * 50)))                       # sin varianza


def test_report_with_empty_inputs_and_html_escaping():
    empty = pd.DataFrame(columns=["ticker", "entrada", "p_ganar", "cierre", "stop_sugerido", "riesgo_pct"])
    html = build_report(empty, {"regime_ok": True, "model_mode": "<script>x</script>", "n_train": 0}, None, 1000)
    assert "Sin señales" in html and "<script>x" not in html and "&lt;script&gt;" in html


def test_model_predict_without_fit_and_with_missing_features():
    m = MetaModel(CFG.model)
    out = m.predict(pd.DataFrame({"vol_20d": [0.1]}).reindex(columns=__import__("quant_scanner.config", fromlist=["FEATURE_COLS"]).FEATURE_COLS))
    assert len(out) == 1 and 0 < out[0] < 1
    px = make_prices(14, 2500, seed=3)
    tr = build_trades(px, CFG)
    m = MetaModel(CFG.model).fit(tr)
    feats = tr.head(3).copy()
    feats.loc[:, ["rs_20d", "vix_rank"]] = np.nan                                       # features faltantes -> se imputan, no se cae
    assert np.isfinite(m.predict(feats)).all()


def test_scan_with_no_valid_assets_returns_empty_frame_with_columns():
    from quant_scanner.scanner import SCAN_COLUMNS
    px = make_prices(3, 2400, seed=2)
    short = {"SPY": px["SPY"].iloc[-100:].copy(), "A0": px["A0"].iloc[-50:].copy()}
    scan, _, info = scan_today(short, CFG)
    assert scan.empty and list(scan.columns[:len(SCAN_COLUMNS)]) == SCAN_COLUMNS and "al_dia" in scan.columns
    assert "<script" not in build_report(scan, info, None, 1000)


def test_scan_skips_short_histories():
    px = make_prices(14, 2500, seed=3)
    px["NEW"] = px["A1"].iloc[-100:].copy()                                              # IPO reciente
    scan, _, _ = scan_today(px, CFG)
    assert "NEW" not in set(scan["ticker"])


if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
