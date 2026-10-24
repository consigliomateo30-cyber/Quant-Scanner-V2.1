import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from quant_scanner.matrix import benjamini_hochberg, cluster_bootstrap, run_matrix, operability_map, render_report, trades_for
from quant_scanner.specs import default_specs, make_spec, signals_for
from synth import make_prices

def test_bh():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(q, [0.04, 0.05333333, 0.05333333, 0.2])
    assert len(benjamini_hochberg([])) == 0

def test_cluster_bootstrap():
    rng = np.random.default_rng(0)
    months = np.repeat(np.arange(60), 10)
    p1, ci1 = cluster_bootstrap(rng.normal(0.3, 1, 600), months)
    p0, ci0 = cluster_bootstrap(rng.normal(0.0, 1, 600), months)
    assert p1 < 0.02 and ci1[0] > 0 and p0 > 0.05 and ci0[0] < 0

def test_specs_causal_and_run():
    px = make_prices(8, 1500, seed=4)
    d = px["A1"]
    for tf in ("1d",):
        for sp in default_specs(tf):
            e, x = signals_for(d, sp, None, px["SPY"]["Close"])
            e2, x2 = signals_for(d.iloc[:1000], sp, None, px["SPY"]["Close"].iloc[:1000])
            assert (e[:1000] == e2).all() and (x[:1000] == x2).all(), sp.name       # no usa el futuro

def test_matrix_end_to_end():
    px = make_prices(10, 3200, seed=6, trend_edge=0.0015)
    groups = {"g1": ["A0", "A1", "A2", "A3"], "g2": ["A4", "A5", "A6"]}
    specs = {"1d": default_specs("1d")[:6], "1w": default_specs("1w")[:3]}
    df = run_matrix(px, groups, specs, dev_end="2020-12-31", n_boot=300)
    assert len(df) == 2 * 6 + 2 * 3 and set(df["nivel"]) <= {"A", "B", "C", "D"}
    assert {"q_dev", "break_even_lado_pct", "oos", "margen_costo", "aguanta CEDEAR liquido (supuesto)"} <= set(df.columns)
    assert (df.loc[df["nivel"].isin(["A", "B"]), "oos"] == "confirmado OOS").all()
    assert df["q_dev"].between(0, 1).all() and df.loc[df["n_dev"] < 30, "q_dev"].eq(1.0).all()
    rep = render_report(df, "2020-12-31")
    assert "Celdas probadas" in rep and "Advertencias" in rep
    operability_map(df)
    print("celdas:", len(df), df["nivel"].value_counts().to_dict(), "| n_dev máx:", int(df["n_dev"].max()))

def test_costs_reduce_edge():
    px = make_prices(4, 2000, seed=8, trend_edge=0.001)
    sp = make_spec("donchian", "1d", 2.5, 30, entry_n=20, exit_n=10)
    lo = trades_for(px["A0"], sp, None, None, "A0", 0.0005)
    hi = trades_for(px["A0"], sp, None, None, "A0", 0.01)
    assert len(lo) == len(hi) and lo["R_net"].mean() > hi["R_net"].mean() and np.allclose(lo["ret_gross"], hi["ret_gross"])

def _ar_prices(n_assets, n_days, phi, mu, seed):
    """Activos con autocorrelación de retornos `phi` (negativa = reversión de corto plazo)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n_days)
    out = {}
    for k in range(n_assets):
        r = np.zeros(n_days)
        for t in range(1, n_days):
            r[t] = mu + phi * r[t - 1] + rng.normal(0, 0.01)
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n_days)))
        out[f"X{k}"] = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)
    return out


def test_matrix_power_and_specificity():
    """El embudo debe DETECTAR un edge real (reversión) y NO inventar edge en ruido."""
    edge = _ar_prices(6, 4500, -0.45, 0.0004, 1)
    noise = _ar_prices(6, 4500, 0.0, 0.0003, 2)
    px = {**{f"E{k}": v for k, v in zip(range(6), edge.values())}, **{f"N{k}": v for k, v in zip(range(6), noise.values())}}
    px["SPY"] = list(noise.values())[0].copy()
    groups = {"con_edge": [f"E{k}" for k in range(6)], "ruido": [f"N{k}" for k in range(6)]}
    specs = {"1d": [s for s in default_specs("1d") if s.family in ("rsi_pullback", "ibs_reversion", "bb_reversion", "donchian")]}
    df = run_matrix(px, groups, specs, dev_end="2020-12-31", n_boot=500)
    good = df[(df["grupo"] == "con_edge") & df["familia"].isin(["rsi_pullback", "ibs_reversion", "bb_reversion"])]
    assert (good["nivel"].isin(["A", "B", "C"])).any(), df[["grupo", "spec", "n_dev", "R_dev", "q_dev", "nivel"]].round(3).to_string()
    assert (good["R_exc_dev"].max() > 0)
    nz = df[df["grupo"] == "ruido"]
    # bajo la nula, BH acota la tasa de falsos descubrimientos (~10%): se tolera a lo sumo 1 azar, y nunca nivel A/B
    assert nz["pasa_dev"].sum() <= 1 and not nz["nivel"].isin(["A", "B"]).any() and nz["q_dev"].median() > 0.5, nz[["spec", "n_dev", "R_dev", "R_exc_dev", "q_dev"]].round(3).to_string()
    assert (nz["R_dev"] > 0).any()          # el ruido con deriva SÍ "gana" en absoluto: por eso se testea el exceso sobre la deriva
    print("poder/especificidad OK:", df.groupby("grupo")["nivel"].value_counts().to_dict())


if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
