"""Piezas de despliegue: scripts de Actions, YAMLs, integración de verificación de datos con proveedores simulados."""
import contextlib, io, os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp()
os.environ["QS_DB"] = os.path.join(tmp, "dep.db")
import yaml
import pandas as pd
from quant_scanner import run_daily, store
from quant_scanner.scanner import scan_today as real_scan
from synth import make_prices


def test_workflows_are_valid_and_safe():
    wdir = os.path.join(ROOT, ".github", "workflows")
    names = sorted(os.listdir(wdir))
    assert names == ["ci.yml", "daily-scan.yml", "manage.yml", "research.yml"]
    for n in names:
        d = yaml.safe_load(open(os.path.join(wdir, n), encoding="utf-8"))
        assert d["permissions"] == {"contents": "read"}                                    # mínimo privilegio
        raw = open(os.path.join(wdir, n), encoding="utf-8").read()
        assert "secrets." not in raw.split("run:")[0].split("steps:")[0]
        for j in d["jobs"].values():
            for s in j["steps"]:
                assert "${{ inputs" not in s.get("run", "") and "${{ github.event" not in s.get("run", "")   # nada de inyección en scripts
    sched = yaml.safe_load(open(os.path.join(wdir, "daily-scan.yml")))[True]["schedule"][0]["cron"]
    assert sched == "30 21 * * 1-5"


def test_yahoo_down_falls_back_to_tiingo():
    import types
    import quant_scanner.data as qd
    import quant_scanner.providers as P
    os.environ["QS_CACHE_DIR"] = tempfile.mkdtemp()
    qd.CACHE_DIR = __import__("pathlib").Path(os.environ["QS_CACHE_DIR"])
    ok = P.parse_tiingo([{"date": f"2026-10-{d:02d}T00:00:00.000Z", "adjOpen": 10.0, "adjHigh": 11.0, "adjLow": 9.0, "adjClose": 10.5, "adjVolume": 100} for d in (14, 15, 16)])
    sys.modules["yfinance"] = types.SimpleNamespace(download=lambda *a, **k: (_ for _ in ()).throw(ConnectionError("429")))
    calls = []
    P.fetch_tiingo = lambda tk, start: (calls.append(tk), ok)[1]
    os.environ["TIINGO_TOKEN"] = "x"
    out = qd.download_prices(["SPY", "BTC-USD"], years=1, use_cache=False)
    assert set(out) == {"SPY"} and calls == ["SPY"]                                       # cripto no existe en Tiingo: no se pide
    os.environ.pop("TIINGO_TOKEN")
    assert qd.download_prices(["QQQ"], years=1, use_cache=False) == {}                    # sin token y sin Yahoo: vacío, sin excepción


def test_gh_manage_rejects_unknown_commands_and_parses_quotes():
    env = {**os.environ, "QS_DB": os.path.join(tmp, "gh.db"), "PYTHONPATH": ROOT}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gh_manage.py")], env={**env, "QS_CMD": "scan; rm -rf /"}, capture_output=True, text=True)
    assert r.returncode != 0 and "no permitido" in (r.stderr + r.stdout)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gh_manage.py")], env={**env, "QS_CMD": "add-client", "QS_ARGS": '"Ana López" 5000 --risk 1.0 --chat 123456789'}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gh_manage.py")], env={**env, "QS_CMD": "list-clients"}, capture_output=True, text=True)
    assert "Ana López" in r.stdout and "123456789" not in r.stdout                           # el chat id no se imprime en los logs


def test_signal_blocked_when_sources_conflict():
    import quant_scanner.providers as P
    px = make_prices(14, 2500, seed=9)
    cut = {t: df.iloc[:2400].copy() for t, df in px.items()}
    shift = pd.Timestamp.now().normalize() - cut["SPY"].index[-1]
    for t in cut: cut[t].index = cut[t].index + shift

    def fake_scan(prices, cfg):
        s, m, i = real_scan(prices, cfg); i["regime_ok"] = True
        row = s[(s.ticker == "A3") & (s.sistema == "pullback")].index[0]
        s.loc[row, ["entrada", "p_ganar", "payoff", "edge", "riesgo_pct", "stop_sugerido"]] = [True, 0.75, 0.6, 0.2, 1.0, s.loc[row, "cierre"] * 0.95]
        s["al_dia"] = True
        return s, m, i
    run_daily.download_prices, run_daily.scan_today = (lambda *a, **k: cut), fake_scan
    run_daily.fetch_news, run_daily.next_earnings_date = (lambda tk: ([], [])), (lambda tk: None)
    store.add_client("Prueba", 5000, 1.0, 5, 25.0, True)
    ok = P.parse_tiingo([{"date": "2026-10-19T00:00:00.000Z", "adjOpen": 1, "adjHigh": 2, "adjLow": 1, "adjClose": 2, "adjVolume": 10}])
    bad = ok.copy(); bad["Close"] = 2.5
    os.environ["TIINGO_TOKEN"] = "x"
    P.FETCHERS = {"tiingo": lambda t, s: ok, "yfinance": lambda t, s: bad, "alpaca": lambda t, s: ok}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_daily.main(["scan", "--dry-run"])
    out = buf.getvalue()
    assert "SEÑAL BLOQUEADA A3" in out and "Datos en conflicto" in out and "ENTRADA A3" not in out, out
    os.environ.pop("TIINGO_TOKEN")


if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
