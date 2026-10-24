"""CLI (run_daily.main) y alertas de Telegram con red simulada."""
import contextlib, io, logging, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
tmp = tempfile.mkdtemp()
os.environ["QS_DB"] = os.path.join(tmp, "cli.db")
os.chdir(tmp)
import pandas as pd
import requests
from quant_scanner import run_daily, store, alerts
from quant_scanner.events import all_events
from synth import make_prices


def call(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_daily.main(list(argv))
    return rc, buf.getvalue()


def test_cli_client_lifecycle_and_validation():
    assert call("add-client", "Ana", "5000", "--risk", "1.0", "--chat", "123456789")[0] == 0
    rc, out = call("list-clients"); assert "Ana" in out and "5000" in out
    assert call("fill", "1", "NVDA", "3", "--price", "100", "--stop", "90", "--system", "pullback")[0] == 0
    assert store.open_positions(1)[0]["system"] == "pullback"
    for bad in (("fill", "1", "NV DA", "3", "--price", "100", "--stop", "90"), ("fill", "1", "NVDA", "3", "--price", "100", "--stop", "110"),
                ("fill", "1", "NVDA", "-3", "--price", "100", "--stop", "90"), ("add-client", "Bob", "-10"), ("add-client", "Bob", "100", "--chat", "abc")):
        try: call(*bad); raise AssertionError(bad)
        except ValueError: pass
    assert call("close", "1", "--price", "105")[0] == 0 and store.open_positions(1) == []


def test_cli_events_and_refresh_override():
    rc, out = call("events", "--days", "60"); assert rc == 0 and ("FOMC" in out or "CPI" in out or "Sin eventos" in out)
    ics = "BEGIN:VCALENDAR\nBEGIN:VEVENT\nDTSTART;VALUE=DATE:20261111\nSUMMARY:Consumer Price Index for October 2026\nEND:VEVENT\nEND:VCALENDAR"
    orig = requests.get
    requests.get = lambda *a, **k: type("R", (), {"text": ics, "raise_for_status": lambda s: None})()
    try:
        assert call("refresh-events")[0] == 0
    finally:
        requests.get = orig
    ev = all_events()
    assert pd.Timestamp("2026-11-11") in set(ev["date"]) and pd.Timestamp("2026-11-10") not in set(ev[ev["name"].str.contains("CPI")]["date"])


def test_cli_check_data_and_backtest():
    import quant_scanner.providers as P
    ok = P.parse_tiingo([{"date": "2026-10-19T00:00:00.000Z", "adjOpen": 1, "adjHigh": 2, "adjLow": 1, "adjClose": 2, "adjVolume": 10}])
    P.FETCHERS = {"tiingo": lambda t, s: ok, "yfinance": lambda t, s: ok, "alpaca": lambda t, s: ok}
    assert call("check-data", "SPY")[0] == 0
    bad = ok.copy(); bad["Close"] = 3.0
    P.FETCHERS = {"tiingo": lambda t, s: ok, "yfinance": lambda t, s: bad, "alpaca": lambda t, s: ok}
    rc, out = call("check-data", "SPY"); assert rc == 1 and "CUARENTENA" in out
    px = make_prices(14, 2500, seed=9)
    run_daily.download_prices = lambda *a, **k: px
    rc, out = call("backtest", "--capital", "3000")
    assert rc in (0, 2) and ("CAGR" in out or "Pocos trades" in out or "No se generaron" in out)
    rc, out = call("backtest", "--capital", "3000", "--universe", "etf")
    assert rc in (0, 2)


def test_telegram_chunking_failure_and_no_token_leak():
    sent = []
    class R: ok = True
    orig = requests.post
    requests.post = lambda url, json=None, timeout=None: (sent.append((url, json)), R())[1]
    try:
        assert alerts.send_telegram("x" * 9000, token="TOKEN-SECRETO", chat_id="42") is True
        assert len(sent) == 3 and all(len(j["text"]) <= 3800 for _, j in sent) and sent[0][1]["chat_id"] == "42"
        assert alerts.send_telegram("hola", chat_override="77", token="T", chat_id="42") is True and sent[-1][1]["chat_id"] == "77"
        os.environ.pop("TELEGRAM_BOT_TOKEN", None); os.environ.pop("TELEGRAM_CHAT_ID", None)
        assert alerts.send_telegram("sin credenciales") is False                         # sin credenciales: no envía, no falla
        stream = io.StringIO(); h = logging.StreamHandler(stream); logging.getLogger("quant_scanner").addHandler(h)
        def boom(*a, **k): raise requests.ConnectionError("fallo con https://api.telegram.org/botTOKEN-SECRETO/sendMessage")
        requests.post = boom
        assert alerts.send_telegram("hola", token="TOKEN-SECRETO", chat_id="42") is False
        assert "TOKEN-SECRETO" not in stream.getvalue()                                    # el token jamás llega al log
    finally:
        requests.post = orig


if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
