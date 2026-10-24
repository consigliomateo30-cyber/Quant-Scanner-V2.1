"""Flujo diario completo con noticias/eventos simulados: veto por hecho confirmado y reducción por evento macro."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["QS_DB"] = os.path.join(tempfile.mkdtemp(), "ctx.db")
from datetime import datetime, timedelta, timezone
import pandas as pd
from quant_scanner import run_daily, store
from quant_scanner.events import all_events
from quant_scanner.news import NewsItem
from quant_scanner.scanner import scan_today as real_scan
from synth import make_prices

px = make_prices(14, 2500, seed=9)
cut = {t: df.iloc[:2400].copy() for t, df in px.items()}
shift = pd.Timestamp.now().normalize() - cut["SPY"].index[-1]
for t in cut: cut[t].index = cut[t].index + shift
today = cut["SPY"].index[-1]

def fake_scan(prices, cfg):
    s, m, i = real_scan(prices, cfg); i["regime_ok"] = True
    row = s[(s.ticker == "A3") & (s.sistema == "pullback")].index[0]
    s.loc[row, ["entrada", "p_ganar", "payoff", "edge", "riesgo_pct", "stop_sugerido"]] = [True, 0.75, 0.6, 0.2, 1.0, s.loc[row, "cierre"] * 0.95]
    s["al_dia"] = True
    return s, m, i

run_daily.download_prices = lambda *a, **k: cut
run_daily.scan_today = fake_scan
run_daily.next_earnings_date = lambda tk: None
store.add_client("Prueba", 5000, 1.0, 5, 25.0, True)

# 1) hecho grave confirmado -> bloqueada
now = datetime.now(timezone.utc)
run_daily.fetch_news = lambda tk: ([NewsItem("A3 8-K", "sec.gov", "https://sec.gov/x", now - timedelta(hours=2), ("A3",), form="8-K", items=("4.02",))], [])
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf): run_daily.main(["scan", "--dry-run"])
out = buf.getvalue(); assert "SEÑAL BLOQUEADA A3" in out and "ENTRADA A3" not in out, out

# 2) sin noticias pero con evento de alto impacto mañana -> entra con riesgo reducido y aviso
nxt = (today + pd.offsets.BDay(1)).normalize()
extra = pd.DataFrame([(nxt, "CPI de prueba", "alto", "08:30", "test")], columns=["date", "name", "impact", "time_et", "source"])
run_daily.all_events = lambda: all_events(extra)
run_daily.fetch_news = lambda tk: ([], [])
store.claim_alert  # (las claves del día anterior son distintas: 'entry_veto' vs 'entry')
buf = io.StringIO()
with contextlib.redirect_stdout(buf): run_daily.main(["scan", "--dry-run"])
out = buf.getvalue(); assert "ENTRADA A3" in out and "Riesgo ≈ 0.50%" in out and "CPI de prueba" in out, out
assert "Próximos 7 días" in out
print("OK contexto en el flujo diario\n" + out)
