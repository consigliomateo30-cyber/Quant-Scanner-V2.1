"""Prueba del flujo diario con datos sintéticos y SQLite temporal (sin internet ni Telegram)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["QS_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import pandas as pd
from quant_scanner import store
from quant_scanner.config import AppConfig
from quant_scanner.scanner import scan_today
from synth import make_prices

px = make_prices(14, 2600, seed=11)
# que el último dato sea "de hoy" para no marcarlo vencido
shift = pd.Timestamp.now().normalize() - px["SPY"].index[-1]
for k in px: px[k].index = px[k].index + shift
scan, model, info = scan_today(px, AppConfig())
assert {"ticker", "entrada", "p_ganar", "stop_sugerido"} <= set(scan.columns) and len(scan) == 28 and set(scan["sistema"]) == {"trend", "pullback"}
assert info["model_mode"] in ("rf+platt", "prior") and (scan["dias_de_atraso"] == 0).all()
store.add_client("Test", 5000, 1.0, 5, 25.0, True)
cl = store.list_clients()[0]
store.record_fill(cl["id"], "A1", 10, "2026-01-02", 100.0, 90.0)
assert len(store.open_positions(cl["id"])) == 1
assert store.claim_alert("k1", cl["id"], "entry", "A1", "x") is True
assert store.claim_alert("k1", cl["id"], "entry", "A1", "x") is False   # idempotente
print("OK flujo diario | señales hoy:", int(scan["entrada"].sum()), "| modo:", info["model_mode"], "| n_train:", info["n_train"])
