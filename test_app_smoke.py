"""Ejecuta app.py de punta a punta con un Streamlit simulado, pulsando cada botón de cada pestaña."""
import os, sys, runpy, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["QS_DB"] = os.path.join(tempfile.mkdtemp(), "app.db")
os.environ["QS_CACHE_DIR"] = tempfile.mkdtemp()
from datetime import datetime, timedelta, timezone
import pandas as pd
import stub_streamlit
from synth import make_prices

st = stub_streamlit.install()
import quant_scanner.data as qdata
import quant_scanner.context as qctx
import quant_scanner.research as qres
from quant_scanner.news import NewsItem

PX = make_prices(14, 2500, seed=9)
_shift = pd.Timestamp.now().normalize() - PX["SPY"].index[-1]
for _k in PX: PX[_k].index = PX[_k].index + _shift
qdata.download_prices = lambda *a, **k: PX
now = datetime.now(timezone.utc)
qctx.fetch_news = lambda tk, include_sec=True: ([NewsItem("A0 8-K", "sec.gov", "https://sec.gov/a", now - timedelta(hours=2), ("A0",), form="8-K", items=("2.02",))], [])
qctx.next_earnings_date = lambda tk: None
qres.compare_variants = lambda prices, variants, capital=5000.0: pd.DataFrame({"CAGR": [0.1]}, index=["variante"])
APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def run(press=(), text=None):
    st.press, st.text = set(press), text or {}
    try:
        runpy.run_path(APP, run_name="__main__")
    except stub_streamlit.Stop:
        pass


def test_app_all_tabs():
    run()                                                                       # carga inicial sin botones
    run(press={"Escanear mercado"})                                             # escáner (usa el meta-modelo y los datos)
    assert "scan" in st.session_state and len(st.session_state["scan"][0]) > 0
    run()                                                                       # re-render con el escaneo guardado (tabla + gráfico + informe)
    run(press={"Revisar frescura", "Buscar noticias y 8-K"})                    # pestaña Contexto
    run(press={"Correr backtest"})                                              # backtest completo
    bt = st.session_state.get("bt")
    assert bt and "error" not in bt, bt
    run()                                                                       # re-render con backtest (curvas, calibración, riesgo Monte Carlo)
    run(press={"Comparar variantes (puede tardar varios minutos)"})
    assert "variants" in st.session_state
    run(press={"Agregar cliente"}, text={"Nombre": "Cliente Smoke"})            # clientes
    from quant_scanner import store
    assert any(c["name"] == "Cliente Smoke" for c in store.list_clients())
    run()
    run(press={"Registrar compra"}, text={"Ticker": "A1"})                      # ticker válido pero sin unidades: no debe romper
    assert st.errors == [], st.errors                                         # la app nunca mostró un error al usuario


def test_app_password_gate():
    st.secrets = {"APP_PASSWORD": "clave-test"}
    st.session_state.clear()
    stopped = False
    try:
        runpy.run_path(APP, run_name="__main__")
    except stub_streamlit.Stop:
        stopped = True
    assert stopped and not st.session_state.get("auth_ok")                      # sin clave correcta, la app se detiene antes de mostrar nada
    st.secrets = {}


if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
