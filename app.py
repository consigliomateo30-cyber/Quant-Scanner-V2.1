"""Quant Scanner Pro v7 — interfaz Streamlit sobre el paquete quant_scanner."""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from quant_scanner import store
from quant_scanner.backtest import run_backtest
from quant_scanner.backtest import market_context
from quant_scanner.context import evaluate_context, fetch_news, next_earnings_date
from quant_scanner.data import freshness_report
from quant_scanner.events import all_events, upcoming
from quant_scanner.news import assess
from quant_scanner.config import DEFAULT_UNIVERSE, MARKET_PROXY, VIX_TICKER, AppConfig, CostConfig, RiskConfig, StrategyConfig
from quant_scanner.research import compare_variants, default_variants
from quant_scanner.data import download_prices, stale_tickers
from quant_scanner.metrics import deflated_sharpe, monte_carlo_trades, probabilistic_sharpe
from quant_scanner.report import build_report
from quant_scanner.scanner import scan_today
from quant_scanner.signals import add_indicators

st.set_page_config(page_title="Quant Scanner Pro", page_icon="⚡", layout="wide")
st.markdown("""<style>
.stApp{background:#08080a;color:#e4e4e7} h1,h2,h3{color:#f4f4f5!important}
[data-testid="stMetricValue"]{color:#d4af37!important}
.stButton>button{border:1px solid #d4af37;color:#d4af37;background:#12100a}
</style>""", unsafe_allow_html=True)


def password_gate():
    """Si definís APP_PASSWORD en los secrets, la app pide clave (comparación en tiempo constante y bloqueo
    tras 5 intentos fallidos por sesión). Imprescindible si hay datos de clientes."""
    import hmac
    import time
    try:
        pw = st.secrets.get("APP_PASSWORD", None)
    except Exception:           # no hay archivo de secrets
        pw = None
    if not pw:
        st.warning("⚠️ La app NO tiene contraseña. Definí APP_PASSWORD en los secrets antes de cargar datos de clientes.")
        return
    if st.session_state.get("auth_ok"):
        return
    fails = st.session_state.get("auth_fails", 0)
    if fails >= 5:
        st.error("Demasiados intentos. Cerrá y abrí la app para reintentar.")
        st.stop()
    entered = st.text_input("Contraseña", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state["auth_ok"] = True
            st.rerun()
        st.session_state["auth_fails"] = fails + 1
        time.sleep(1.5)
        st.error("Contraseña incorrecta.")
    st.stop()


def secrets_to_env():
    """En Streamlit Cloud los secretos viven en st.secrets: se copian a variables de entorno (sin pisar las existentes)
    para que alerts/providers/edgar/crypto las encuentren igual que en tu máquina."""
    try:
        for k in list(st.secrets.keys()):
            v = st.secrets[k]
            if isinstance(v, (str, int, float)) and k not in os.environ:
                os.environ[k] = str(v)
    except Exception:
        pass


secrets_to_env()
password_gate()

# ------------------------------ sidebar ------------------------------
st.sidebar.markdown("### ⚡ Parámetros")
universe = st.sidebar.multiselect("Universo", DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)
years = st.sidebar.slider("Años de historia", 5, 15, 10)
capital = st.sidebar.number_input("Capital (USD)", value=5000.0, step=500.0, min_value=100.0)
st.sidebar.markdown("**Costos por lado (%)**")
commission = st.sidebar.number_input("Comisión", value=0.15, step=0.05, min_value=0.0)
slippage = st.sidebar.number_input("Slippage / spread", value=0.10, step=0.05, min_value=0.0)
st.sidebar.markdown("**Riesgo**")
risk_max = st.sidebar.slider("Riesgo máx. por trade (%)", 0.25, 2.0, 1.0, 0.25)
risk_flat = st.sidebar.slider("Riesgo sin modelo (%)", 0.25, 2.0, 0.75, 0.25)
max_pos_pct = st.sidebar.slider("Máx. % por activo", 5, 50, 25, 5)
max_positions = st.sidebar.slider("Máx. posiciones", 1, 15, 8)
fractional = st.sidebar.checkbox("Mi broker permite fracciones", value=False)
max_dd = st.sidebar.slider("Drawdown máximo tolerado (%)", 10, 30, 20, 5)
st.sidebar.markdown("**Estrategia (swing 1-3 semanas)**")
pb_enabled = st.sidebar.checkbox("Sistema de pullbacks (mayor acierto)", value=True)
partial_opt = st.sidebar.selectbox("Parcial en tendencia (R)", ["Sin parcial", "1.0", "1.5", "2.0"], index=0)
stop_atr = st.sidebar.slider("Stop tendencia (ATR)", 1.0, 4.0, 2.0, 0.5)
use_regime = st.sidebar.checkbox("Filtro régimen SPY > SMA200", value=True)

cfg = AppConfig(
    strategy=StrategyConfig(stop_atr=stop_atr, use_regime=use_regime, pb_enabled=pb_enabled,
                            partial_r=None if partial_opt == "Sin parcial" else float(partial_opt)),
    costs=CostConfig(commission_pct=commission, slippage_pct=slippage),
    risk=RiskConfig(risk_max_pct=risk_max, risk_flat_pct=risk_flat, max_pos_pct=float(max_pos_pct),
                    max_positions=max_positions, allow_fractional=fractional,
                    max_drawdown_pct=float(max_dd)),
)
tickers = tuple(sorted(set(universe) | {MARKET_PROXY, VIX_TICKER}))


@st.cache_data(ttl=1800, show_spinner="Descargando precios…")
def load_prices(tks, yrs):
    return download_prices(list(tks), years=yrs)


st.title("QUANT SCANNER PRO")
st.caption("Señales de tendencia con meta-modelo calibrado · backtest walk-forward con costos · alertas por cliente")
tab_scan, tab_ctx, tab_res, tab_bt, tab_risk, tab_clients, tab_report = st.tabs(["⚡ Escáner", "🧭 Contexto", "🔬 Investigación", "🧪 Backtest", "🛡️ Riesgo", "👥 Clientes", "📄 Informe"])

# ------------------------------ escáner ------------------------------
with tab_scan:
    if st.button("Escanear mercado", type="primary", use_container_width=True):
        prices = load_prices(tickers, years)
        if MARKET_PROXY not in prices:
            st.error("No hay datos de SPY: no se pueden emitir señales.")
        else:
            scan, model, info = scan_today(prices, cfg)
            st.session_state["scan"] = (scan, info, stale_tickers(prices))
    if "scan" in st.session_state:
        scan, info, stale = st.session_state["scan"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Régimen SPY>SMA200", "ALCISTA" if info["regime_ok"] else "BAJISTA")
        c2.metric("Señales de entrada hoy", int(scan["entrada"].sum()))
        c3.metric("Modelo", info["model_mode"], f"n={info['n_train']}")
        c4.metric("Payoff medio (R)", " / ".join(f"{k}: {v:.2f}" for k, v in info["payoff_sys"].items()) or f"{info['payoff']:.2f}")
        if stale:
            st.warning(f"Datos vencidos (>5 días): {', '.join(stale)}. No operes señales de esos activos.")
        if not info["regime_ok"] and use_regime:
            st.info("El filtro de régimen está activo y el mercado está bajista: no se emiten compras.")
        show = scan.copy()
        show["Acción"] = np.where(show["entrada"] & (show["riesgo_pct"] > 0), "🟢 ENTRADA",
                          np.where(show["entrada"], "🟡 señal sin edge", np.where(show["en_zona"], "🔵 en zona", "⚪ sin señal")))
        cols = ["ticker", "sistema", "Acción", "cierre", "p_ganar", "edge", "riesgo_pct", "stop_sugerido", "al_dia"]
        st.dataframe(show.sort_values(["entrada", "edge"], ascending=False)[cols].style.format(
            {"cierre": "{:.2f}", "p_ganar": "{:.1%}", "edge": "{:+.2f}", "riesgo_pct": "{:.2f}", "stop_sugerido": "{:.2f}"}),
            use_container_width=True)
        sel = st.selectbox("Ver gráfico", sorted(set(scan["ticker"])))
        prices = load_prices(tickers, years)
        d = add_indicators(prices[sel], cfg.strategy, *market_context(prices, cfg)).tail(180)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Cierre", line=dict(color="#e4e4e7")))
        fig.add_trace(go.Scatter(x=d.index, y=d["don_high"], name="Máx. 10d", line=dict(color="#d4af37", dash="dash")))
        fig.add_trace(go.Scatter(x=d.index, y=d["don_low"], name="Mín. 5d", line=dict(color="#ef4444", dash="dash")))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10), title=sel)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Las señales se confirman al cierre y se ejecutan en la apertura siguiente. El stop es una referencia: el stop real se calcula desde tu precio de compra.")

# ------------------------------ contexto ------------------------------
with tab_ctx:
    st.markdown("**Próximos eventos de mercado (30 días)**")
    ev = upcoming(all_events(), pd.Timestamp.now(), 30)
    st.dataframe(ev[["date", "en_dias", "impact", "time_et", "name"]] if len(ev) else pd.DataFrame({"info": ["Sin eventos"]}), use_container_width=True)
    st.caption("Fechas verificadas el 2026-10-20 contra varias fuentes. Las futuras pueden moverse: confirmá en federalreserve.gov y bls.gov, "
               "o corré `python -m quant_scanner.run_daily refresh-events` para bajar el calendario oficial del BLS.")
    st.markdown("**Frescura de datos** (un activo atrasado vs SPY no emite entradas)")
    if st.button("Revisar frescura"):
        fr = freshness_report(load_prices(tickers, years))
        st.dataframe(fr, use_container_width=True)
        if not fr["al_dia"].all():
            st.warning("Hay activos desactualizados: " + ", ".join(fr.loc[~fr["al_dia"], "ticker"]))
    st.caption("Diseño pensado para swing: el escaneo debe correr DESPUÉS del cierre de EE.UU. (≥ 16:30 ET). La barra del día en curso se descarta sola.")
    st.markdown("**Noticias con credibilidad**")
    tk_n = st.selectbox("Activo", [u for u in universe if u != MARKET_PROXY] or universe, key="news_tk")
    if st.button("Buscar noticias y 8-K"):
        items, errs = fetch_news(tk_n, include_sec=bool(os.environ.get("SEC_USER_AGENT")))
        if errs:
            st.warning("No pude consultar todo: " + "; ".join(errs) + ". Un error NO significa 'sin noticias'.")
        sc = assess(items)
        if sc:
            st.dataframe(pd.DataFrame([{"estado": x.label, "puntaje": x.score, "nivel_fuente": x.best_tier, "fuentes_indep.": x.n_sources,
                                        "impacto": x.impact, "horas": x.age_hours, "titular": x.headline[:110], "motivo": x.reason} for x in sc]),
                         use_container_width=True)
        ctx = evaluate_context(items, pd.Timestamp.now(), all_events(), cfg, cfg.strategy.max_hold, next_earnings_date(tk_n), errs)
        (st.error if ctx["decision"] == "VETO" else st.warning if ctx["decision"] == "AVISO" else st.success)(f"Decisión del filtro: {ctx['decision']} (riesgo x{ctx['risk_mult']:g})")
        for n in ctx["notes"]:
            st.write(n)
    st.info("Regla: una noticia nunca genera una compra. Solo veta un hecho grave CONFIRMADO (8-K de la SEC o prensa de nivel 1-2 corroborada) "
            "o avisa. Los rumores sin confirmar se muestran pero no pesan. Para sumar un dato alternativo al modelo debe pasar `validate_signal`.")

# ------------------------------ investigación ------------------------------
with tab_res:
    st.caption("Matriz estrategia x clase de activo/industria x temporalidad, con corrección por comparaciones múltiples, "
               "período reservado (fuera de muestra) y margen de costo. Se genera con `python -m quant_scanner.run_daily matrix`.")
    up = st.file_uploader("Cargá matriz.csv", type="csv")
    path = "reports/matriz.csv"
    mat = pd.read_csv(up) if up else (pd.read_csv(path) if os.path.exists(path) else None)
    if mat is None:
        st.info("Todavía no hay resultados. Corré el comando de arriba en tu máquina y cargá el archivo.")
    else:
        lv = mat["nivel"].value_counts().to_dict()
        c1, c2, c3, c4 = st.columns(4)
        for col, k in zip((c1, c2, c3, c4), "ABCD"):
            col.metric(f"Nivel {k}", lv.get(k, 0))
        sel = st.multiselect("Niveles a mostrar", list("ABCD"), default=["A", "B", "C"])
        cols = ["nivel", "grupo", "tf", "spec", "n_dev", "acierto_dev", "R_dev", "R_exc_dev", "PF_dev", "q_dev", "break_even_lado_pct", "oos", "n_hold", "R_exc_hold"]
        st.dataframe(mat[mat["nivel"].isin(sel)][[c for c in cols if c in mat.columns]].style.format(precision=3), use_container_width=True)
        st.caption("Operá solo celdas A/B. El break-even es el costo por lado al que la celda deja de ganar: compará con TU costo real (CEDEAR: spread + comisión + CCL).")

# ------------------------------ backtest ------------------------------
with tab_bt:
    st.caption("Walk-forward: el modelo se re-entrena cada ~3 meses SOLO con trades ya cerrados. Incluye costos y topes de cartera.")
    n_trials = st.number_input("¿Cuántas variantes de parámetros probaste hasta llegar a esta? (para el Deflated Sharpe)", 1, 1000, 1)
    if st.button("Correr backtest", type="primary", use_container_width=True):
        with st.spinner("Simulando…"):
            st.session_state["bt"] = run_backtest(load_prices(tickers, years), cfg, capital)
    bt = st.session_state.get("bt")
    if bt:
        if "error" in bt:
            st.error(bt["error"])
        else:
            st.caption(f"Ventana evaluada desde {bt['eval_start'].date()} (antes de eso el modelo no tenía trades suficientes).")
            rows = {}
            for k, lab in (("base", "Sin meta-modelo"), ("modelo", "Con meta-modelo"), ("spy", "SPY buy&hold")):
                if k in bt:
                    rows[lab] = {**bt[k]["metrics"], **bt[k].get("trade_metrics", {})}
            st.dataframe(pd.DataFrame(rows).T.style.format(precision=3), use_container_width=True)
            fig = go.Figure()
            for k, lab, colr in (("base", "Sin meta-modelo", "#a1a1aa"), ("modelo", "Con meta-modelo", "#d4af37"), ("spy", "SPY", "#3b82f6")):
                if k in bt:
                    fig.add_trace(go.Scatter(x=bt[k]["equity"].index, y=bt[k]["equity"], name=lab, line=dict(color=colr)))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10), title="Curva de capital (USD)")
            st.plotly_chart(fig, use_container_width=True)
            if not bt["modelo"]["by_system"].empty:
                st.markdown("**Resultado por sistema (con meta-modelo)**")
                st.dataframe(bt["modelo"]["by_system"].style.format(precision=3), use_container_width=True)
            if len(bt.get("edge_curve", [])):
                st.markdown("**Curva de edge (fuera de muestra):** qué pasó realmente con los trades según el edge estimado por el modelo. Sirve para elegir el umbral con evidencia.")
                st.dataframe(bt["edge_curve"].style.format({"acierto": "{:.1%}", "R_medio": "{:+.3f}", "profit_factor": "{:.2f}"}), use_container_width=True)
            cal = bt.get("calibration")
            if cal:
                st.markdown(f"**Calibración:** Brier {cal['brier']:.4f} vs {cal['brier_base']:.4f} de un modelo sin información (menor es mejor).")
                st.dataframe(cal["tabla"].style.format({"p_media": "{:.1%}", "acierto_real": "{:.1%}", "n": "{:.0f}"}), use_container_width=True)
                if cal["brier"] >= cal["brier_base"]:
                    st.warning("El modelo no supera al azar en Brier: sus probabilidades no aportan información. No lo uses para filtrar.")
            r = bt["modelo"]["equity"].pct_change().dropna()
            st.metric("PSR (prob. de Sharpe > 0)", f"{probabilistic_sharpe(r):.1%}")
            st.metric(f"Deflated Sharpe ({n_trials} variantes)", f"{deflated_sharpe(r, int(n_trials)):.1%}")
            if bt["modelo"]["metrics"].get("Sharpe", 0) <= bt["base"]["metrics"].get("Sharpe", 0):
                st.warning("El meta-modelo NO mejora al sistema base en este backtest. En ese caso, es más sano operar sin él.")
            ex = bt["modelo"]["executed"]
            if not ex.empty:
                st.download_button("Descargar trades (CSV)", ex.to_csv(index=False), "trades_backtest.csv", "text/csv")

    st.markdown("---")
    st.markdown("**Comparar variantes** (cada variante cuenta como una prueba y se descuenta en el Deflated Sharpe)")
    if st.button("Comparar variantes (puede tardar varios minutos)"):
        with st.spinner("Corriendo variantes…"):
            st.session_state["variants"] = compare_variants(load_prices(tickers, years), default_variants(cfg), capital)
    if "variants" in st.session_state:
        st.dataframe(st.session_state["variants"].style.format(precision=3), use_container_width=True)
        st.caption("Elegí por R medio, profit factor y drawdown (no por acierto), y desconfiá de una variante que gana por mucho a las demás.")

# ------------------------------ riesgo ------------------------------
with tab_risk:
    bt = st.session_state.get("bt")
    if not bt or "error" in bt:
        st.info("Primero corré el backtest.")
    else:
        which = st.radio("Trades a usar", ["modelo", "base"], horizontal=True)
        rp = st.slider("Riesgo por trade (%)", 0.25, 3.0, float(risk_max), 0.25)
        nt = st.slider("Trades a simular", 30, 300, 100, 10)
        mc = monte_carlo_trades(bt[which]["executed"]["R_net"].to_numpy(), rp, nt)
        if not mc:
            st.warning("Muy pocos trades para simular.")
        else:
            c = st.columns(4)
            c[0].metric("P(pérdida al final)", f"{mc['P(pérdida al final)']:.0%}")
            c[1].metric("Retorno mediano", f"{mc['Retorno mediano']:.0%}")
            c[2].metric("Drawdown peor 5%", f"{mc['Drawdown P95 (peor 5%)']:.0%}")
            c[3].metric("P(DD > 20%)", f"{mc['P(drawdown > 20%)']:.0%}")
            st.caption("Bootstrap sobre los R reales de los trades: mide cuánto puede doler una mala racha con tu riesgo por trade.")

# ------------------------------ clientes ------------------------------
with tab_clients:
    st.caption("Cada cliente tiene su capital, límites de riesgo y posiciones. Los datos se guardan en SQLite (quant_scanner.db).")
    with st.form("nuevo_cliente"):
        n1, n2, n3 = st.columns(3)
        name = n1.text_input("Nombre")
        cap = n2.number_input("Capital (USD)", 0.0, step=500.0, value=1000.0)
        chat = n3.text_input("Chat ID de Telegram (opcional)")
        if st.form_submit_button("Agregar cliente") and name:
            try:
                store.add_client(name, cap, risk_max, max_positions, float(max_pos_pct), fractional, chat or None)
                st.success("Cliente agregado.")
            except Exception as ex_:
                st.error(f"No se pudo agregar: {ex_}")
    clients = store.list_clients(False)
    if clients:
        st.dataframe(pd.DataFrame(clients), use_container_width=True)
        cid = st.selectbox("Cliente", [c["id"] for c in clients], format_func=lambda i: next(c["name"] for c in clients if c["id"] == i))
        pos = store.open_positions(cid)
        st.markdown("**Posiciones abiertas**")
        st.dataframe(pd.DataFrame(pos) if pos else pd.DataFrame({"info": ["Sin posiciones"]}), use_container_width=True)
        with st.form("fill"):
            f1, f2, f3, f4 = st.columns(4)
            tk, un = f1.text_input("Ticker"), f2.number_input("Unidades", 0.0, step=1.0)
            px_, sp_ = f3.number_input("Precio de compra", 0.0), f4.number_input("Stop", 0.0)
            if st.form_submit_button("Registrar compra") and tk and un > 0 and px_ > sp_ > 0:
                store.record_fill(cid, tk, un, str(datetime.now().date()), px_, sp_)
                st.success("Compra registrada. Recargá la pestaña para verla.")
    st.info("Las alertas automáticas salen del job diario (`python -m quant_scanner.run_daily scan`), no de esta pantalla.")

# ------------------------------ informe ------------------------------
with tab_report:
    if "scan" not in st.session_state:
        st.info("Corré el escáner primero.")
    else:
        scan, info, _ = st.session_state["scan"]
        doc = build_report(scan, info, st.session_state.get("bt"), capital)
        components.html(doc, height=600, scrolling=True)
        st.download_button("Descargar informe (HTML)", doc, f"informe_{datetime.now():%Y%m%d}.html", "text/html")
