"""Job diario sin interfaz: escanea, arma alertas por cliente y las manda (sin duplicar).

Uso:
  python -m quant_scanner.run_daily scan [--dry-run] [--years 10]
  python -m quant_scanner.run_daily add-client "Mateo" 5000 --risk 1.0 --fractional
  python -m quant_scanner.run_daily fill 1 NVDA 3 --price 180.5 --stop 170.2 --system trend
  python -m quant_scanner.run_daily close 7 --price 195 --date 2026-11-03
  python -m quant_scanner.run_daily backtest --capital 5000
Programalo una vez por día después del cierre (cron, GitHub Actions o similar)."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from datetime import date

import pandas as pd

from . import store
from .alerts import send_telegram
from .config import DEFAULT_UNIVERSE, AppConfig, MARKET_PROXY, RiskConfig, VIX_TICKER
from .context import evaluate_context, fetch_news, next_earnings_date
from .data import download_prices
from .events import BLS_ICS_URL, OVERRIDE_FILE, all_events, parse_ics, upcoming
from .scanner import scan_today
from .sizing import position_units


def _client_risk(c: dict, base: RiskConfig) -> RiskConfig:
    return dataclasses.replace(base, risk_max_pct=c["risk_max_pct"], max_positions=c["max_positions"],
                               max_pos_pct=c["max_pos_pct"], allow_fractional=bool(c["allow_fractional"]))


def cmd_scan(args) -> int:
    cfg = AppConfig()
    tickers = sorted(set(DEFAULT_UNIVERSE + [MARKET_PROXY, VIX_TICKER]))
    prices = download_prices(tickers, years=args.years)
    if MARKET_PROXY not in prices:
        print("ERROR: no hay datos de SPY; no se emiten señales.")
        return 2
    scan, _model, info = scan_today(prices, cfg)
    if scan.empty:
        print("ERROR: ningún activo tiene historia suficiente (>= 260 ruedas); no se emiten señales.")
        return 2
    day = str(scan["fecha_dato"].max().date())
    msgs: list[tuple[str, str, str, str, str | None]] = []   # (key, kind, ticker, text, chat)
    events = all_events()
    ctx_cache: dict[tuple[str, str], dict] = {}

    def context_for(tk: str, system: str) -> dict:
        key = (tk, system)
        if key not in ctx_cache:
            items, errs = ([], []) if args.no_news else fetch_news(tk)
            earn = None if args.no_news else next_earnings_date(tk)
            ctx = evaluate_context(items, pd.Timestamp(day), events, cfg, cfg.strategy.hold_for(system), earn, errs)
            if not args.no_datacheck and any(os.environ.get(k) for k in ("TIINGO_TOKEN", "ALPACA_KEY_ID")):
                from datetime import date, timedelta
                from .providers import fetch_validated
                _, rep = fetch_validated(tk, (date.today() - timedelta(days=30)).isoformat())
                if rep["cuarentena"]:
                    ctx["decision"] = "VETO"
                    ctx["notes"].append("⛔ Datos en conflicto entre fuentes (" + "; ".join(rep["detalle"] or [rep["estado"]]) + "): no operar hasta verificar")
                elif rep["estado"] == "advertencia":
                    ctx["notes"].append("⚠️ Las fuentes de datos no coinciden del todo: " + "; ".join(rep["detalle"]))
            ctx_cache[key] = ctx
        return ctx_cache[key]
    clients = store.list_clients()
    if not clients:
        print("No hay clientes cargados (usá add-client). Se muestra solo el escaneo.")
    for c in clients:
        rc = _client_risk(c, cfg.risk)
        positions = store.open_positions(c["id"])
        held = {p["ticker"]: p for p in positions}
        # 1) entradas nuevas (un solo sistema por activo: el de mayor edge estimado)
        if info["regime_ok"] and len(held) < rc.max_positions:
            cand = scan[scan["entrada"] & (scan["riesgo_pct"] > 0) & (~scan["ticker"].isin(held)) & scan["al_dia"]]
            cand = cand.sort_values("edge", ascending=False).drop_duplicates("ticker")
            for _, r in cand.iterrows():
                ctx = context_for(r["ticker"], r["sistema"])
                if ctx["decision"] == "VETO":
                    txt = (f"⛔ SEÑAL BLOQUEADA {r['ticker']} [{r['sistema']}]\n" + "\n".join(ctx["notes"]))
                    msgs.append((f"{day}:{c['id']}:{r['ticker']}:entry_veto", "entry_veto", r["ticker"], f"[{c['name']}] {txt}", c["telegram_chat_id"]))
                    continue
                risk_eff = r["riesgo_pct"] * ctx["risk_mult"]
                u = position_units(c["capital"], r["cierre"], r["stop_sugerido"], risk_eff, rc, cost_per_side=cfg.costs.per_side)
                u = round(u, 2) if rc.allow_fractional else u
                hold = cfg.strategy.hold_for(r["sistema"])
                if u <= 0:
                    txt = (f"⚪ {r['ticker']} ({r['sistema']}): señal de entrada pero tu capital no alcanza para una posición "
                           f"válida con el riesgo configurado.")
                    kind = "entry_skip"
                else:
                    exit_rule = ("cierre bajo el mínimo de 5 días" if r["sistema"] == "trend" else "cierre sobre la media de 5 días")
                    txt = (f"🟢 ENTRADA {r['ticker']} [{r['sistema']}] — comprar mañana a la apertura\n"
                           f"Unidades: {u:g} (≈ USD {u * r['cierre']:,.0f}) | Riesgo ≈ {risk_eff:.2f}% del capital\n"
                           f"Stop inicial ≈ {r['stop_sugerido']:.2f} | Cierre ref. {r['cierre']:.2f}\n"
                           f"Salida: {exit_rule}, o máx. {hold} ruedas.\n"
                           f"Prob. estimada de ganar {r['p_ganar']*100:.0f}% | edge estimado {r['edge']:+.2f}R "
                           f"(modelo: {info['model_mode']}, n={info['n_train']})")
                    if ctx["notes"]:
                        txt += "\n" + "\n".join(ctx["notes"])
                    kind = "entry"
                msgs.append((f"{day}:{c['id']}:{r['ticker']}:{kind}", kind, r["ticker"], f"[{c['name']}] {txt}", c["telegram_chat_id"]))
        # 2) salidas / stops / tiempo de posiciones abiertas
        for tk, pos in held.items():
            system = pos.get("system") or "trend"
            row = scan[(scan["ticker"] == tk) & (scan["sistema"] == system)]
            if row.empty or tk not in info["frames"]:
                continue
            fr = info["frames"][tk]
            last = fr.iloc[-1]
            bars_held = int((fr.index >= pd.Timestamp(pos["entry_date"])).sum())
            if last["Low"] <= pos["stop"]:
                txt = f"🔴 STOP tocado en {tk} (mín. del día {last['Low']:.2f} ≤ stop {pos['stop']:.2f}). Revisá la orden y vendé."
                kind = "stop"
            elif bool(row.iloc[0]["salida"]):
                txt = f"🟠 SALIDA {tk} [{system}]: se dio la señal de salida. Vender mañana a la apertura."
                kind = "exit"
            elif bars_held >= cfg.strategy.hold_for(system):
                txt = f"🟠 SALIDA POR TIEMPO {tk} [{system}]: {bars_held} ruedas en posición. Vender mañana a la apertura."
                kind = "time"
            else:
                continue
            msgs.append((f"{day}:{c['id']}:{tk}:{kind}", kind, tk, f"[{c['name']}] {txt}", c["telegram_chat_id"]))

    sent = 0
    for key, kind, tk, text, chat in msgs:
        cid = int(key.split(":")[1])
        if not store.claim_alert(key, cid, kind, tk, text):
            continue
        print(text, "\n")
        if not args.dry_run and send_telegram(text, chat_override=chat):
            store.mark_sent(key); sent += 1
    header = (f"Quant Scanner {day} | régimen SPY>SMA200: {'SÍ' if info['regime_ok'] else 'NO (sin compras)'} | "
              f"señales de entrada hoy: {int(scan['entrada'].sum())} | alertas enviadas: {sent}")
    behind = scan.loc[~scan["al_dia"], "ticker"].unique().tolist()
    if behind:
        header += f"\n⚠️ Datos desactualizados (no se emiten entradas): {', '.join(behind)}"
    up = upcoming(events, pd.Timestamp(day), 7)
    up = up[up["impact"] == "alto"]
    if len(up):
        header += "\n📅 Próximos 7 días: " + "; ".join(f"{r.date:%d/%m} {r.name}" for r in up.itertuples())
    print(header)
    if not args.dry_run:
        send_telegram(header)
    return 0


def cmd_backtest(args) -> int:
    from .backtest import run_backtest
    if args.universe == "etf":      # la prueba más limpia: ETFs (sin el sesgo de supervivencia de elegir acciones ganadoras)
        from .universe import GROUPS
        base = sorted({t for g in ("indices_us", "sectores_us", "internacional", "bonos", "materias_primas") for t in GROUPS[g]})
    else:
        base = DEFAULT_UNIVERSE
    prices = download_prices(sorted(set(base + [MARKET_PROXY, VIX_TICKER])), years=args.years)
    out = run_backtest(prices, AppConfig(), capital=args.capital)
    if "error" in out:
        print(out["error"]); return 2
    print(f"Ventana evaluada desde {out['eval_start'].date()}")
    for k in ("base", "modelo", "spy"):
        print(f"\n== {k.upper()} ==")
        for m, v in out[k]["metrics"].items():
            print(f"  {m}: {v:.3f}")
        for m, v in out[k].get("trade_metrics", {}).items():
            print(f"  {m}: {v:.3f}" if isinstance(v, float) else f"  {m}: {v}")
    return 0


def cmd_matrix(args) -> int:
    """Matriz de investigación: estrategia x grupo x temporalidad con FDR, holdout y margen de costo."""
    from pathlib import Path
    from .matrix import operability_map, render_report, run_matrix
    from .universe import all_tickers
    prices = download_prices(all_tickers(), years=args.years, use_cache=True)
    if "SPY" not in prices:
        print("ERROR: sin datos de SPY."); return 2
    df = run_matrix(prices, dev_end=args.dev_end, n_boot=args.boot)
    if df.empty:
        print("Sin resultados."); return 2
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "matriz.csv", index=False)
    operability_map(df).to_csv(out / "mapa_operabilidad.csv", index=False)
    (out / "matriz.md").write_text(render_report(df, args.dev_end), encoding="utf-8")
    print(df["nivel"].value_counts().to_string()); print(f"Resultados en {out}/ (matriz.csv, mapa_operabilidad.csv, matriz.md)")
    return 0


def cmd_check_data(args) -> int:
    """Prueba tus claves gratuitas y reconcilia los últimos cierres entre fuentes (Tiingo / yfinance / Alpaca)."""
    from datetime import date, timedelta
    from .providers import fetch_validated
    start = (date.today() - timedelta(days=30)).isoformat()
    bad = 0
    for tk in args.tickers:
        df, rep = fetch_validated(tk, start)
        flag = "⛔ CUARENTENA" if rep["cuarentena"] else "✅"
        print(f"{flag} {tk}: estado={rep['estado']} fuentes={rep['fuentes']} errores={rep['errores']} {rep['detalle']}")
        bad += int(rep["cuarentena"])
    return 1 if bad else 0


def cmd_events(args) -> int:
    ev = upcoming(all_events(), pd.Timestamp.now(), args.days)
    print(ev[["date", "en_dias", "impact", "time_et", "name"]].to_string(index=False) if len(ev) else "Sin eventos en el período.")
    return 0


def cmd_refresh_events(args) -> int:
    """Baja el calendario OFICIAL del BLS (.ics) y lo guarda como override (reemplaza las fechas BLS incorporadas)."""
    import os
    import requests
    ua = os.environ.get("SEC_USER_AGENT", "quant-scanner personal use")
    r = requests.get(BLS_ICS_URL, headers={"User-Agent": ua}, timeout=20)
    r.raise_for_status()
    df = parse_ics(r.text)
    if df.empty:
        print("El .ics no trajo CPI/Empleo/PPI; no se modifica nada."); return 2
    df.to_csv(OVERRIDE_FILE, index=False)
    print(f"{len(df)} eventos del BLS guardados en {OVERRIDE_FILE}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="quant_scanner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--dry-run", action="store_true"); s.add_argument("--years", type=int, default=10)
    s.add_argument("--no-news", action="store_true", help="no consulta noticias ni balances (más rápido, menos contexto)")
    s.add_argument("--no-datacheck", action="store_true", help="no reconcilia precios con Tiingo/Alpaca")
    s.set_defaults(fn=cmd_scan)
    a = sub.add_parser("add-client"); a.add_argument("name"); a.add_argument("capital", type=float)
    a.add_argument("--risk", type=float, default=1.0); a.add_argument("--max-positions", type=int, default=8)
    a.add_argument("--max-pos-pct", type=float, default=25.0); a.add_argument("--fractional", action="store_true")
    a.add_argument("--chat", default=None)
    a.set_defaults(fn=lambda x: store.add_client(x.name, x.capital, x.risk, x.max_positions, x.max_pos_pct, x.fractional, x.chat) or 0)
    sub.add_parser("list-clients").set_defaults(fn=lambda x: print(pd.DataFrame(store.list_clients(False))[["id", "name", "capital", "risk_max_pct", "max_positions", "max_pos_pct", "active"]].to_string(index=False)) or 0)
    f = sub.add_parser("fill"); f.add_argument("client_id", type=int); f.add_argument("ticker"); f.add_argument("units", type=float)
    f.add_argument("--price", type=float, required=True); f.add_argument("--stop", type=float, required=True)
    f.add_argument("--date", default=str(date.today())); f.add_argument("--system", default="trend", choices=["trend", "pullback"])
    f.set_defaults(fn=lambda x: store.record_fill(x.client_id, x.ticker, x.units, x.date, x.price, x.stop, x.system) or 0)
    c = sub.add_parser("close"); c.add_argument("position_id", type=int); c.add_argument("--price", type=float, required=True)
    c.add_argument("--date", default=str(date.today()))
    c.set_defaults(fn=lambda x: store.close_position(x.position_id, x.date, x.price) or 0)
    m = sub.add_parser("matrix"); m.add_argument("--years", type=int, default=15); m.add_argument("--dev-end", default="2021-12-31")
    m.add_argument("--boot", type=int, default=1000); m.add_argument("--out", default="reports"); m.set_defaults(fn=cmd_matrix)
    k = sub.add_parser("check-data"); k.add_argument("tickers", nargs="*", default=["SPY", "QQQ", "AAPL"]); k.set_defaults(fn=cmd_check_data)
    e = sub.add_parser("events"); e.add_argument("--days", type=int, default=21); e.set_defaults(fn=cmd_events)
    sub.add_parser("refresh-events").set_defaults(fn=cmd_refresh_events)
    b = sub.add_parser("backtest"); b.add_argument("--capital", type=float, default=5000.0); b.add_argument("--years", type=int, default=10)
    b.add_argument("--universe", choices=["default", "etf"], default="default")
    b.set_defaults(fn=cmd_backtest)
    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
