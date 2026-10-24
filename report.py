"""Informe HTML. Todo el texto dinámico se escapa; incluye métricas reales del backtest
y advertencias de riesgo. Es informativo: no constituye recomendación de inversión."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def fecha_es(dt: datetime) -> str:
    return f"{dt.day} de {MESES[dt.month - 1]} de {dt.year}, {dt:%H:%M}"


def _pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def _num(x, d=2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{d}f}"


def build_report(scan: pd.DataFrame, info: dict, bt: dict | None, capital: float) -> str:
    e = html.escape
    signals = scan[scan["entrada"].astype(bool)] if len(scan) else scan.iloc[0:0]
    if len(signals):
        signals = signals.sort_values("p_ganar", ascending=False)
    rows = "".join(
        f"<tr><td><b>{e(r.ticker)}</b></td><td>{_num(r.cierre)}</td><td>{_pct(r.p_ganar)}</td>"
        f"<td>{_num(r.stop_sugerido)}</td><td>{_num(r.riesgo_pct)}%</td></tr>"
        for r in signals.itertuples()) or "<tr><td colspan='5'>Sin señales de entrada en la fecha.</td></tr>"

    perf = ""
    if bt and "error" not in bt:
        def col(k, label):
            m, t = bt[k]["metrics"], bt[k].get("trade_metrics", {})
            return (f"<tr><td>{label}</td><td>{_pct(m.get('CAGR'))}</td><td>{_pct(m.get('MaxDD'))}</td>"
                    f"<td>{_num(m.get('Sharpe'))}</td><td>{t.get('Trades', '—')}</td>"
                    f"<td>{_pct(t.get('Tasa de acierto'))}</td><td>{_num(t.get('R medio (expectancy)'))}</td></tr>")
        perf = ("<h2>Backtest walk-forward (con costos)</h2><table><tr><th>Variante</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th>"
                "<th>Trades</th><th>Acierto</th><th>R medio</th></tr>"
                + col("base", "Sin meta-modelo") + col("modelo", "Con meta-modelo")
                + (f"<tr><td>SPY (buy &amp; hold)</td><td>{_pct(bt['spy']['metrics'].get('CAGR'))}</td><td>{_pct(bt['spy']['metrics'].get('MaxDD'))}</td>"
                   f"<td>{_num(bt['spy']['metrics'].get('Sharpe'))}</td><td>—</td><td>—</td><td>—</td></tr>" if "spy" in bt else "")
                + f"</table><p>Ventana evaluada desde {e(str(bt['eval_start'].date()))}. Resultados pasados no garantizan resultados futuros.</p>")
    else:
        perf = "<h2>Backtest</h2><p>No hay resultados de backtest cargados en esta sesión.</p>"

    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Quant Scanner — Informe</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;margin:32px;color:#18181b;line-height:1.5}}
h1{{border-bottom:3px solid #d4af37;padding-bottom:8px}}table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
th,td{{border:1px solid #e4e4e7;padding:8px;text-align:left}}th{{background:#18181b;color:#fff}}.note{{font-size:12px;color:#71717a}}</style></head><body>
<h1>Quant Scanner — Informe de señales</h1>
<p>Emitido: {e(fecha_es(datetime.now()))} · Capital de referencia: USD {capital:,.0f} · Régimen de mercado (SPY &gt; SMA200): <b>{'alcista' if info.get('regime_ok') else 'bajista (sin compras)'}</b></p>
<p>Sistema: breakout de Donchian con stop inicial en ATR y salida por mínimo de 10 días. Meta-modelo ({e(str(info.get('model_mode')))}, {info.get('n_train', 0)} trades de entrenamiento) que estima la probabilidad de ganar; el tamaño de la posición usa riesgo fijo por trade con Kelly fraccionado y topes por activo.</p>
<h2>Señales de entrada del día</h2><table><tr><th>Activo</th><th>Cierre</th><th>Prob. estimada</th><th>Stop inicial</th><th>Riesgo</th></tr>{rows}</table>
{perf}
<p class="note">Documento informativo generado automáticamente. No constituye asesoramiento ni recomendación de inversión. Operar en mercados implica riesgo de pérdida de capital.</p>
</body></html>"""
