"""Contexto de una señal: eventos macro + noticias con credibilidad + balances. Reduce o veta, nunca agrega."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .config import AppConfig
from .events import event_risk
from .news import NewsItem, assess, decide, parse_yf_news


def next_earnings_date(ticker: str) -> pd.Timestamp | None:
    """Fecha ESTIMADA de balance (yfinance, no oficial). Se usa solo para prudencia y se muestra como estimada."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        today = pd.Timestamp.now().normalize()
        fut = sorted(pd.Timestamp(d) for d in (dates or []) if pd.Timestamp(d) >= today)
        return fut[0] if fut else None
    except Exception:
        return None


def fetch_news(ticker: str, include_sec: bool = True) -> tuple[list[NewsItem], list[str]]:
    """Trae noticias de yfinance y 8-K de la SEC. Devuelve (items, errores): un error NUNCA se toma como 'sin noticias'."""
    items, errors = [], []
    try:
        import yfinance as yf
        items += parse_yf_news(yf.Ticker(ticker).news, ticker)
    except Exception as e:
        errors.append(f"yfinance: {type(e).__name__}")
    if include_sec:
        try:
            from .edgar import eight_k_news
            items += eight_k_news(ticker)
        except Exception as e:
            errors.append(f"SEC: {e}" if "SEC_USER_AGENT" in str(e) else f"SEC: {type(e).__name__}")
    return items, errors


def evaluate_context(items: list[NewsItem], today: pd.Timestamp, events: pd.DataFrame, cfg: AppConfig,
                     hold_bdays: int, earnings: pd.Timestamp | None = None, fetch_errors: list[str] | None = None,
                     now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    mult, notes = 1.0, []
    ev_mult, ev_flags = event_risk(events, today, cfg.events)
    if ev_flags:
        mult *= ev_mult
        notes += [f"📅 Evento de alto impacto cerca de la ejecución (riesgo x{ev_mult:g}): {f}" for f in ev_flags]
    if earnings is not None:
        horizon_end = pd.Timestamp(today).normalize() + pd.offsets.BDay(hold_bdays + 1)
        if pd.Timestamp(today).normalize() <= earnings <= horizon_end:
            mult *= cfg.events.earnings_risk_mult
            notes.append(f"📊 Balance estimado el {earnings:%d/%m} dentro del horizonte de la operación (riesgo x{cfg.events.earnings_risk_mult:g}; fecha NO oficial, verificar)")
    decision, news_notes = decide(assess(items, now))
    notes += news_notes
    if fetch_errors:
        notes.append("⚠️ No pude verificar noticias (" + "; ".join(fetch_errors) + "): revisá a mano antes de operar")
    return {"decision": decision, "risk_mult": mult, "notes": notes}
