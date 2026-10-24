"""Descarga y validación de precios. yfinance es una fuente NO oficial: sirve para
investigar y para probar, pero para producción conviene una API de datos contratada."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .logging_setup import get_logger

log = get_logger()
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
CACHE_DIR = Path(os.environ.get("QS_CACHE_DIR", ".qs_cache"))


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1) if df.columns.nlevels > 1 else df
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        return None
    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df["Volume"] = df["Volume"].fillna(0)
    return df if len(df) > 0 else None


def drop_partial_bar(df: pd.DataFrame, ticker: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Saca la barra diaria de HOY si todavía está en formación (señales sobre una barra incompleta son falsas).
    Acciones/ETFs: la rueda de EE.UU. se considera cerrada recién a las 16:20 ET. Cripto (-USD): la barra de hoy (UTC)
    siempre está incompleta."""
    if df is None or df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC") if now is None else (pd.Timestamp(now).tz_localize("UTC") if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now))
    last = df.index[-1].normalize()
    if ticker.upper().endswith("-USD"):
        return df.iloc[:-1] if last >= now.tz_localize(None).normalize() else df
    et = now.tz_convert("America/New_York")
    if last >= et.tz_localize(None).normalize() and et.weekday() < 5 and (et.hour * 60 + et.minute) < 16 * 60 + 20:
        return df.iloc[:-1]
    return df


def freshness_report(prices: dict[str, pd.DataFrame], ref: str = "SPY", now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Frescura por activo. Para acciones se compara contra SPY (a prueba de feriados): un activo cuya última barra es
    anterior a la de SPY está vencido. Cripto: vencido si no tiene la barra de ayer (UTC)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else (pd.Timestamp(now).tz_localize("UTC") if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now))
    ref_last = prices[ref].index[-1].normalize() if ref in prices else None
    rows = []
    for tk, df in prices.items():
        if tk.startswith("^"):
            continue
        last = df.index[-1].normalize()
        if tk.upper().endswith("-USD"):
            behind = int((now.tz_localize(None).normalize() - last).days) - 1
            ok = behind <= 0
        else:
            behind = int(len(pd.bdate_range(last, ref_last)) - 1) if ref_last is not None and last < ref_last else 0
            ok = behind == 0
        rows.append({"ticker": tk, "ultima_barra": last, "ruedas_de_atraso": max(behind, 0), "al_dia": bool(ok)})
    return pd.DataFrame(rows, columns=["ticker", "ultima_barra", "ruedas_de_atraso", "al_dia"])


def _tiingo_fallback(missing: list[str], years: int, out: dict, max_calls: int = 45) -> None:
    """Si Yahoo no devolvió un activo y hay TIINGO_TOKEN, se pide a Tiingo (plan gratis: ~50 pedidos/hora, por eso el tope)."""
    if not missing or not os.environ.get("TIINGO_TOKEN"):
        return
    from .providers import ProviderError, fetch_tiingo, supports
    start = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
    calls = 0
    for tk in missing:
        if calls >= max_calls or not supports("tiingo", tk):
            continue
        calls += 1
        try:
            df = fetch_tiingo(tk, start)
        except ProviderError as e:
            log.warning("respaldo Tiingo falló para %s (%s)", tk, e)
            continue
        if df is not None and len(df):
            out[tk] = df
            df.to_csv(CACHE_DIR / f"{tk.replace('/', '_')}.csv")


def download_prices(tickers: list[str], years: int = 10, use_cache: bool = True,
                    max_cache_age_hours: float = 1.0) -> dict[str, pd.DataFrame]:
    """Devuelve {ticker: DataFrame OHLCV ajustado}. Usa caché en disco como respaldo."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []
    for tk in tickers:
        f = CACHE_DIR / f"{tk.replace('/', '_')}.csv"
        if use_cache and f.exists():
            age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
            if age < timedelta(hours=max_cache_age_hours):
                df = _clean(pd.read_csv(f, index_col=0, parse_dates=True))
                if df is not None:
                    out[tk] = df
                    continue
        to_fetch.append(tk)

    if to_fetch:
        try:
            import yfinance as yf
            start = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
            raw = yf.download(to_fetch, start=start, auto_adjust=True, group_by="ticker",
                              progress=False, threads=True)
            for tk in to_fetch:
                try:
                    sub = raw[tk] if len(to_fetch) > 1 else raw
                    df = _clean(sub)
                except Exception:
                    df = None
                if df is not None:
                    out[tk] = df
                    df.to_csv(CACHE_DIR / f"{tk.replace('/', '_')}.csv")
        except Exception as e:      # noqa: BLE001  (se registra y se usa la caché: un fallo de red no debe tumbar el escaneo)
            log.warning("descarga de precios falló (%s); se usa la caché disponible", type(e).__name__)
        _tiingo_fallback([t for t in to_fetch if t not in out], years, out)
        # respaldo: caché vieja antes que nada
        for tk in to_fetch:
            f = CACHE_DIR / f"{tk.replace('/', '_')}.csv"
            if tk not in out and f.exists():
                df = _clean(pd.read_csv(f, index_col=0, parse_dates=True))
                if df is not None:
                    out[tk] = df
    return {tk: drop_partial_bar(df, tk) for tk, df in out.items()}


def stale_tickers(prices: dict[str, pd.DataFrame], max_days: int = 5) -> list[str]:
    """Activos cuyo último dato es más viejo que max_days (datos vencidos = señales falsas)."""
    now = pd.Timestamp.now().normalize()
    return [tk for tk, df in prices.items() if (now - df.index[-1]).days > max_days]
