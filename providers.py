"""Proveedores de datos de mercado (planes GRATUITOS) con reconciliación entre fuentes.

 - Tiingo   : EOD consolidado y ajustado. Token gratis (TIINGO_TOKEN). Límite free ~50 pedidos/hora, 1000/día.
 - Alpaca   : barras IEX (volumen PARCIAL: la bolsa IEX es una fracción del volumen total). Cuenta gratis
              (ALPACA_KEY_ID / ALPACA_SECRET_KEY). Sirve para precios; NO para features de volumen.
 - Finnhub  : cotización actual (FINNHUB_TOKEN), 60 pedidos/min en el plan gratis.
 - yfinance : no oficial; respaldo.
Los límites de los planes gratis cambian: verificá en cada sitio. Las conexiones en vivo NO se pudieron probar
desde mi entorno (sin internet); los parsers, el backoff y la reconciliación están probados con datos de ejemplo.
Las claves viajan en HEADERS (no en la URL) y nunca se incluyen en mensajes de error."""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .data import _clean
from .logging_setup import get_logger

log = get_logger()
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class ProviderError(RuntimeError):
    pass


def http_get(url: str, params: dict | None = None, headers: dict | None = None, retries: int = 3, timeout: int = 20, sleep=time.sleep):
    """GET con backoff ante 429/5xx. Nunca expone URL con parámetros ni cabeceras en el error."""
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if k == retries - 1:
                raise ProviderError(f"red: {type(e).__name__}") from None
            sleep(2 ** k)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if k == retries - 1:
                raise ProviderError(f"HTTP {r.status_code}")
            sleep(2 ** k)
            continue
        if r.status_code in (401, 403):
            raise ProviderError("credenciales rechazadas")
        if not r.ok:
            raise ProviderError(f"HTTP {r.status_code}")
        return r.json()
    raise ProviderError("sin respuesta")


def supports(provider: str, ticker: str) -> bool:
    crypto_or_index = ticker.upper().endswith("-USD") or ticker.startswith("^")
    return provider == "yfinance" or not crypto_or_index


# ----------------------------------------------------------------- Tiingo
def parse_tiingo(js: list[dict]) -> pd.DataFrame | None:
    if not js:
        return None
    df = pd.DataFrame(js)
    need = {"date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"}
    if not need <= set(df.columns):
        return None
    df.index = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"adjOpen": "Open", "adjHigh": "High", "adjLow": "Low", "adjClose": "Close", "adjVolume": "Volume"})
    return _clean(df[OHLCV])


def fetch_tiingo(ticker: str, start: str) -> pd.DataFrame | None:
    token = os.environ.get("TIINGO_TOKEN")
    if not token:
        raise ProviderError("falta TIINGO_TOKEN")
    js = http_get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices", {"startDate": start, "format": "json"},
                  {"Authorization": f"Token {token}", "Content-Type": "application/json"})
    return parse_tiingo(js)


# ----------------------------------------------------------------- Alpaca
def parse_alpaca(js: dict, symbol: str) -> pd.DataFrame | None:
    bars = (js or {}).get("bars")
    if isinstance(bars, dict):
        bars = bars.get(symbol)
    if not bars:
        return None
    df = pd.DataFrame(bars).rename(columns={"t": "date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    if not set(OHLCV + ["date"]) <= set(df.columns):
        return None
    df.index = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return _clean(df[OHLCV])


def fetch_alpaca(ticker: str, start: str) -> pd.DataFrame | None:
    kid, sec = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if not kid or not sec:
        raise ProviderError("faltan ALPACA_KEY_ID / ALPACA_SECRET_KEY")
    hdr, frames, token = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}, [], None
    for _ in range(20):                                               # paginación acotada
        params = {"symbols": ticker, "timeframe": "1Day", "start": start, "adjustment": "all", "feed": "iex", "limit": 10000}
        if token:
            params["page_token"] = token
        js = http_get("https://data.alpaca.markets/v2/stocks/bars", params, hdr)
        part = parse_alpaca(js, ticker)
        if part is not None:
            frames.append(part)
        token = js.get("next_page_token")
        if not token:
            break
    return pd.concat(frames).sort_index() if frames else None


# ----------------------------------------------------------------- Finnhub (cotización actual)
def parse_finnhub_quote(js: dict) -> dict | None:
    if not js or not js.get("c"):
        return None
    return {"precio": float(js["c"]), "var_pct": float(js.get("dp") or 0.0), "maximo": js.get("h"), "minimo": js.get("l"),
            "apertura": js.get("o"), "cierre_previo": js.get("pc"), "hora_utc": pd.to_datetime(js.get("t"), unit="s", utc=True) if js.get("t") else None}


def fetch_quote_finnhub(ticker: str) -> dict | None:
    token = os.environ.get("FINNHUB_TOKEN")
    if not token:
        raise ProviderError("falta FINNHUB_TOKEN")
    return parse_finnhub_quote(http_get("https://finnhub.io/api/v1/quote", {"symbol": ticker}, {"X-Finnhub-Token": token}))


# ----------------------------------------------------------------- yfinance
def fetch_yfinance(ticker: str, start: str) -> pd.DataFrame | None:
    import yfinance as yf
    return _clean(yf.download(ticker, start=start, auto_adjust=True, progress=False))


FETCHERS = {"tiingo": fetch_tiingo, "alpaca": fetch_alpaca, "yfinance": fetch_yfinance}


# ----------------------------------------------------------------- reconciliación
def reconcile(frames: dict[str, pd.DataFrame], tol: float = 0.005, last_n: int = 5) -> dict:
    """Compara los últimos cierres entre fuentes. Una diferencia > tol (0.5%) entre dos fuentes = CONFLICTO:
    puede ser un tick malo, un split no ajustado o un dato vencido. El activo se pone en cuarentena."""
    names = [n for n, f in frames.items() if f is not None and len(f)]
    out = {"fuentes": names, "estado": "ok", "detalle": [], "cuarentena": False}
    if len(names) < 2:
        out["estado"] = "una sola fuente"
        return out
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = frames[a].index.intersection(frames[b].index)[-last_n:]
            if len(common) == 0:
                out["detalle"].append(f"{a} vs {b}: sin fechas en común")
                out["estado"] = "advertencia"
                continue
            diff = (frames[a].loc[common, "Close"] / frames[b].loc[common, "Close"] - 1).abs().max()
            if diff > tol:
                out["estado"], out["cuarentena"] = "conflicto", True
                out["detalle"].append(f"{a} vs {b}: diferencia máxima {diff:.2%} en {last_n} cierres")
    last_dates = {n: frames[n].index[-1] for n in names}
    if len(set(last_dates.values())) > 1 and out["estado"] == "ok":
        out["estado"] = "advertencia"
        out["detalle"].append("las fuentes tienen distinta fecha de último dato: " + ", ".join(f"{n}={d:%Y-%m-%d}" for n, d in last_dates.items()))
    return out


def fetch_validated(ticker: str, start: str, order=("tiingo", "yfinance", "alpaca")) -> tuple[pd.DataFrame | None, dict]:
    """Baja de cada fuente disponible, reconcilia y devuelve la primaria + el reporte. Una fuente caída no frena el resto."""
    frames, errors = {}, {}
    for name in order:
        if not supports(name, ticker):
            continue
        try:
            frames[name] = FETCHERS[name](ticker, start)
        except ProviderError as e:
            errors[name] = str(e)
        except Exception as e:      # noqa: BLE001  (un proveedor roto no debe tumbar el escaneo)
            errors[name] = type(e).__name__
    rep = reconcile(frames)
    rep["errores"] = errors
    primary = next((frames[n] for n in order if n in frames and frames[n] is not None and len(frames[n])), None)
    if primary is None:
        rep["estado"], rep["cuarentena"] = "sin datos", True
    return primary, rep
