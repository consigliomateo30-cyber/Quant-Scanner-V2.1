"""Cliente mínimo de la SEC (EDGAR). Fuente OFICIAL y gratuita; exige declarar un User-Agent con contacto
y limita a 10 pedidos/segundo. Definí SEC_USER_AGENT="Nombre Apellido tu@email.com".
NOTA: los parsers están probados con datos de ejemplo; la conexión en vivo NO se pudo probar en mi entorno."""
from __future__ import annotations

import os
import time
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from .news import NewsItem

_LAST = [0.0]
_TICKERS: dict[str, str] = {}


def _ua() -> str | None:
    return os.environ.get("SEC_USER_AGENT")


def _get(url: str, as_json: bool = True):
    ua = _ua()
    if not ua:
        raise RuntimeError("Falta SEC_USER_AGENT (la SEC exige identificarte: 'Nombre Apellido email').")
    wait = 0.15 - (time.time() - _LAST[0])
    if wait > 0:
        time.sleep(wait)
    _LAST[0] = time.time()
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=20)
    r.raise_for_status()
    return r.json() if as_json else r.text


def ticker_to_cik(ticker: str) -> str | None:
    if not _TICKERS:
        js = _get("https://www.sec.gov/files/company_tickers.json")
        for v in js.values():
            _TICKERS[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
    return _TICKERS.get(ticker.upper().replace("-", "."))


def parse_submissions(js: dict) -> pd.DataFrame:
    rec = (js.get("filings") or {}).get("recent") or {}
    n = len(rec.get("form", []))
    if n == 0:
        return pd.DataFrame(columns=["form", "filing_date", "accepted", "items", "accession", "primary_doc"])
    get = lambda k: rec.get(k, [None] * n)
    df = pd.DataFrame({"form": get("form"), "filing_date": pd.to_datetime(get("filingDate")),
                       "accepted": pd.to_datetime(get("acceptanceDateTime"), utc=True, errors="coerce"),
                       "items": [tuple(x.strip() for x in (i or "").split(",") if x.strip()) for i in get("items")],
                       "accession": get("accessionNumber"), "primary_doc": get("primaryDocument")})
    return df


def recent_filings(ticker: str) -> tuple[pd.DataFrame, str | None]:
    cik = ticker_to_cik(ticker)
    if not cik:
        return pd.DataFrame(), None
    return parse_submissions(_get(f"https://data.sec.gov/submissions/CIK{cik}.json")), cik


def eight_k_news(ticker: str, days: int = 10) -> list[NewsItem]:
    df, cik = recent_filings(ticker)
    if df.empty:
        return []
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    out = []
    for r in df[(df["form"] == "8-K") & (df["accepted"] >= since)].itertuples():
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{r.accession.replace('-', '')}/{r.primary_doc}"
        out.append(NewsItem(f"{ticker} presentó 8-K (ítems {', '.join(r.items) or 's/d'})", "sec.gov", url,
                            r.accepted.to_pydatetime(), (ticker,), form="8-K", items=tuple(r.items), kind="filing"))
    return out


def _txt(el, path: str) -> str | None:
    x = el.find(path)
    return x.text.strip() if x is not None and x.text else None


def parse_form4_xml(xml_text: str) -> list[dict]:
    """Devuelve las transacciones NO derivadas de un Form 4 (código P = compra en mercado abierto)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    names = [_txt(o, "reportingOwnerId/rptOwnerName") for o in root.findall("reportingOwner")]
    rel = root.find("reportingOwner/reportingOwnerRelationship")
    is_off = (_txt(rel, "isOfficer") in ("1", "true")) if rel is not None else False
    is_dir = (_txt(rel, "isDirector") in ("1", "true")) if rel is not None else False
    title = _txt(rel, "officerTitle") if rel is not None else None
    rows = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        try:
            shares = float(_txt(t, "transactionAmounts/transactionShares/value") or 0)
            price = float(_txt(t, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        rows.append({"symbol": _txt(root, "issuer/issuerTradingSymbol"), "insider": names[0] if names else None,
                     "is_officer": is_off, "is_director": is_dir, "title": title,
                     "trade_date": _txt(t, "transactionDate/value"), "code": _txt(t, "transactionCoding/transactionCode"),
                     "ad": _txt(t, "transactionAmounts/transactionAcquiredDisposedCode/value"),
                     "shares": shares, "price": price, "value_usd": shares * price})
    return rows


def fetch_form4_purchases(ticker: str, days: int = 45, max_filings: int = 25) -> pd.DataFrame:
    """Compras en mercado abierto (código P) con fecha de PRESENTACIÓN (punto en el tiempo, sin look-ahead)."""
    df, cik = recent_filings(ticker)
    if df.empty:
        return pd.DataFrame()
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    rows = []
    for r in df[(df["form"] == "4") & (df["accepted"] >= since)].head(max_filings).itertuples():
        doc = (r.primary_doc or "").split("/")[-1]           # el .xml crudo, no la versión XSL/HTML
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{r.accession.replace('-', '')}/{doc}"
        try:
            for t in parse_form4_xml(_get(url, as_json=False)):
                if t["code"] == "P" and t["ad"] == "A":
                    rows.append({**t, "ticker": ticker.upper(), "filed": r.accepted})
        except Exception:
            continue
    return pd.DataFrame(rows)
