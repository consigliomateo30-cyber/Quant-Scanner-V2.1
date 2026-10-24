"""Noticias con credibilidad. Principio: una noticia NUNCA genera una compra. Solo puede
(1) VETAR una entrada si es un hecho grave y CONFIRMADO, (2) AVISAR, o (3) quedar como información.
Una noticia sin confirmar no mueve ningún parámetro."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------- fuentes por nivel
TIER1 = {"sec.gov", "federalreserve.gov", "bls.gov", "bea.gov", "treasury.gov", "fred.stlouisfed.org", "census.gov"}
TIER2 = {"reuters.com", "bloomberg.com", "apnews.com", "wsj.com", "ft.com", "dowjones.com", "djnewswires.com"}
TIER2_NAMES = {"reuters", "bloomberg", "associated press", "ap", "the wall street journal", "wsj", "financial times", "dow jones"}
TIER3 = {"cnbc.com", "marketwatch.com", "barrons.com", "finance.yahoo.com", "yahoo.com", "businesswire.com", "prnewswire.com",
         "globenewswire.com", "investors.com", "forbes.com", "fortune.com", "nytimes.com", "washingtonpost.com", "economist.com"}
TIER3_NAMES = {"cnbc", "marketwatch", "barron's", "barrons", "yahoo finance", "business wire", "pr newswire", "globenewswire",
               "investor's business daily", "forbes", "fortune", "the new york times"}
BASE_SCORE = {1: 90, 2: 70, 3: 45, 4: 20}

HEDGE_WORDS = re.compile(r"\b(reportedly|rumou?rs?|rumou?red|sources? say|according to sources|people familiar|speculat\w+|unconfirmed|"
                         r"could|may|might|possibly|allegedly|claims?|según fuentes|se rumorea)\b", re.I)
NEG_KEYWORDS = re.compile(r"\b(bankruptc\w*|chapter 11|fraud\w*|sec investigation|subpoena\w*|restat\w+|delist\w+|going concern|"
                          r"profit warning|cuts? guidance|lower\w* guidance|withdraws? guidance|recall\w*|halt\w*|default\w*|"
                          r"accounting irregularit\w+|class action)\b", re.I)
# Ítems del 8-K de la SEC: hechos graves (negativos) y eventos a vigilar
SEC_NEG_ITEMS = {"1.03": "quiebra/concurso", "3.01": "aviso de deslistado", "4.02": "no confiar en balances previos (restatement)",
                 "2.04": "aceleración de obligaciones", "1.05": "incidente de ciberseguridad material"}
SEC_WATCH_ITEMS = {"2.02": "resultados (balance)", "5.02": "cambio de directivos", "4.01": "cambio de auditor"}


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published: datetime                      # siempre UTC
    tickers: tuple[str, ...] = ()
    form: str | None = None                  # "8-K", etc. para presentaciones oficiales
    items: tuple[str, ...] = ()              # ítems del 8-K
    kind: str = "article"


@dataclass
class Scored:
    headline: str
    label: str                                # confirmado | confirmado (prensa) | probable | sin confirmar
    score: int
    best_tier: int
    n_sources: int
    hedged: bool
    negative: bool
    impact: str                               # alto | vigilar | info
    age_hours: float
    reason: str
    items: list[NewsItem] = field(default_factory=list)


def clean(text: str, limit: int = 240) -> str:
    t = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", t).strip()[:limit]


def source_tier(source: str, url: str = "") -> int:
    dom = (urlparse(url).netloc or "").lower().removeprefix("www.")
    for d in TIER1:
        if dom == d or dom.endswith("." + d):
            return 1
    name = (source or "").strip().lower()
    for d in TIER2:
        if dom == d or dom.endswith("." + d):
            return 2
    if name in TIER2_NAMES:
        return 2
    for d in TIER3:
        if dom == d or dom.endswith("." + d):
            return 3
    return 3 if name in TIER3_NAMES else 4


def _tokens(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-záéíóúñ0-9]{4,}", title.lower())}


def _similar(a: set[str], b: set[str]) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= 0.45


def cluster(items: list[NewsItem], window_hours: float = 48.0) -> list[list[NewsItem]]:
    """Agrupa notas que cuentan lo mismo (títulos parecidos, mismo activo, ventana corta)."""
    groups: list[list[NewsItem]] = []
    for it in sorted(items, key=lambda x: x.published):
        tk = _tokens(it.title)
        for g in groups:
            if abs((it.published - g[0].published).total_seconds()) / 3600 <= window_hours and (
                    (it.form and g[0].form == it.form and set(it.items) == set(g[0].items)) or _similar(tk, _tokens(g[0].title))):
                g.append(it)
                break
        else:
            groups.append([it])
    return groups


def score_cluster(g: list[NewsItem], now: datetime | None = None) -> Scored:
    now = now or datetime.now(timezone.utc)
    tiers = [source_tier(i.source, i.url) for i in g]
    best = min(tiers)
    # fuentes independientes: distintas por nombre/dominio, de nivel 1 a 3
    indep = {(i.source or urlparse(i.url).netloc).lower() for i, t in zip(g, tiers) if t <= 3}
    hedged = any(HEDGE_WORDS.search(i.title) for i in g) and best > 1
    sec_items = {x for i in g for x in i.items}
    neg_sec = sorted(sec_items & SEC_NEG_ITEMS.keys())
    neg_kw = any(NEG_KEYWORDS.search(i.title) for i in g)
    score = BASE_SCORE[best] + min(24, 8 * max(len(indep) - 1, 0)) - (20 if hedged else 0)
    score = int(max(0, min(100, score)))
    if best == 1:
        label = "confirmado"
    elif len([1 for t in tiers if t <= 2]) >= 2 and not hedged and len(indep) >= 2:
        label = "confirmado (prensa)"
    elif score >= 60:
        label = "probable"
    else:
        label = "sin confirmar"
    negative = bool(neg_sec) or neg_kw
    if neg_sec:
        impact, reason = "alto", "SEC 8-K: " + ", ".join(f"{x} {SEC_NEG_ITEMS[x]}" for x in neg_sec)
    elif sec_items & SEC_WATCH_ITEMS.keys():
        w = sorted(sec_items & SEC_WATCH_ITEMS.keys())
        impact, reason = "vigilar", "SEC 8-K: " + ", ".join(f"{x} {SEC_WATCH_ITEMS[x]}" for x in w)
    elif neg_kw:
        impact, reason = "alto", "palabras clave de riesgo en el titular (dirección estimada por texto, poco precisa)"
    else:
        impact, reason = "info", "sin hechos de alto impacto detectados"
    age = (now - max(i.published for i in g)).total_seconds() / 3600
    head = max(g, key=lambda i: (-source_tier(i.source, i.url), i.published)).title
    return Scored(head, label, score, best, len(indep), hedged, negative, impact, round(age, 1), reason, g)


def assess(items: list[NewsItem], now: datetime | None = None) -> list[Scored]:
    now = now or datetime.now(timezone.utc)
    out = [score_cluster(g, now) for g in cluster(items)]
    return sorted(out, key=lambda s: (-(s.impact == "alto"), -s.score, s.age_hours))


def decide(scored: list[Scored], veto_days: float = 10.0) -> tuple[str, list[str]]:
    """VETO solo si hay un hecho grave y CONFIRMADO reciente. Rumores solo informan."""
    notes, decision = [], "OK"
    for s in scored:
        old = s.age_hours > veto_days * 24
        confirmed = s.label in ("confirmado", "confirmado (prensa)")
        if s.impact == "alto" and s.negative and confirmed and not old:
            decision = "VETO"
            notes.append(f"⛔ {s.label.upper()}: {s.reason} — «{s.headline[:90]}» ({s.age_hours:.0f} h)")
        elif s.impact in ("alto", "vigilar") and not old and s.label != "sin confirmar":
            if decision != "VETO":
                decision = "AVISO"
            notes.append(f"⚠️ {s.label}: {s.reason} — «{s.headline[:90]}» ({s.age_hours:.0f} h)")
        elif s.impact == "alto" and not old:
            notes.append(f"ℹ️ rumor SIN confirmar (no se usa): «{s.headline[:90]}»")
    return decision, notes


# ---------------------------------------------------------------- parsers de fuentes
def _dt(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    s = str(value).replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        from email.utils import parsedate_to_datetime
        d = parsedate_to_datetime(str(value))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_yf_news(raw: list[dict], ticker: str) -> list[NewsItem]:
    """yfinance cambió el formato: soporta el nuevo (content anidado) y el viejo (plano)."""
    out = []
    for r in raw or []:
        try:
            c = r.get("content") or r
            title = clean(c.get("title", ""))
            prov = c.get("provider") or {}
            source = prov.get("displayName") if isinstance(prov, dict) else (c.get("publisher") or "")
            url = ((c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else None) or c.get("link") or ""
            when = c.get("pubDate") or c.get("providerPublishTime")
            if title and when:
                out.append(NewsItem(title, source or "", url, _dt(when), (ticker,)))
        except Exception:
            continue
    return out


def parse_rss(xml_text: str, default_source: str, tickers: tuple[str, ...] = ()) -> list[NewsItem]:
    if len(xml_text) > 3_000_000:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title, link, pub = clean(it.findtext("title", "")), (it.findtext("link") or "").strip(), it.findtext("pubDate")
        if title and pub:
            try:
                out.append(NewsItem(title, default_source, link, _dt(pub), tickers))
            except Exception:
                continue
    return out
