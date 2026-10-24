"""Calendario de eventos de mercado (Fed y datos macro de alto impacto).

Las fechas de abajo se verificaron el 2026-10-20 contra varias fuentes (calendario de la Fed
vía terceros, calendario del BLS y su publicación oficial). Las fechas FUTURAS pueden cambiar
(ej.: cierres de gobierno movieron el empleo y el CPI en 2025/2026): por eso `refresh_from_ics`
permite actualizar desde el calendario oficial del BLS, y `events_override.csv` permite corregir a mano.
CONFIRMAR SIEMPRE en federalreserve.gov y bls.gov antes de operar un evento puntual."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import EventConfig

VERIFIED_ON = "2026-10-20"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
OVERRIDE_FILE = Path("events_override.csv")   # columnas: date,name,impact,time_et,source

# (fecha, nombre, impacto, hora ET, fuente)
_FOMC_DECISIONS = [
    "2026-01-28", "2026-03-18*", "2026-04-29", "2026-06-17*", "2026-07-29", "2026-09-16*", "2026-10-28",
    "2026-12-09*", "2027-01-27", "2027-03-17*", "2027-04-28", "2027-06-09*", "2027-07-28", "2027-09-15*",
    "2027-10-27", "2027-12-08*",
]   # * = con proyecciones económicas (SEP / dot plot). Decisión 14:00 ET, conferencia 14:30 ET.
_BUILTIN = [
    ("2026-11-06", "Empleo (NFP) de octubre", "alto", "08:30", "BLS"),
    ("2026-12-04", "Empleo (NFP) de noviembre", "alto", "08:30", "BLS"),
    ("2026-11-10", "CPI de octubre", "alto", "08:30", "BLS"),
    ("2026-12-10", "CPI de noviembre", "alto", "08:30", "BLS"),
    ("2026-12-15", "PPI de noviembre", "medio", "08:30", "BLS"),
    ("2026-11-18", "Minutas FOMC (reunión de octubre)", "medio", "14:00", "Fed"),
    ("2026-12-30", "Minutas FOMC (reunión de diciembre)", "medio", "14:00", "Fed"),
    ("2026-11-25", "Beige Book", "medio", "14:00", "Fed"),
    ("2027-01-13", "Beige Book", "medio", "14:00", "Fed"),
]


def _builtin_frame() -> pd.DataFrame:
    rows = list(_BUILTIN)
    for d in _FOMC_DECISIONS:
        sep = d.endswith("*")
        rows.append((d.rstrip("*"), "Decisión FOMC (tasas) + conferencia de prensa" + (" + proyecciones/dot plot" if sep else ""),
                     "alto", "14:00", "Fed"))
    df = pd.DataFrame(rows, columns=["date", "name", "impact", "time_et", "source"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def parse_ics(text: str) -> pd.DataFrame:
    """Parsea un .ics del BLS y se queda con CPI, Empleo y PPI. Maneja líneas plegadas y fechas con/sin hora."""
    text = re.sub(r"\r?\n[ \t]", "", text)                      # unfold
    rows = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, flags=re.S):
        m_s = re.search(r"^SUMMARY[^:]*:(.*)$", block, flags=re.M)
        m_d = re.search(r"^DTSTART[^:]*:(\d{8})(?:T(\d{2})(\d{2})(\d{2})Z?)?", block, flags=re.M)
        if not m_s or not m_d:
            continue
        name = m_s.group(1).strip()
        low = name.lower()
        if "consumer price index" in low:
            impact = "alto"
        elif "employment situation" in low and "veterans" not in low and "state" not in low and "metropolitan" not in low:
            impact = "alto"
        elif "producer price index" in low:
            impact = "medio"
        else:
            continue
        date = pd.to_datetime(m_d.group(1), format="%Y%m%d")
        rows.append((date, name, impact, "08:30", "BLS (ics)"))
    return pd.DataFrame(rows, columns=["date", "name", "impact", "time_et", "source"])


def refresh_from_ics(ics_text: str) -> pd.DataFrame:
    """Reemplaza las fechas BLS incorporadas por las del calendario oficial (si el ics trae datos)."""
    base = _builtin_frame()
    new = parse_ics(ics_text)
    if new.empty:
        return base
    keep = base[base["source"] != "BLS"]
    return pd.concat([keep, new], ignore_index=True).sort_values("date").reset_index(drop=True)


def all_events(extra: pd.DataFrame | None = None) -> pd.DataFrame:
    df = _builtin_frame()
    if OVERRIDE_FILE.exists():
        o = pd.read_csv(OVERRIDE_FILE, parse_dates=["date"])
        if o["source"].astype(str).str.startswith("BLS (ics)").any():
            df = df[df["source"] != "BLS"]            # el calendario oficial descargado manda sobre el incorporado
        df = pd.concat([df, o], ignore_index=True)
    if extra is not None and len(extra):
        df = pd.concat([df, extra], ignore_index=True)
    return df.drop_duplicates(subset=["date", "name"]).sort_values("date").reset_index(drop=True)


def upcoming(events: pd.DataFrame, today: pd.Timestamp, days: int = 14) -> pd.DataFrame:
    t = pd.Timestamp(today).normalize()
    e = events[(events["date"] >= t) & (events["date"] <= t + pd.Timedelta(days=days))].copy()
    e["en_dias"] = (e["date"] - t).dt.days
    return e.reset_index(drop=True)


def event_risk(events: pd.DataFrame, today: pd.Timestamp, cfg: EventConfig) -> tuple[float, list[str]]:
    """Señal al cierre de `today` -> ejecución en la apertura de la rueda siguiente. Si hay un evento de
    alto impacto en las `window_bdays` ruedas desde esa ejecución, se reduce el riesgo (nunca se aumenta)."""
    if not cfg.enabled:
        return 1.0, []
    t = pd.Timestamp(today).normalize()
    first = t + pd.offsets.BDay(1)
    last = first + pd.offsets.BDay(max(cfg.window_bdays - 1, 0))
    hit = events[(events["date"] >= first) & (events["date"] <= last) & (events["impact"] == "alto")]
    flags = [f"{r.date:%d/%m} {r.time_et} ET — {r.name}" for r in hit.itertuples()]
    return (cfg.risk_mult if flags else 1.0), flags
