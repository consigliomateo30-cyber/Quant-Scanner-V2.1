"""Macro desde FRED (Reserva Federal de St. Louis; clave gratis: FRED_API_KEY). Se usa como CONTEXTO; no entra
al modelo hasta pasar la validación (ver altdata.validate_signal)."""
from __future__ import annotations

import os

import pandas as pd

from .providers import ProviderError, http_get

SERIES = {"DFF": "Tasa de fondos federales (efectiva)", "DGS10": "Bono EE.UU. 10 años", "DGS2": "Bono EE.UU. 2 años",
          "T10Y2Y": "Pendiente de curva 10Y-2Y", "VIXCLS": "VIX (cierre)", "BAMLH0A0HYM2": "Spread high yield (OAS)"}


def parse_observations(js: dict) -> pd.Series:
    rows = (js or {}).get("observations", [])
    s = pd.Series({pd.Timestamp(r["date"]): pd.to_numeric(r["value"], errors="coerce") for r in rows if r.get("value") not in (None, ".")})
    return s.dropna().sort_index()


def fetch_series(series_id: str, start: str = "2024-01-01") -> pd.Series:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ProviderError("falta FRED_API_KEY")
    js = http_get("https://api.stlouisfed.org/fred/series/observations",
                  {"series_id": series_id, "api_key": key, "file_type": "json", "observation_start": start})
    return parse_observations(js)


def macro_snapshot(series: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for sid, label in (series or SERIES).items():
        try:
            s = fetch_series(sid)
            if len(s):
                prev = s[s.index <= s.index[-1] - pd.Timedelta(days=30)]
                rows.append({"serie": sid, "descripcion": label, "ultimo": float(s.iloc[-1]), "fecha": s.index[-1],
                              "cambio_30d": float(s.iloc[-1] - prev.iloc[-1]) if len(prev) else None})
        except ProviderError as e:
            rows.append({"serie": sid, "descripcion": label, "ultimo": None, "fecha": None, "cambio_30d": None, "error": str(e)})
    return pd.DataFrame(rows)
