"""Validación de entradas externas (CLI, formularios, archivos). Rechazar temprano, con mensajes claros."""
from __future__ import annotations

import math
import re

_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$|^\^[A-Z0-9]{1,8}$")


def valid_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not _TICKER.match(t):
        raise ValueError(f"ticker inválido: {t!r}")
    return t


def valid_amount(x, name: str = "monto", lo: float = 0.0, hi: float = 1e9) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} inválido: {x!r}") from None
    if not math.isfinite(v) or not (lo < v <= hi):
        raise ValueError(f"{name} fuera de rango ({lo}, {hi}]: {v}")
    return v


def valid_name(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip()
    if not s or len(s) > maxlen or any(ord(ch) < 32 for ch in s):
        raise ValueError("nombre inválido")
    return s


def valid_chat_id(s: str | None) -> str | None:
    if s in (None, ""):
        return None
    if not re.fullmatch(r"-?\d{3,20}", str(s).strip()):
        raise ValueError("chat id de Telegram inválido (debe ser numérico)")
    return str(s).strip()
