"""Tamaño de posición: riesgo fijo por trade, Kelly fraccionado como escala y topes duros."""
from __future__ import annotations

import math

from .config import RiskConfig


def kelly_fraction(p: float, payoff: float) -> float:
    """Kelly para apuestas medidas en R: f* = p - (1-p)/b."""
    return p - (1.0 - p) / payoff if payoff > 0 else -1.0


def risk_pct_for(p: float | None, payoff: float | None, rc: RiskConfig) -> float:
    """% del capital a arriesgar en el trade. Sin modelo -> riesgo plano. Con modelo ->
    cuarto de Kelly con techo; si el edge estimado es chico o negativo, no se opera."""
    if p is None or payoff is None:
        return rc.risk_flat_pct
    f = kelly_fraction(p, payoff)
    if f < rc.min_edge:
        return 0.0
    return max(0.0, min(rc.risk_max_pct, rc.kelly_mult * f * 100.0))


def position_units(capital: float, entry: float, stop: float, risk_pct: float, rc: RiskConfig,
                   cash_available: float | None = None, cost_per_side: float = 0.0) -> float:
    """Unidades a comprar. Respeta: riesgo por trade, % máximo por activo, caja disponible
    (sin apalancamiento) y monto mínimo. Devuelve 0 si no alcanza."""
    if not all(math.isfinite(x) for x in (capital, entry, stop, risk_pct, cost_per_side)):
        return 0.0
    if risk_pct <= 0 or entry <= 0 or stop >= entry or capital <= 0:
        return 0.0
    eff_entry = entry * (1.0 + cost_per_side)
    per_unit_risk = eff_entry - stop
    if per_unit_risk <= 0:
        return 0.0
    units = (capital * risk_pct / 100.0) / per_unit_risk
    units = min(units, capital * rc.max_pos_pct / 100.0 / eff_entry)
    if cash_available is not None:
        units = min(units, max(cash_available, 0.0) / eff_entry)
    units = units if rc.allow_fractional else math.floor(units)
    return units if units * entry >= rc.min_trade_usd else 0.0


def dd_scale(drawdown: float, rc: RiskConfig) -> float:
    """Multiplicador de riesgo según el drawdown actual (0-1). Con tolerancia L:
    hasta 40% de L -> riesgo completo; entre 40% y 75% de L baja linealmente a 25%;
    desde 75% de L no se abren posiciones nuevas hasta recuperar."""
    if not rc.dd_throttle:
        return 1.0
    limit = rc.max_drawdown_pct / 100.0
    lo, hi = 0.40 * limit, 0.75 * limit
    if drawdown <= lo:
        return 1.0
    if drawdown >= hi:
        return 0.0
    return 1.0 - 0.75 * (drawdown - lo) / (hi - lo)
