"""Universos de investigación por clase de activo / industria y perfiles de costo.
OJO sesgo de supervivencia: las listas de acciones son empresas que HOY existen y son grandes; eso infla
cualquier backtest de acciones. Los ETFs tienen mucho menos sesgo: por eso se prueban primero."""
from __future__ import annotations

GROUPS: dict[str, list[str]] = {
    "indices_us": ["SPY", "QQQ", "IWM", "DIA"],
    "sectores_us": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"],
    "internacional": ["EFA", "EEM", "EWZ", "EWJ", "FXI"],
    "bonos": ["TLT", "IEF", "HYG", "LQD"],
    "materias_primas": ["GLD", "SLV", "USO", "DBC"],
    "cripto": ["BTC-USD", "ETH-USD"],
    "mega_caps": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V", "WMT"],
    "semiconductores": ["NVDA", "AVGO", "AMD", "QCOM", "MU", "TSM", "ASML", "AMAT"],
    "energia": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "financieras": ["JPM", "BAC", "GS", "MS", "WFC"],
    "salud_consumo": ["UNH", "JNJ", "LLY", "PG", "KO", "COST", "MCD"],
    "argentina_adr": ["YPF", "GGAL", "BMA", "PAM", "TGS", "VIST", "CEPU", "MELI"],
}

# costo por lado (comisión + spread + slippage) en %, SUPUESTOS para el screening base
GROUP_COST_SIDE_PCT: dict[str, float] = {
    "indices_us": 0.05, "sectores_us": 0.05, "internacional": 0.08, "bonos": 0.05, "materias_primas": 0.08,
    "cripto": 0.20, "mega_caps": 0.08, "semiconductores": 0.10, "energia": 0.10, "financieras": 0.08,
    "salud_consumo": 0.08, "argentina_adr": 0.20,
}
# escenarios para comparar contra el break-even de cada celda (por lado, %). A CONFIRMAR con tu broker.
COST_SCENARIOS_SIDE_PCT: dict[str, float] = {
    "ETF/acción USA, broker barato": 0.08,
    "acción USA, broker local": 0.25,
    "CEDEAR liquido (supuesto)": 0.50,
    "CEDEAR menos liquido (supuesto)": 1.00,
}
MARKET_PROXY = "SPY"


def all_tickers(groups: dict[str, list[str]] | None = None) -> list[str]:
    g = groups or GROUPS
    return sorted({t for v in g.values() for t in v} | {MARKET_PROXY})
