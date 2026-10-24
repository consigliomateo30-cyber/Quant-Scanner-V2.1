"""Parámetros centrales. Todo lo que afecta resultados vive acá, en un solo lugar.
Preset por defecto: SWING de 1 a 3 semanas, tope de drawdown 20%. Los valores NO están optimizados
(a propósito): se comparan variantes con `research.compare_variants` y se corrigen por nº de pruebas."""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "VT", "IBIT", "BTC-USD", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA",
    "META", "AVGO", "QCOM", "MU", "TSM", "ASML", "BRK-B", "MELI", "NU",
    "YPF", "VIST", "PAMP", "TGS", "OKLO", "IREN",
]

MARKET_PROXY = "SPY"
VIX_TICKER = "^VIX"


@dataclass(frozen=True)
class StrategyConfig:
    # --- Sistema 1: breakout de tendencia (ventanas cortas para swing de 1-3 semanas)
    entry_window: int = 10
    exit_window: int = 5
    max_hold: int = 15              # ~3 semanas de rueda
    stop_atr: float = 2.0
    # --- Sistema 2: pullback en tendencia (alta tasa de acierto, payoff menor)
    pb_enabled: bool = True
    pb_rsi_entry: float = 10.0      # RSI(2) por debajo de este nivel
    pb_trend_sma: int = 200         # solo comprar si cierra por encima
    pb_exit_sma: int = 5            # sale cuando cierra por encima de esta media
    pb_stop_atr: float = 3.0        # stop duro (sin stop, la cola de pérdidas es el riesgo de este sistema)
    pb_max_hold: int = 10
    # --- Comunes
    atr_window: int = 14
    cooldown_after_stop: int = 5    # barras sin re-entrar tras un stop
    partial_r: float | None = None  # ej. 1.5 -> vende parte al llegar a +1.5R (solo sistema de tendencia)
    partial_frac: float = 0.5
    use_regime: bool = True         # solo comprar si SPY > SMA(regime_sma)
    regime_sma: int = 200

    def __post_init__(self):
        for name in ("entry_window", "exit_window", "max_hold", "pb_trend_sma", "pb_exit_sma", "pb_max_hold", "atr_window", "regime_sma"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} debe ser >= 1")
        if not (0.5 <= self.stop_atr <= 10 and 0.5 <= self.pb_stop_atr <= 10):
            raise ValueError("el stop en ATR debe estar entre 0.5 y 10")
        if not (0 < self.pb_rsi_entry < 50):
            raise ValueError("pb_rsi_entry debe estar entre 0 y 50")
        if self.partial_r is not None and not (0.25 <= self.partial_r <= 10 and 0 < self.partial_frac < 1):
            raise ValueError("parcial inválido")

    def stop_for(self, system: str) -> float:
        return self.pb_stop_atr if system == "pullback" else self.stop_atr

    def hold_for(self, system: str) -> int:
        return self.pb_max_hold if system == "pullback" else self.max_hold


@dataclass(frozen=True)
class CostConfig:
    """Costos por lado (compra o venta). AJUSTAR al arancel real de tu broker."""
    commission_pct: float = 0.15
    slippage_pct: float = 0.10
    min_commission_usd: float = 0.0

    def __post_init__(self):
        if not (0 <= self.commission_pct <= 5 and 0 <= self.slippage_pct <= 5):
            raise ValueError("comisión y slippage deben estar entre 0 y 5 (% por lado)")

    @property
    def per_side(self) -> float:
        return (self.commission_pct + self.slippage_pct) / 100.0


@dataclass(frozen=True)
class RiskConfig:
    risk_max_pct: float = 1.0       # riesgo máximo por trade (% del capital)
    risk_flat_pct: float = 0.75     # riesgo por trade cuando NO se usa el meta-modelo
    kelly_mult: float = 0.25
    min_edge: float = 0.02
    max_pos_pct: float = 25.0
    max_positions: int = 8
    allow_fractional: bool = False
    min_trade_usd: float = 20.0
    max_drawdown_pct: float = 20.0  # tu tolerancia; el freno de riesgo se calcula a partir de este valor
    dd_throttle: bool = True

    def __post_init__(self):
        if not (0 < self.risk_max_pct <= 5 and 0 < self.risk_flat_pct <= 5):
            raise ValueError("el riesgo por trade debe estar entre 0 y 5 %")
        if not (0 < self.max_pos_pct <= 100 and 1 <= self.max_positions <= 100):
            raise ValueError("topes de posición inválidos")
        if not (5 <= self.max_drawdown_pct <= 50):
            raise ValueError("max_drawdown_pct debe estar entre 5 y 50")
        if not (0 < self.kelly_mult <= 1):
            raise ValueError("kelly_mult debe estar entre 0 y 1")


@dataclass(frozen=True)
class ModelConfig:
    min_train_trades: int = 80
    retrain_every_days: int = 126   # ~6 meses de ruedas
    n_estimators: int = 150
    cv_splits: int = 4              # bloques temporales para calibrar (CV con purga y embargo)
    cv_trees: int = 80
    embargo_days: int = 5
    max_depth: int = 3
    min_samples_leaf: int = 15
    random_state: int = 42


@dataclass(frozen=True)
class EventConfig:
    """Riesgo por eventos macro. NO está validado en backtest (falta historial verificado): es una
    regla de prudencia que reduce el tamaño, nunca aumenta ni genera señales."""
    enabled: bool = True
    risk_mult: float = 0.5          # multiplicador de riesgo si hay un evento de alto impacto cerca de la ejecución
    window_bdays: int = 2           # ruedas desde la ejecución (apertura de mañana) que se consideran expuestas
    earnings_risk_mult: float = 0.5  # si hay balance dentro del horizonte de la operación


FEATURE_COLS = [
    "vol_20d", "atr_norm", "dist_donchian", "vol_ratio", "rsi14", "ret_5d", "ret_20d",
    "dist_sma200", "mkt_regime",
    "rs_20d", "rsi2", "ibs", "dist_sma50", "vix_rank", "is_pullback",
]

SYSTEMS = ("trend", "pullback")


@dataclass(frozen=True)
class AppConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    events: EventConfig = field(default_factory=EventConfig)

    def systems(self) -> tuple[str, ...]:
        return SYSTEMS if self.strategy.pb_enabled else ("trend",)
