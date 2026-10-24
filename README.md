# Quant Scanner Pro v10 — investigación, seguridad y datos reales

Motor cuantitativo de señales (tendencia + pullback + meta-modelo calibrado), backtest walk-forward con costos,
gestión de clientes y alertas por Telegram.


> **Para ponerlo en marcha:** seguí `GUIA_GITHUB.md` (GitHub + Streamlit Cloud + alertas + backtests desde Actions).

## v10 — qué hay de nuevo (auditoría de punta a punta)
**Investigación (lo central):** `python -m quant_scanner.run_daily matrix` prueba 23 variantes de estrategia (tendencia, rupturas, pullbacks, reversión,
momentum, 52 semanas, cruces con ADX, squeeze) x 12 grupos (ETFs, sectores, cripto, mega caps, semis, energía, ADRs argentinos...) x 2 temporalidades
(diaria y semanal) y entrega un **mapa de operabilidad** con escala A-D:
1. Screening con datos de desarrollo (hasta `--dev-end`), con costos y **restando la deriva pasiva** del activo (estar invertido gana sin timing; además mitiga el sesgo de supervivencia).
2. Corrección por comparaciones múltiples (**Benjamini-Hochberg**) sobre TODAS las celdas, e IC por bootstrap por mes.
3. Las que pasan se miran UNA vez en el período reservado (holdout). 4. **Margen de costo**: costo por lado que aguanta cada celda (comparalo con tu broker / CEDEARs).
Probado con datos sintéticos: detecta un edge real y no inventa edge en ruido (ver `tests/test_matrix.py`).
**Biblioteca de indicadores** (`indicators.py`, 23 indicadores, todos con test de no look-ahead) y **familias de estrategia** (`specs.py`).

**Correcciones:** calibración del meta-modelo con CV temporal con purga y embargo (antes OOB, que filtraba información entre trades superpuestos); núcleo único de simulación de trades;
validación de toda la configuración; PRAGMA foreign_keys, transacciones explícitas y WAL.
**Seguridad** (ver `SECURITY.md`): cifrado en reposo, base con permisos 600, auditoría con cadena de hashes, redacción de secretos y datos personales, validación de entradas, contraseña con bloqueo.
**Datos reales gratis** (`providers.py`, `fred.py`): Tiingo, Alpaca (IEX), Finnhub, yfinance y FRED, con reconciliación entre fuentes y cuarentena ante discrepancias.
`python -m quant_scanner.run_daily check-data SPY QQQ AAPL` prueba tus claves. **Profesionalización:** CI (GitHub Actions), ruff, pip-audit, `pyproject.toml`, `.env.example`.

**Backtests:** no pude correrlos con datos reales (mi entorno no tiene internet). Orden recomendado en tu máquina: `check-data` -> `matrix` -> `backtest` -> 2-3 meses de paper trading.

## Estructura
- `quant_scanner/signals.py` indicadores, dos sistemas (tendencia y pullback) y simulación de trades sin look-ahead (señal al cierre, ejecución en la apertura siguiente, stops con gap, parcial opcional)
- `quant_scanner/model.py` meta-modelo único (pooled) con probabilidades calibradas
- `quant_scanner/backtest.py` backtest de cartera walk-forward, con costos y sin apalancamiento
- `quant_scanner/sizing.py` riesgo fijo por trade, cuarto de Kelly, topes duros y freno por drawdown (tolerancia configurable, 20% por defecto)
- `quant_scanner/events.py` calendario Fed/BLS verificado (+ actualización desde el .ics oficial del BLS) y reducción de riesgo por eventos
- `quant_scanner/news.py` noticias con credibilidad por fuente (niveles 1-4), corroboración y reglas de veto/aviso
- `quant_scanner/edgar.py` cliente de la SEC (8-K y Form 4); `altdata.py` insiders agrupados + `validate_signal`
- `quant_scanner/context.py` une eventos + noticias + balances en una decisión por señal
- `quant_scanner/research.py` comparador de variantes con Deflated Sharpe según el nº de pruebas
- `quant_scanner/metrics.py` CAGR, DD, Sharpe, PSR, Deflated Sharpe, Monte Carlo sobre R de trades
- `quant_scanner/store.py` clientes, posiciones y registro de alertas (SQLite)
- `quant_scanner/run_daily.py` job diario + CLI
- `app.py` interfaz Streamlit

## Uso
```bash
pip install -r requirements.txt
python tests/test_core.py && python tests/test_daily.py      # pruebas (datos sintéticos)
streamlit run app.py
python -m quant_scanner.run_daily backtest --capital 5000     # backtest con datos reales
python -m quant_scanner.run_daily add-client "Mateo" 5000 --risk 1.0
python -m quant_scanner.run_daily scan --dry-run              # sin enviar nada
```
Alertas: definí `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` como variables de entorno (o en `.streamlit/secrets.toml`),
y programá `python -m quant_scanner.run_daily scan` una vez por día después del cierre de EE.UU. en una máquina
con disco persistente (la base SQLite guarda clientes y alertas ya enviadas).
Si la app está desplegada, agregá `APP_PASSWORD` en los secrets.

## Qué corrige respecto a la v6
1. Las señales salen de la **última barra** (antes salían de la última entrada histórica por un `dropna()`).
2. El modelo ya no se predice sobre filas de su propio entrenamiento; entrena solo con trades cerrados antes de cada decisión.
3. Un modelo único con ~cientos de trades en vez de uno por activo con ~20; probabilidades calibradas (Platt).
4. Stop/salida coherentes con lo que el modelo etiqueta (R-múltiplo del trade real).
5. Topes por activo y por cartera, sin apalancamiento; costos en todo el backtest.
6. Retornos de riesgo calculados sobre retornos diarios reales; PSR/DSR con la fórmula correcta.
7. Alertas idempotentes, salidas y stops de posiciones abiertas, y modelo multicliente.

## Lo que NO está validado (v9)
- Las conexiones en vivo (noticias de yfinance, SEC EDGAR, .ics del BLS) las escribí según la documentación pero **no pude probarlas con internet**; los parsers y la lógica sí están probados con datos de ejemplo. Definí `SEC_USER_AGENT="Nombre Apellido tu@email.com"` (la SEC lo exige).
- El efecto de los eventos macro y de los balances en el backtest no está medido (falta historial verificado); por eso solo reducen riesgo.
- "Tiempo real": yfinance es una fuente no oficial y con demora. Para swing de 1-3 semanas alcanza con datos diarios al cierre; para alertas intradía de stops hace falta una API de broker.

## Lo que NO está validado (leer antes de usar plata)
- Las pruebas corrieron con datos **sintéticos**: verifican que la lógica es correcta, no que haya rentabilidad.
- El backtest con datos reales hay que correrlo en tu máquina (`backtest`); los costos por defecto son supuestos, ajustalos a tu broker.
- La interfaz de Streamlit se compiló pero no se ejecutó en mi entorno.
- Los parámetros por defecto NO están optimizados a propósito, para no sobreajustar. La tasa de acierto no es el objetivo: se optimiza expectativa y drawdown.
- Esto replica un sistema de tendencia clásico; no está verificado que sea la metodología de Iván Scherman.
- yfinance no es una fuente oficial: para operar con dinero de terceros conviene una API de datos contratada.

## Novedades de la v8
1. **Sistema de pullbacks** (RSI(2) bajo dentro de tendencia alcista, salida al recuperar la media de 5 ruedas, stop duro de 3 ATR y tope de 10 ruedas) además del breakout de tendencia ajustado a 1-3 semanas (máx. 10 / mín. 5 / 15 ruedas).
2. **Meta-modelo más rico:** fuerza relativa vs SPY, RSI(2), IBS, distancia a la SMA50 y percentil del VIX; payoff propio de cada sistema; calibración de Platt con predicciones fuera de bolsa (5-7x más rápido).
3. **Curva de edge y calibración (Brier):** muestran, fuera de muestra, cuánto mejora realmente el acierto y el R medio al exigir más edge, y a qué costo en cantidad de trades.
4. **Freno por drawdown:** el riesgo por trade baja al acercarse a tu límite y se frenan las entradas desde el 75% de la tolerancia.
5. **Comparador de variantes:** cada configuración probada cuenta como una prueba para el Deflated Sharpe. Elegí por R medio, profit factor y drawdown, no por tasa de acierto.
6. **Alertas:** entradas con regla de salida, stops, salida por señal y salida por tiempo, por sistema.

## Novedades de la v9: actualidad y contexto
1. **Datos al día:** se descarta la barra del día en curso (señales sobre una barra incompleta son falsas), caché de 1 hora y chequeo de frescura contra SPY: un activo atrasado no emite entradas. El escaneo se programa DESPUÉS del cierre de EE.UU. (≥ 16:30 ET).
2. **Calendario de mercado:** FOMC 2026-2027, empleo (NFP), CPI, PPI, minutas y Beige Book. Si un evento de alto impacto cae en la apertura de mañana o la siguiente, el riesgo del trade baja a la mitad (regla de prudencia: NO validada en backtest). `refresh-events` baja el calendario oficial del BLS.
3. **Noticias con sustento:** cada nota recibe nivel de fuente (1 oficial: SEC/Fed/BLS; 2 agencias; 3 medios; 4 desconocido), corroboración entre fuentes independientes y penalización por lenguaje de rumor. Una noticia **nunca genera una compra**: solo veta un hecho grave confirmado (p. ej. 8-K de quiebra, deslistado o restatement) o avisa. Los rumores sin confirmar se muestran pero no pesan. Si falla la consulta de noticias, la alerta lo dice (un error no es "sin noticias").
4. **Datos alternativos con rigor:** compras agrupadas de insiders (Form 4). Solo "gradúan" al modelo si `validate_signal` muestra muestra suficiente, diferencia de R con intervalo de confianza > 0 y efecto estable en ambas mitades. Si no, quedan como información.
5. **Balances:** si hay un balance estimado dentro del horizonte de la operación, el riesgo baja (fecha de yfinance, no oficial).

## Pendiente
- Filtro de balances (earnings): requiere una fuente de datos de calendario confiable.
- Freno por drawdown en el job diario por cliente (hoy aplica en el backtest; en vivo requiere registrar el P&L de cada cliente).
- Pestañas de macro, fundamentales y renta fija de la v6 (no portadas).
