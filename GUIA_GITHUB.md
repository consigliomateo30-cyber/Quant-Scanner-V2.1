# Guía paso a paso: poner Quant Scanner en funcionamiento con GitHub

Resultado final: (1) el código en un repositorio **privado** de GitHub, (2) la app web en Streamlit Cloud como la que ya tenías
(`algo.streamlit.app`), (3) alertas diarias por Telegram y (4) backtests con datos reales corridos desde GitHub.

## Parte A — Subir el código a GitHub
1. Creá una cuenta en github.com y activá la verificación en dos pasos (Settings > Password and authentication).
2. **New repository** → nombre `quant-scanner` → marcá **Private** → NO agregues README ni .gitignore → *Create repository*.
3. Descomprimí `quant_scanner_pro_v10.zip` en tu computadora. Subilo de una de estas dos formas:
   - **Navegador:** en el repositorio vacío, *uploading an existing file* → arrastrá TODO el contenido de la carpeta (incluidas las
     carpetas ocultas `.github`, `.streamlit` y el archivo `.gitignore`; en Mac mostrá ocultos con Cmd+Shift+. y en Windows activá
     "Elementos ocultos") → *Commit changes*. Si GitHub no te deja subir carpetas ocultas, usá la opción de terminal.
   - **Terminal** (más confiable; instalá Git desde git-scm.com). Dentro de la carpeta descomprimida:
     ```
     git init
     git add .
     git commit -m "Quant Scanner v10"
     git branch -M main
     git remote add origin https://github.com/TU_USUARIO/quant-scanner.git
     git push -u origin main
     ```
4. Abrí la pestaña **Actions**. Debe aparecer el workflow **ci** corriendo las pruebas. Tilde verde = todo bien.
5. Revisá que NO se hayan subido archivos `.env`, `*.db` ni `secrets.toml` (el `.gitignore` los excluye).

## Parte B — Claves (Secrets)
*Settings > Secrets and variables > Actions > New repository secret.* Cargá solo las que vayas a usar:

| Secret | Cómo obtenerla |
|---|---|
| `QS_ENCRYPTION_KEY` | Generala en colab.research.google.com con `from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())`. **Guardala también en tu gestor de contraseñas: si la perdés, los datos cifrados no se recuperan.** |
| `TELEGRAM_BOT_TOKEN` | En Telegram hablá con **@BotFather** → `/newbot` → copiá el token. |
| `TELEGRAM_CHAT_ID` | Escribile un mensaje a tu bot y abrí en el navegador `https://api.telegram.org/bot<TOKEN>/getUpdates`: buscá `"chat":{"id":NÚMERO`. No compartas esa URL (lleva el token). |
| `SEC_USER_AGENT` | Tu nombre y correo, por ejemplo `Mateo Consigli mateo@correo.com` (la SEC lo exige). |
| `TIINGO_TOKEN` | Cuenta gratis en tiingo.com. Sirve de respaldo si Yahoo bloquea a GitHub y para verificar precios. |
| `ALPACA_KEY_ID` y `ALPACA_SECRET_KEY` | Cuenta gratis en alpaca.markets (cuenta *paper*) → claves de API. Datos IEX (volumen parcial). |
| `FINNHUB_TOKEN`, `FRED_API_KEY` | Opcionales (cotización actual y macro). |

Los límites de los planes gratis cambian: confirmalos en cada sitio.

## Parte C — La app web (como la de `...streamlit.app`)
1. Entrá a **share.streamlit.io** → *Continue with GitHub* y autorizá el acceso a tu repositorio privado.
2. **Create app** → repositorio `quant-scanner`, rama `main`, archivo principal `app.py`.
3. **Advanced settings**: Python 3.12 y en **Secrets** pegá el contenido de `.streamlit/secrets.toml.example` con tus valores
   (`APP_PASSWORD` es obligatoria si va a mostrar datos de clientes). → **Deploy**.
4. En **Share** de la app restringí quién puede verla (la app es pública por defecto; confirmá qué permite tu plan).
5. Importante: el disco de Streamlit Cloud es **efímero** (se borra al reiniciar). Usá la app para escanear, investigar y backtestear;
   los clientes y las alertas viven en la Parte D.

## Parte D — Alertas diarias automáticas (GitHub Actions)
1. **Actions > manage > Run workflow**: command `add-client`, args `"Tu Nombre" 5000 --risk 1.0 --fractional --chat TU_CHAT_ID`.
   Probá `list-clients` para verificar.
2. **Actions > daily-scan > Run workflow** con *dry_run* marcado: revisá el log. Después corrélo sin *dry_run* y comprobá que llegue
   el mensaje de Telegram.
3. Desde ahí corre solo de lunes a viernes a las 21:30 UTC (después del cierre de EE.UU.).
4. Cuando compres lo que te alertó: **manage > fill**, args `1 NVDA 3 --price 180.5 --stop 170.2 --system trend` (el 1 es el id del cliente).
   Cuando vendas: **manage > close**, args `ID_POSICIÓN --price 190`.
5. El estado (clientes, posiciones, alertas enviadas) se guarda en la caché de Actions: se renueva con cada corrida y GitHub la borra si pasan
   ~7 días sin usarse. Si GitHub desactiva el cron por inactividad, reactivalo en Actions. Para algo permanente conviene una PC o servidor propio con `cron`.
6. Mantené el repositorio **privado**: los registros de Actions muestran capital y tickers.

## Parte E — Backtests reales (sin computadora)
**Actions > research > Run workflow.** Corre la matriz de investigación y el backtest de cartera con datos reales y deja los resultados en el
*resumen* de la corrida y como archivos descargables (`reportes`). Pasale `matriz.md` y el resumen del backtest a quien te ayude a interpretarlos.
Regla de oro: solo se opera lo que aparezca como nivel A o B y haya pasado 2 a 3 meses de paper trading.

## Si algo falla
- **Actions en rojo:** abrí la corrida, mirá el paso que falló y copiá el mensaje.
- **No llegan mensajes:** probá `manage > check-data` y revisá `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`. Escribile primero a tu bot.
- **Yahoo bloquea a GitHub:** cargá `TIINGO_TOKEN` (respaldo automático, con tope de ~45 activos por hora en el plan gratis).
