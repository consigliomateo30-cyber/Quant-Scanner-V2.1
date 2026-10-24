# Seguridad y manejo de datos

**Principio:** el software maneja datos de clientes y dinero. Se diseñó para exponer lo mínimo, detectar manipulaciones y no filtrar secretos.

| Riesgo | Control implementado |
|---|---|
| Datos de clientes en reposo | Cifrado Fernet (AES + HMAC) de nombre y chat id si definís `QS_ENCRYPTION_KEY`; base con permisos `600`; `secure_delete` |
| Duplicados / búsquedas sin exponer nombres | Índice ciego (HMAC) para unicidad |
| Integridad referencial | `PRAGMA foreign_keys=ON`; transacciones explícitas (`BEGIN IMMEDIATE`) |
| Manipulación del historial de alertas | Registro de auditoría con **cadena de hashes** (SHA-256); `verify_audit_chain()` detecta cualquier edición |
| Datos personales en logs | El registro de auditoría redacta el nombre del cliente; el logger redacta tokens, claves y correos |
| Secretos | Solo variables de entorno / `st.secrets`; nunca en el código ni en la interfaz; `.gitignore` y `.env.example` |
| Claves de API | Viajan en cabeceras (no en la URL); los errores no las incluyen |
| Entradas externas | Validación estricta de tickers, montos, nombres y chat ids; consultas SQL parametrizadas; HTML escapado |
| Acceso a la app | Contraseña (`APP_PASSWORD`) con comparación en tiempo constante y bloqueo tras 5 intentos |
| Datos de mercado erróneos | Reconciliación entre fuentes (>0,5% de diferencia = cuarentena) y chequeo de frescura |
| Dependencias | Versiones acotadas; correr `pip-audit` en CI |

**Lo que NO cubre (hacelo vos):** respaldo cifrado de la base, rotación de claves, 2FA en cuentas de brokers/APIs, un firewall o VPN si desplegás en un servidor, y el cumplimiento regulatorio (CNV/SEC) si gestionás dinero de terceros. Si desplegás en Streamlit Cloud, la app es pública por defecto: activá la autenticación de la plataforma además de `APP_PASSWORD`.
