"""Ejecuta un comando del CLI desde GitHub Actions (workflow 'manage'). Los argumentos llegan por variables de
entorno y se parten con shlex: nada se interpreta en el shell (sin inyección de comandos)."""
import os
import shlex
import sys

from quant_scanner import run_daily

ALLOWED = {"add-client", "list-clients", "fill", "close", "events", "check-data"}

if __name__ == "__main__":
    cmd = os.environ.get("QS_CMD", "").strip()
    if cmd not in ALLOWED:
        sys.exit(f"comando no permitido: {cmd!r}")
    sys.exit(run_daily.main([cmd, *shlex.split(os.environ.get("QS_ARGS", ""))]))
