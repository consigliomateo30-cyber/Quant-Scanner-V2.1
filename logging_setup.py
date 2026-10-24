"""Logging con rotación y redacción automática de secretos (tokens de Telegram, claves de API, correos)."""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler

_PATTERNS = [
    (re.compile(r"bot\d{6,}:[A-Za-z0-9_\-]{20,}"), "bot***REDACTED***"),
    (re.compile(r"(?i)(token|api[_-]?key|secret|password|authorization)(=|:)\s*[^\s&\"']+"), r"\1\2***REDACTED***"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "***EMAIL***"),
]


def redact(text: str) -> str:
    for pat, rep in _PATTERNS:
        text = pat.sub(rep, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        return True


def get_logger(name: str = "quant_scanner") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(os.environ.get("QS_LOG_LEVEL", "INFO"))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    path = os.environ.get("QS_LOG_FILE")
    if path:
        fh = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    log.addFilter(RedactingFilter())
    return log
