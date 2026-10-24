"""Alertas por Telegram. Las credenciales salen de variables de entorno o st.secrets,
NUNCA de un campo de texto en la interfaz."""
from __future__ import annotations

import os

import requests

from .logging_setup import get_logger

log = get_logger()

MAX_LEN = 3800


def get_credentials() -> tuple[str | None, str | None]:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        return token, chat
    try:  # dentro de Streamlit
        import streamlit as st
        return st.secrets.get("TELEGRAM_BOT_TOKEN"), st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        return None, None


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None, chat_override: str | None = None) -> bool:
    token, default_chat = (token, chat_id) if token and chat_id else get_credentials()
    chat = chat_override or default_chat
    if not token or not chat:
        return False
    ok = True
    for i in range(0, len(text), MAX_LEN):
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": text[i:i + MAX_LEN]}, timeout=15)
            ok = ok and r.ok
        except requests.RequestException as e:
            log.warning("Telegram: fallo de red (%s)", type(e).__name__)   # nunca se loguea la URL (lleva el token)
            ok = False
    return ok
