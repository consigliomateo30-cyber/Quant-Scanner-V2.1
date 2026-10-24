"""Cifrado de campos sensibles en reposo (Fernet = AES-128-CBC + HMAC, de la librería `cryptography`).
Definí QS_ENCRYPTION_KEY (generala con `python -m quant_scanner.crypto`). Sin clave, los datos quedan en claro
y la app lo advierte. La clave NUNCA va en el repositorio: variable de entorno o gestor de secretos."""
from __future__ import annotations

import hashlib
import hmac
import os

PREFIX = "enc:"


def _fernet():
    key = os.environ.get("QS_ENCRYPTION_KEY")
    if not key:
        return None
    from cryptography.fernet import Fernet
    return Fernet(key.encode())


def enabled() -> bool:
    return bool(os.environ.get("QS_ENCRYPTION_KEY"))


def encrypt(text: str | None) -> str | None:
    f = _fernet()
    if text is None or f is None:
        return text
    return PREFIX + f.encrypt(text.encode()).decode()


def decrypt(value: str | None) -> str | None:
    if value is None or not str(value).startswith(PREFIX):
        return value
    f = _fernet()
    if f is None:
        raise RuntimeError("Hay datos cifrados pero falta QS_ENCRYPTION_KEY.")
    return f.decrypt(value[len(PREFIX):].encode()).decode()


def blind_index(text: str) -> str:
    """Huella determinística (HMAC) para buscar/forzar unicidad sin guardar el nombre en claro."""
    key = (os.environ.get("QS_ENCRYPTION_KEY") or "sin-clave").encode()
    return hmac.new(key, text.strip().lower().encode(), hashlib.sha256).hexdigest()


if __name__ == "__main__":
    from cryptography.fernet import Fernet
    print(Fernet.generate_key().decode())
