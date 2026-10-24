"""Persistencia en SQLite: clientes/mandatos, posiciones y registro de alertas (auditoría).
La clave de alerta evita mandar la misma señal dos veces."""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from . import crypto
from .validation import valid_amount, valid_chat_id, valid_name, valid_ticker

DB_PATH = os.environ.get("QS_DB", "quant_scanner.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
  capital REAL NOT NULL, risk_max_pct REAL NOT NULL DEFAULT 1.0, max_positions INTEGER NOT NULL DEFAULT 8,
  max_pos_pct REAL NOT NULL DEFAULT 25.0, allow_fractional INTEGER NOT NULL DEFAULT 0,
  telegram_chat_id TEXT, active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, client_id INTEGER NOT NULL REFERENCES clients(id),
  ticker TEXT NOT NULL, units REAL NOT NULL, entry_date TEXT NOT NULL, entry_price REAL NOT NULL,
  stop REAL NOT NULL, status TEXT NOT NULL DEFAULT 'open', exit_date TEXT, exit_price REAL,
  system TEXT NOT NULL DEFAULT 'trend');
CREATE TABLE IF NOT EXISTS alerts_log(
  key TEXT PRIMARY KEY, client_id INTEGER, kind TEXT, ticker TEXT, message TEXT, sent INTEGER, created_at TEXT);
"""


def _migrate(c: sqlite3.Connection) -> None:
    """Migraciones idempotentes para bases creadas con versiones anteriores."""
    def cols(table):
        return {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
    if "system" not in cols("positions"):
        c.execute("ALTER TABLE positions ADD COLUMN system TEXT NOT NULL DEFAULT 'trend'")
    if "name_key" not in cols("clients"):
        c.execute("ALTER TABLE clients ADD COLUMN name_key TEXT")
    for col in ("prev_hash", "row_hash"):
        if col not in cols("alerts_log"):
            c.execute(f"ALTER TABLE alerts_log ADD COLUMN {col} TEXT")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_clients_name_key ON clients(name_key)")


@contextmanager
def conn():
    path = os.environ.get("QS_DB") or DB_PATH       # se lee en cada uso: sirve con secretos cargados después del import
    new_file = not os.path.exists(path)
    c = sqlite3.connect(path, timeout=30, isolation_level=None)     # transacciones explícitas
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA secure_delete=ON")
    c.executescript(SCHEMA)
    _migrate(c)
    if new_file:
        try:
            os.chmod(path, 0o600)          # solo el dueño puede leer la base de clientes
        except OSError:
            pass
    c.execute("BEGIN IMMEDIATE")
    try:
        yield c
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    finally:
        c.close()


def add_client(name, capital, risk_max_pct=1.0, max_positions=8, max_pos_pct=25.0, allow_fractional=False, telegram_chat_id=None):
    name = valid_name(name)
    capital = valid_amount(capital, "capital", 0.0, 1e9)
    risk_max_pct = valid_amount(risk_max_pct, "riesgo máx. %", 0.0, 5.0)
    max_pos_pct = valid_amount(max_pos_pct, "máx. % por activo", 0.0, 100.0)
    max_positions = int(valid_amount(max_positions, "máx. posiciones", 0.0, 100.0))
    chat = valid_chat_id(telegram_chat_id)
    with conn() as c:
        c.execute("INSERT INTO clients(name,name_key,capital,risk_max_pct,max_positions,max_pos_pct,allow_fractional,telegram_chat_id,created_at) "
                  "VALUES(?,?,?,?,?,?,?,?,?)",
                  (crypto.encrypt(name), crypto.blind_index(name), capital, risk_max_pct, max_positions, max_pos_pct,
                   int(allow_fractional), crypto.encrypt(chat), datetime.now().isoformat()))


def list_clients(active_only=True) -> list[dict]:
    with conn() as c:
        q = "SELECT * FROM clients" + (" WHERE active=1" if active_only else "")
        rows = [dict(r) for r in c.execute(q)]
    for r in rows:
        r["name"], r["telegram_chat_id"] = crypto.decrypt(r["name"]), crypto.decrypt(r["telegram_chat_id"])
        r.pop("name_key", None)
    return rows


def open_positions(client_id: int) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM positions WHERE client_id=? AND status='open'", (client_id,))]


def record_fill(client_id, ticker, units, entry_date, entry_price, stop, system="trend"):
    ticker = valid_ticker(ticker)
    units, entry_price, stop = (valid_amount(units, "unidades"), valid_amount(entry_price, "precio"), valid_amount(stop, "stop"))
    if stop >= entry_price:
        raise ValueError("el stop debe estar por debajo del precio de compra")
    if system not in ("trend", "pullback"):
        raise ValueError("sistema inválido")
    with conn() as c:
        c.execute("INSERT INTO positions(client_id,ticker,units,entry_date,entry_price,stop,system) VALUES(?,?,?,?,?,?,?)",
                  (client_id, ticker, units, str(entry_date), entry_price, stop, system))


def close_position(position_id, exit_date, exit_price):
    with conn() as c:
        c.execute("UPDATE positions SET status='closed', exit_date=?, exit_price=? WHERE id=?", (exit_date, exit_price, position_id))


def _row_hash(prev: str, key, client_id, kind, ticker, message, created_at) -> str:
    raw = "\x1f".join(str(x) for x in (prev, key, client_id, kind, ticker, message, created_at))
    return hashlib.sha256(raw.encode()).hexdigest()


def claim_alert(key, client_id, kind, ticker, message) -> bool:
    """True si la alerta es nueva (y la registra con cadena de hashes). False si ya se envió antes.
    El nombre del cliente se redacta: el registro de auditoría no guarda datos personales."""
    message = re.sub(r"^\[[^\]]*\]", "[cliente]", message)
    with conn() as c:
        if c.execute("SELECT 1 FROM alerts_log WHERE key=?", (key,)).fetchone():
            return False
        last = c.execute("SELECT row_hash FROM alerts_log WHERE row_hash IS NOT NULL ORDER BY rowid DESC LIMIT 1").fetchone()
        prev = last["row_hash"] if last else "GENESIS"
        now = datetime.now().isoformat()
        c.execute("INSERT INTO alerts_log(key,client_id,kind,ticker,message,sent,created_at,prev_hash,row_hash) VALUES(?,?,?,?,?,0,?,?,?)",
                  (key, client_id, kind, ticker, message, now, prev, _row_hash(prev, key, client_id, kind, ticker, message, now)))
        return True


def verify_audit_chain() -> tuple[bool, int | None]:
    """Recalcula la cadena. Devuelve (ok, rowid de la primera fila alterada o None)."""
    with conn() as c:
        prev = "GENESIS"
        for r in c.execute("SELECT rowid AS rid, * FROM alerts_log WHERE row_hash IS NOT NULL ORDER BY rowid"):
            if r["prev_hash"] != prev or r["row_hash"] != _row_hash(prev, r["key"], r["client_id"], r["kind"], r["ticker"], r["message"], r["created_at"]):
                return False, r["rid"]
            prev = r["row_hash"]
    return True, None


def mark_sent(key):
    with conn() as c:
        c.execute("UPDATE alerts_log SET sent=1 WHERE key=?", (key,))
