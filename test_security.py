import os, sys, sqlite3, stat, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
d = tempfile.mkdtemp()
os.environ["QS_DB"] = os.path.join(d, "sec.db")
from cryptography.fernet import Fernet
os.environ["QS_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
from quant_scanner import store, crypto
from quant_scanner.validation import valid_ticker, valid_amount, valid_chat_id, valid_name
from quant_scanner.logging_setup import redact

def test_validation():
    assert valid_ticker(" brk-b ") == "BRK-B" and valid_ticker("^vix") == "^VIX"
    for bad in ("", "A B", "DROP TABLE;", "x" * 20, "../etc"):
        try: valid_ticker(bad)
        except ValueError: continue
        raise AssertionError(bad)
    for bad in (-1, 0, "abc", float("nan"), 1e12):
        try: valid_amount(bad)
        except ValueError: continue
        raise AssertionError(bad)
    assert valid_chat_id("123456789") == "123456789" and valid_chat_id("") is None
    try: valid_chat_id("12; DROP"); raise AssertionError()
    except ValueError: pass
    try: valid_name("a\\x00b".replace("\\x00", "\x00")); raise AssertionError()
    except ValueError: pass

def test_store_hardening():
    store.add_client("Cliente Secreto S.A.", 5000, 1.0, 5, 25.0, True, "123456789")
    st = os.stat(store.DB_PATH)
    assert stat.S_IMODE(st.st_mode) == 0o600
    raw = sqlite3.connect(store.DB_PATH).execute("SELECT name, telegram_chat_id FROM clients").fetchone()
    assert raw[0].startswith("enc:") and "Secreto" not in raw[0] and raw[1].startswith("enc:")         # cifrado en reposo
    blob = open(store.DB_PATH, "rb").read() + (open(store.DB_PATH + "-wal", "rb").read() if os.path.exists(store.DB_PATH + "-wal") else b"")
    assert b"Cliente Secreto" not in blob and b"123456789" not in blob
    cl = store.list_clients()[0]
    assert cl["name"] == "Cliente Secreto S.A." and cl["telegram_chat_id"] == "123456789"
    try: store.add_client("cliente secreto s.a.", 100); raise AssertionError("debe rechazar duplicado")
    except sqlite3.IntegrityError: pass
    try: store.add_client("X", -5); raise AssertionError()
    except ValueError: pass
    # integridad referencial: no se pueden crear posiciones de clientes inexistentes
    try: store.record_fill(999, "AAPL", 1, "2026-01-01", 100, 90); raise AssertionError("FK")
    except sqlite3.IntegrityError: pass
    try: store.record_fill(cl["id"], "AAPL; DROP TABLE", 1, "2026-01-01", 100, 90); raise AssertionError()
    except ValueError: pass
    store.record_fill(cl["id"], "aapl", 2, "2026-01-01", 100, 90, "pullback")
    assert store.open_positions(cl["id"])[0]["ticker"] == "AAPL"

def test_audit_chain():
    cid = store.list_clients()[0]["id"]
    for i in range(4):
        assert store.claim_alert(f"2026-10-2{i}:{cid}:AAPL:entry", cid, "entry", "AAPL", f"[Cliente Secreto S.A.] 🟢 ENTRADA {i}")
    assert store.claim_alert("2026-10-20:%d:AAPL:entry" % cid, cid, "entry", "AAPL", "x") is False
    ok, bad = store.verify_audit_chain(); assert ok and bad is None
    c = sqlite3.connect(store.DB_PATH)
    msgs = [r[0] for r in c.execute("SELECT message FROM alerts_log")]
    assert all("Secreto" not in m and m.startswith("[cliente]") for m in msgs)                      # sin datos personales
    c.execute("UPDATE alerts_log SET message='ALTERADO' WHERE rowid=2"); c.commit(); c.close()
    ok, bad = store.verify_audit_chain(); assert not ok and bad == 2                                  # la manipulación se detecta

def test_crypto_and_redaction():
    assert crypto.decrypt(crypto.encrypt("hola")) == "hola" and crypto.blind_index("A ") == crypto.blind_index("a")
    fake = "123456789" + ":" + "Zq" * 17 + "x"                        # token FALSO armado en ejecución (evita falsos positivos del escáner de secretos de GitHub)
    s = redact(f"fallo https://api.telegram.org/bot{fake}/sendMessage token=abc123 mail yo@x.com")
    assert "Zq" not in s and "abc123" not in s and "yo@x.com" not in s

if __name__ == "__main__":
    for n_, f_ in list(globals().items()):
        if n_.startswith("test_"): f_(); print("OK ", n_)
