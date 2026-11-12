"""Baselines with the same API as LineageStore.

PlainLogical : plaintext SQLite, forget = DELETE of the source row only (ForgetAgent-style logical deletion
               without lineage). Residue: SQLite free pages keep the bytes unless secure_delete/VACUUM.
PerObjectKey : one PPRF key per object, no lineage (Holepunch-style). Forget punctures only the source.
EpochKey     : one key per epoch (Ghost-Vectors-style). Forget of any object -> destroy that epoch key
               after re-encrypting the epoch's other objects under a fresh epoch key.
"""
import sqlite3, json, secrets, os
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from lf.pprf import PPRF, Punctured

class PlainLogical:
    def __init__(self, path, state_path=None):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS obj (tag INTEGER PRIMARY KEY, kind TEXT, parents TEXT, body TEXT)")
    def put(self, content, kind="source", parents=()):
        t = secrets.randbits(32)
        self.db.execute("INSERT INTO obj VALUES (?,?,?,?)", (t, kind, json.dumps(sorted(parents)), content)); self.db.commit(); return t
    def get(self, t):
        r = self.db.execute("SELECT body FROM obj WHERE tag=?", (t,)).fetchone()
        if r is None: raise KeyError(t)
        return r[0]
    def readable(self, t):
        try: self.get(t); return True
        except KeyError: return False
    def forget(self, t, delete_rows=True):
        self.db.execute("DELETE FROM obj WHERE tag=?", (t,)); self.db.commit(); return {t}
    def tags(self): return [t for (t,) in self.db.execute("SELECT tag FROM obj")]
    def close(self): self.db.close()

class PerObjectKey:
    def __init__(self, path, state_path=None):
        self.db = sqlite3.connect(path); self.f = PPRF()
        self.db.execute("CREATE TABLE IF NOT EXISTS obj (tag INTEGER PRIMARY KEY, kind TEXT, parents TEXT, nonce BLOB, ct BLOB)")
    def put(self, content, kind="source", parents=()):
        t = secrets.randbits(32); n = secrets.token_bytes(12)
        ct = ChaCha20Poly1305(self.f.eval(t)).encrypt(n, content.encode(), t.to_bytes(4, "big"))
        self.db.execute("INSERT INTO obj VALUES (?,?,?,?,?)", (t, kind, json.dumps(sorted(parents)), n, ct)); self.db.commit(); return t
    def get(self, t):
        r = self.db.execute("SELECT nonce, ct FROM obj WHERE tag=?", (t,)).fetchone()
        if r is None: raise KeyError(t)
        return ChaCha20Poly1305(self.f.eval(t)).decrypt(r[0], r[1], t.to_bytes(4, "big")).decode()
    def readable(self, t):
        try: self.get(t); return True
        except (KeyError, Punctured): return False
    def forget(self, t, delete_rows=True):
        self.f.puncture(t)
        if delete_rows: self.db.execute("DELETE FROM obj WHERE tag=?", (t,)); self.db.commit()
        return {t}
    def tags(self): return [t for (t,) in self.db.execute("SELECT tag FROM obj")]
    def close(self): self.db.close()

class EpochKey:
    """Objects are assigned to the current epoch; set_epoch() starts a new one."""
    def __init__(self, path, state_path=None):
        self.db = sqlite3.connect(path); self.keys = {0: os.urandom(32)}; self.epoch = 0
        self.db.execute("CREATE TABLE IF NOT EXISTS obj (tag INTEGER PRIMARY KEY, kind TEXT, parents TEXT, epoch INTEGER, nonce BLOB, ct BLOB)")
    def set_epoch(self, e):
        self.epoch = e; self.keys.setdefault(e, os.urandom(32))
    def put(self, content, kind="source", parents=()):
        t = secrets.randbits(32); n = secrets.token_bytes(12)
        ct = ChaCha20Poly1305(self.keys[self.epoch]).encrypt(n, content.encode(), t.to_bytes(4, "big"))
        self.db.execute("INSERT INTO obj VALUES (?,?,?,?,?,?)", (t, kind, json.dumps(sorted(parents)), self.epoch, n, ct)); self.db.commit(); return t
    def get(self, t):
        r = self.db.execute("SELECT epoch, nonce, ct FROM obj WHERE tag=?", (t,)).fetchone()
        if r is None or r[0] not in self.keys: raise KeyError(t)
        return ChaCha20Poly1305(self.keys[r[0]]).decrypt(r[1], r[2], t.to_bytes(4, "big")).decode()
    def readable(self, t):
        try: self.get(t); return True
        except KeyError: return False
    def forget(self, t, delete_rows=True):
        r = self.db.execute("SELECT epoch FROM obj WHERE tag=?", (t,)).fetchone()
        if r is None: return set()
        e = r[0]; others = [x for (x,) in self.db.execute("SELECT tag FROM obj WHERE epoch=? AND tag<>?", (e, t))]
        new_e = max(self.keys) + 1; self.keys[new_e] = os.urandom(32)
        for x in others:
            body = self.get(x); n = secrets.token_bytes(12)
            ct = ChaCha20Poly1305(self.keys[new_e]).encrypt(n, body.encode(), x.to_bytes(4, "big"))
            self.db.execute("UPDATE obj SET epoch=?, nonce=?, ct=? WHERE tag=?", (new_e, n, ct, x))
        del self.keys[e]
        self.db.execute("DELETE FROM obj WHERE tag=?", (t,)); self.db.commit()
        return {t}
    def tags(self): return [t for (t,) in self.db.execute("SELECT tag FROM obj")]
    def close(self): self.db.close()
