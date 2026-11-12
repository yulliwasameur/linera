"""Uniform memory back-ends (text + vector) for the replay experiments.

scheme        text storage                         vector storage                       forget(o)
plain_logical plaintext SQLite (DELETE row)        hnswlib file, mark_deleted (soft)    delete o only
per_object    PPRF key per object (Holepunch-like) vector AEAD-encrypted with o's key   puncture o only
epoch         one key per epoch (Ghost-Vectors fix) vector encrypted with epoch key     re-key epoch, drop o
linera        LinEra key DAG                        ShardedIndex (shard keys from DAG)   puncture o -> closure
"""
from __future__ import annotations
import os, json, sqlite3, secrets, tempfile
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from lf.store import LineageStore
from lf.vindex import ShardedIndex
from lf.keydag import Forgotten
from lf.pprf import PPRF, Punctured

class _Base:
    def __init__(self, d, dim):
        self.dir, self.dim = d, dim
        self.kind = {}
    def search(self, q, k=10, kinds=None):
        raise NotImplementedError

class LinEraMem(_Base):
    name = "linera"
    def __init__(self, d, dim, shard_size=64):
        super().__init__(d, dim)
        self.db_path, self.state_path = os.path.join(d, "m.db"), os.path.join(d, "state.json")
        self.s = LineageStore(self.db_path, self.state_path)
        self.ix = ShardedIndex(self.s, dim, shard_size)
    def put(self, text, kind, parents, vec):
        t = self.s.put(text, kind, parents, commit=False); self.kind[t] = kind
        self.ix.add(t, vec, commit=False)
        return t
    def commit(self): self.ix.flush(); self.s.db.commit()
    def readable(self, t): return self.s.readable(t)
    def get(self, t): return self.s.get(t)
    def forget(self, t):
        if not self.s.readable(t) and t not in self.s.dag.parents: return set()
        return self.ix.forget(t)
    def search(self, q, k=10, kinds=None):
        allow = (lambda t: self.kind.get(t) in kinds) if kinds else None
        return self.ix.search(q, k, allow)
    def files(self): return [self.db_path]
    def close(self): self.ix.flush(); self.s.close()
    # attacker view: reopen from disk with the current (stolen) state
    def thief(self):
        s = LineageStore(self.db_path, self.state_path); ix = ShardedIndex(s, self.dim, self.ix.B)
        texts, vecs = {}, []
        for t in s.tags():
            try: texts[t] = s.get(t)
            except Forgotten: pass
        for (sid,) in s.db.execute("SELECT sid FROM shard").fetchall():
            try: vecs.append(ix._read(sid)[1])
            except Forgotten: pass
        return texts, (np.vstack(vecs) if vecs else np.zeros((0, self.dim), np.float32))

class PlainLogicalMem(_Base):
    name = "plain_logical"
    def __init__(self, d, dim):
        super().__init__(d, dim)
        import hnswlib
        self.db_path, self.ix_path = os.path.join(d, "m.db"), os.path.join(d, "vec.hnsw")
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("CREATE TABLE obj (tag INTEGER PRIMARY KEY, kind TEXT, parents TEXT, body TEXT)")
        self.ix = hnswlib.Index("ip", dim); self.ix.init_index(200000, ef_construction=100, M=16); self.ix.set_ef(100)
        self.alive = set()
    def put(self, text, kind, parents, vec):
        t = secrets.randbits(32)
        while t in self.kind: t = secrets.randbits(32)
        self.db.execute("INSERT INTO obj VALUES (?,?,?,?)", (t, kind, json.dumps(sorted(parents)), text))
        self.ix.add_items(np.asarray(vec, np.float32).reshape(1, -1), [t]); self.kind[t] = kind; self.alive.add(t)
        return t
    def commit(self): self.db.commit(); self.ix.save_index(self.ix_path)
    def readable(self, t): return t in self.alive
    def get(self, t): return self.db.execute("SELECT body FROM obj WHERE tag=?", (t,)).fetchone()[0]
    def forget(self, t):
        if t not in self.alive: return set()
        self.db.execute("DELETE FROM obj WHERE tag=?", (t,)); self.ix.mark_deleted(t); self.alive.discard(t)
        return {t}
    def search(self, q, k=10, kinds=None):
        n = len(self.alive)
        if n == 0: return []
        labels, dist = self.ix.knn_query(np.asarray(q, np.float32).reshape(1, -1), k=min(n, k * 6))
        out = [(1 - float(s), int(l)) for l, s in zip(labels[0], dist[0]) if (not kinds or self.kind[int(l)] in kinds)]
        return out[:k]
    def files(self): return [self.db_path, self.ix_path]
    def close(self): self.commit(); self.db.close()
    def thief(self):
        import hnswlib
        db = sqlite3.connect(self.db_path)
        texts = {t: b for t, b in db.execute("SELECT tag, body FROM obj")}
        ix = hnswlib.Index("ip", self.dim); ix.load_index(self.ix_path)
        ids = ix.get_ids_list()
        for i in ids:                       # Ghost-Vectors-style undelete of soft-deleted points
            try: ix.unmark_deleted(i)
            except RuntimeError: pass
        return texts, np.asarray(ix.get_items(ids), np.float32)

class _EncBase(_Base):
    """Shared code for per-object and epoch keys: text and vector AEAD-encrypted under the same key."""
    def __init__(self, d, dim):
        super().__init__(d, dim)
        self.db_path = os.path.join(d, "m.db"); self.db = sqlite3.connect(self.db_path)
        self.db.execute("CREATE TABLE obj (tag INTEGER PRIMARY KEY, kind TEXT, parents TEXT, grp INTEGER, nonce BLOB, ct BLOB, vnonce BLOB, vct BLOB)")
        self.vec = {}
    def _enc(self, k, t, text, vec):
        n, vn = secrets.token_bytes(12), secrets.token_bytes(12); ad = t.to_bytes(4, "big")
        return n, ChaCha20Poly1305(k).encrypt(n, text.encode(), ad), vn, ChaCha20Poly1305(k).encrypt(vn, np.asarray(vec, "<f4").tobytes(), ad)
    def _dec(self, k, t, row):
        ad = t.to_bytes(4, "big")
        return ChaCha20Poly1305(k).decrypt(row[0], row[1], ad).decode(), np.frombuffer(ChaCha20Poly1305(k).decrypt(row[2], row[3], ad), "<f4")
    def put(self, text, kind, parents, vec):
        t = secrets.randbits(32)
        while t in self.kind: t = secrets.randbits(32)
        g = self._grp(t); k = self._key(t, g)
        self.db.execute("INSERT INTO obj VALUES (?,?,?,?,?,?,?,?)", (t, kind, json.dumps(sorted(parents)), g, *self._enc(k, t, text, vec)))
        self.kind[t] = kind; self.vec[t] = np.asarray(vec, np.float32)
        return t
    def commit(self): self.db.commit()
    def _row(self, t, db=None):
        return (db or self.db).execute("SELECT nonce, ct, vnonce, vct, grp FROM obj WHERE tag=?", (t,)).fetchone()
    def readable(self, t): return t in self.vec
    def get(self, t):
        r = self._row(t); return self._dec(self._key(t, r[4]), t, r)[0]
    def search(self, q, k=10, kinds=None):
        tags = [t for t in self.vec if not kinds or self.kind[t] in kinds]
        if not tags: return []
        M = np.vstack([self.vec[t] for t in tags]); sc = M @ np.asarray(q, np.float32)
        return [(float(sc[i]), tags[i]) for i in np.argsort(-sc)[:k]]
    def files(self): return [self.db_path]
    def close(self): self.db.commit(); self.db.close()
    def thief(self):
        db = sqlite3.connect(self.db_path); texts, vecs = {}, []
        for (t,) in db.execute("SELECT tag FROM obj").fetchall():
            r = self._row(t, db)
            try:
                k = self._key(t, r[4])
                tx, v = self._dec(k, t, r); texts[t] = tx; vecs.append(v)
            except Exception:
                pass
        return texts, (np.vstack(vecs) if vecs else np.zeros((0, self.dim), np.float32))

class PerObjectMem(_EncBase):
    name = "per_object"
    def __init__(self, d, dim):
        super().__init__(d, dim); self.f = PPRF()
    def _grp(self, t): return 0
    def _key(self, t, g): return self.f.eval(t)
    def forget(self, t):
        if t not in self.vec: return set()
        self.f.puncture(t); self.db.execute("DELETE FROM obj WHERE tag=?", (t,)); self.vec.pop(t); return {t}

class EpochMem(_EncBase):
    """Epoch = session counter // epoch_len (set by the harness through set_epoch)."""
    name = "epoch"
    def __init__(self, d, dim):
        super().__init__(d, dim); self.keys = {0: os.urandom(32)}; self.epoch = 0; self.next = 1
    def set_epoch(self, e):
        if e != self.epoch:
            self.epoch = self.next; self.next += 1; self.keys[self.epoch] = os.urandom(32)
    def _grp(self, t): return self.epoch
    def _key(self, t, g):
        if g not in self.keys: raise KeyError(g)
        return self.keys[g]
    def forget(self, t):
        if t not in self.vec: return set()
        g = self._row(t)[4]
        others = [x for (x,) in self.db.execute("SELECT tag FROM obj WHERE grp=? AND tag<>?", (g, t)).fetchall()]
        ng = self.next; self.next += 1; self.keys[ng] = os.urandom(32)
        for x in others:
            r = self._row(x); tx, v = self._dec(self.keys[g], x, r)
            n, ct, vn, vct = self._enc(self.keys[ng], x, tx, v)
            self.db.execute("UPDATE obj SET grp=?, nonce=?, ct=?, vnonce=?, vct=? WHERE tag=?", (ng, n, ct, vn, vct, x))
        del self.keys[g]
        if self.epoch == g: self.epoch = ng
        self.db.execute("DELETE FROM obj WHERE tag=?", (t,)); self.vec.pop(t); return {t}

SCHEMES = {"plain_logical": PlainLogicalMem, "per_object": PerObjectMem, "epoch": EpochMem, "linera": LinEraMem}

def make(scheme, dim):
    return SCHEMES[scheme](tempfile.mkdtemp(prefix=f"lf_{scheme}_"), dim)
