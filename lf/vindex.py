"""Encrypted, sharded vector index whose shard keys follow the lineage key DAG.

Each shard holds up to `shard_size` (tag, vector) pairs as one flat float32 block.
Shard key:  k_shard = HKDF(k_m1 || ... || k_mn, info = "lf/shard" || sid || version)  (members sorted by tag),
where k_m are the LinEra object keys. Hence a shard ciphertext is readable only while *every* member is
readable: forgetting any ancestor of any member kills the shard ciphertext, including stale copies in
backups or free pages. On forget, affected shards are decrypted first, the PPRF is punctured, then the
shard is rebuilt from its surviving members under a fresh version and re-encrypted.
Crash between puncture and rebuild: survivors' vectors are recomputed from their (still readable) text.
"""
from __future__ import annotations
import json, secrets
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from .keydag import Forgotten

class ShardedIndex:
    def __init__(self, store, dim: int, shard_size: int = 256):
        self.s, self.dim, self.B = store, dim, shard_size
        self.s.db.execute("CREATE TABLE IF NOT EXISTS shard (sid INTEGER PRIMARY KEY, ver INTEGER, members TEXT, nonce BLOB, ct BLOB)")
        self.s.db.commit()
        self._cache: dict[int, tuple[list[int], np.ndarray]] = {}
        self.where: dict[int, int] = {}
        for sid, members in self.s.db.execute("SELECT sid, members FROM shard"):
            for m in json.loads(members):
                self.where[m] = sid
        self.open_sid = max([sid for (sid,) in self.s.db.execute("SELECT sid FROM shard")], default=-1)
        self.rebuilds = 0
        self._dirty: set[int] = set()

    def _key(self, sid, ver, members):
        mat = b"".join(self.s.dag.key(m) for m in sorted(members))
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"lf/shard" + sid.to_bytes(4, "big") + ver.to_bytes(4, "big")).derive(mat)

    def _ad(self, sid, ver, members):
        return json.dumps([sid, ver, sorted(members)]).encode()

    def _write(self, sid, ver, members, vecs):
        members = list(members)
        if not members:
            self.s.db.execute("DELETE FROM shard WHERE sid=?", (sid,)); self._cache.pop(sid, None); return
        order = np.argsort(members); members = [members[i] for i in order]; vecs = vecs[order]
        n = secrets.token_bytes(12)
        ct = ChaCha20Poly1305(self._key(sid, ver, members)).encrypt(n, vecs.astype("<f4").tobytes(), self._ad(sid, ver, members))
        self.s.db.execute("INSERT OR REPLACE INTO shard VALUES (?,?,?,?,?)", (sid, ver, json.dumps(members), n, ct))
        self._cache[sid] = (members, vecs.astype(np.float32))
        for m in members:
            self.where[m] = sid

    def _read(self, sid):
        if sid in self._cache:
            return self._cache[sid]
        row = self.s.db.execute("SELECT ver, members, nonce, ct FROM shard WHERE sid=?", (sid,)).fetchone()
        if row is None:
            return [], np.zeros((0, self.dim), np.float32)
        ver, members = row[0], json.loads(row[1])
        pt = ChaCha20Poly1305(self._key(sid, ver, members)).decrypt(row[2], row[3], self._ad(sid, ver, members))
        v = np.frombuffer(pt, dtype="<f4").reshape(len(members), self.dim).copy()
        self._cache[sid] = (members, v)
        return members, v

    def add(self, tag: int, vec: np.ndarray, commit=True):
        """Append to the open shard in RAM; the shard is (re)sealed by flush() (called on commit)."""
        vec = np.asarray(vec, np.float32).reshape(1, self.dim)
        if self.open_sid >= 0:
            members, v = self._read(self.open_sid)
            if len(members) < self.B:
                self._cache[self.open_sid] = (members + [tag], np.vstack([v, vec]))
                self.where[tag] = self.open_sid; self._dirty.add(self.open_sid)
                if commit: self.flush()
                return
            self.flush()
        self.open_sid += 1
        self._cache[self.open_sid] = ([tag], vec); self.where[tag] = self.open_sid; self._dirty.add(self.open_sid)
        if commit: self.flush()

    def flush(self):
        for sid in sorted(self._dirty):
            members, v = self._cache[sid]
            row = self.s.db.execute("SELECT ver FROM shard WHERE sid=?", (sid,)).fetchone()
            self._write(sid, (row[0] + 1) if row else 0, members, v)
        self._dirty.clear(); self.s.db.commit()

    def search(self, q: np.ndarray, k: int = 10, allow=None):
        q = np.asarray(q, np.float32).ravel()
        best = []
        sids = {sid for (sid,) in self.s.db.execute("SELECT sid FROM shard").fetchall()} | set(self._dirty)
        for sid in sorted(sids):
            try:
                members, v = self._read(sid)
            except Forgotten:
                continue
            if not members: continue
            sc = v @ q
            for i in np.argsort(-sc)[: k * 4]:
                if allow is None or allow(members[i]):
                    best.append((float(sc[i]), members[i]))
        best.sort(reverse=True)
        return best[:k]

    def forget(self, tag: int) -> set[int]:
        self.flush()
        closure = {tag} | self.s.dag.descendants(tag)
        affected = {self.where[m] for m in closure if m in self.where}
        plain = {}
        for sid in affected:                      # 1. decrypt affected shards while keys still exist
            try:
                plain[sid] = self._read(sid)
            except Forgotten:
                plain[sid] = None
        self.s.forget(tag)                        # 2. puncture, persist state, drop rows
        for sid in affected:                      # 3. rebuild from survivors under a fresh version
            self._cache.pop(sid, None)
            ver = self.s.db.execute("SELECT ver FROM shard WHERE sid=?", (sid,)).fetchone()[0] + 1
            if plain[sid] is None:
                continue
            members, v = plain[sid]
            keep = [i for i, m in enumerate(members) if m not in closure]
            self._write(sid, ver, [members[i] for i in keep], v[keep])
            self.rebuilds += 1
        for m in closure:
            self.where.pop(m, None)
        self.s.db.commit()
        return closure

    def state_reset_cache(self):
        self._cache.clear()
