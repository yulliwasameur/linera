"""LinEra storage back-end for mem0 (tested with mem0ai 2.2.1).

Two pieces:
  LinEraVectorStore  implements mem0's VectorStoreBase. Every memory (payload incl. its text) is an
                     encrypted LinEra object; vectors live in the encrypted ShardedIndex.
  LinEraMemory       wraps mem0.Memory. It stores each incoming message as a LinEra *source* object and
                     applies the sound lineage rule at the storage layer:
                       parents(new memory) = messages of this add() call
                                           + earlier messages mem0 shows the LLM (last 10 of the scope)
                                           + every memory returned by vector_store.search() during the call
                     forget_message(t) punctures t: every memory written with t in its context dies.
mem0's own history database is a plaintext side channel (it stores raw messages and, on delete, the old
memory text). LinEraMemory therefore runs mem0 with history_db_path=":memory:".
"""
from __future__ import annotations
import json, os, threading, uuid
from collections import defaultdict, deque
import numpy as np
from mem0.vector_stores.base import VectorStoreBase
from lf.store import LineageStore
from lf.vindex import ShardedIndex
from lf.keydag import Forgotten

class _Out:
    def __init__(self, id, score, payload):
        self.id, self.score, self.payload = id, score, payload
    def __repr__(self): return f"_Out({self.id!r}, {self.score!r})"

class LineageContext(threading.local):
    def __init__(self):
        self.active = False; self.tags: list[int] = []
    def add(self, t):
        if self.active and t not in self.tags: self.tags.append(t)

class LinEraVectorStore(VectorStoreBase):
    def __init__(self, path: str, dim: int, shard_size: int = 64, collection_name: str = "mem0"):
        os.makedirs(path, exist_ok=True)
        self.path, self.dim, self.collection_name = path, dim, collection_name
        self.s = LineageStore(os.path.join(path, "linera.db"), os.path.join(path, "linera.state"))
        self.s.db.execute("CREATE TABLE IF NOT EXISTS idmap (mid TEXT PRIMARY KEY, tag INTEGER)")
        self.ix = ShardedIndex(self.s, dim, shard_size)
        self.ctx = LineageContext()
        self.idmap = {m: t for m, t in self.s.db.execute("SELECT mid, tag FROM idmap")}

    # ---- helpers
    def _payload(self, tag):
        return json.loads(self.s.get(tag))
    def _put(self, mid, vec, payload, parents):
        tag = self.s.put(json.dumps(payload), "memory", parents, commit=False)
        self.ix.add(tag, vec, commit=False)
        self.s.db.execute("INSERT OR REPLACE INTO idmap VALUES (?,?)", (mid, tag)); self.idmap[mid] = tag
        self.ix.flush()
        return tag
    def _alive(self, mid):
        t = self.idmap.get(mid)
        return t is not None and self.s.readable(t)
    def _match(self, payload, filters):
        return all(payload.get(k) == v for k, v in (filters or {}).items() if not isinstance(v, (dict, list)))

    # ---- VectorStoreBase
    def create_col(self, name, vector_size, distance): pass
    def insert(self, vectors, payloads=None, ids=None):
        payloads = payloads or [{}] * len(vectors); ids = ids or [str(uuid.uuid4()) for _ in vectors]
        for v, p, i in zip(vectors, payloads, ids):
            self._put(i, np.asarray(v, np.float32) / (np.linalg.norm(v) or 1), p, list(self.ctx.tags))
    def search(self, query, vectors, top_k=5, filters=None):
        q = np.asarray(vectors, np.float32); q = q / (np.linalg.norm(q) or 1)
        rev = {t: m for m, t in self.idmap.items()}
        out = []
        for score, tag in self.ix.search(q, k=top_k * 4, allow=lambda t: t in rev):
            try: p = self._payload(tag)
            except (Forgotten, KeyError): continue
            if not self._match(p, filters): continue
            out.append(_Out(rev[tag], max(0.0, score), p)); self.ctx.add(tag)   # what mem0 shows the LLM
            if len(out) == top_k: break
        return out
    def delete(self, vector_id):
        t = self.idmap.pop(vector_id, None)
        if t is not None:
            self.ix.forget(t); self.s.db.execute("DELETE FROM idmap WHERE mid=?", (vector_id,)); self.s.db.commit()
    def update(self, vector_id, vector=None, payload=None):
        old = self.idmap[vector_id]
        p = payload if payload is not None else self._payload(old)
        if vector is None:
            members, V = self.ix._read(self.ix.where[old]); vector = V[members.index(old)]
        # the new version inherits the old version's lineage plus the current context
        parents = list(self.s.dag.parents.get(old, ())) + list(self.ctx.tags)
        # keep the old ciphertext (its row also carries lineage needed by descendants); mark it superseded
        self.s.db.execute("UPDATE obj SET kind='superseded' WHERE tag=?", (old,))
        self._put(vector_id, np.asarray(vector, np.float32), p, [t for t in parents if t != old])
    def get(self, vector_id):
        if not self._alive(vector_id): return None
        t = self.idmap[vector_id]; self.ctx.add(t)
        return _Out(vector_id, None, self._payload(t))
    def list_cols(self): return [self.collection_name]
    def delete_col(self): pass
    def col_info(self): return {"name": self.collection_name, "count": len(self.idmap)}
    def list(self, filters=None, top_k=None):
        res = []
        for mid, t in list(self.idmap.items()):
            try: p = self._payload(t)
            except (Forgotten, KeyError): continue
            if self._match(p, filters): res.append(_Out(mid, None, p))
        return [res[:top_k] if top_k else res]
    def reset(self): pass

class LinEraMemory:
    """mem0.Memory with LinEra storage and sound lineage capture."""
    def __init__(self, path, config: dict, dim: int):
        from mem0 import Memory
        cfg = dict(config); cfg["history_db_path"] = ":memory:"
        self.m = Memory.from_config(cfg)
        self.vs = LinEraVectorStore(path, dim)
        self.m.vector_store = self.vs                       # swap mem0's store for LinEra
        self.recent = defaultdict(lambda: deque(maxlen=10))  # mirrors mem0's last-10-messages context
    def add_message(self, content, role="user", user_id="u", infer=True):
        """Store one message as a source object and run mem0's add() with lineage capture. Returns its tag."""
        t = self.vs.s.put(json.dumps({"role": role, "content": content}), "message", (), commit=True)
        ctx = self.vs.ctx; ctx.active = True
        # sound rule = exactly what the writer sees: with infer=True mem0 shows the LLM the last 10 messages
        ctx.tags = (list(self.recent[user_id]) if infer else []) + [t]
        try:
            res = self.m.add([{"role": role, "content": content}], user_id=user_id, infer=infer)
        finally:
            ctx.active = False; ctx.tags = []
        self.recent[user_id].append(t)
        return t, res
    def forget_message(self, tag):
        """Right-to-erasure request for one message: its lineage closure becomes undecryptable."""
        for dq in self.recent.values():
            if tag in dq: dq.remove(tag)
        closure = self.vs.ix.forget(tag)
        for mid, t in list(self.vs.idmap.items()):
            if t in closure:
                self.vs.idmap.pop(mid); self.vs.s.db.execute("DELETE FROM idmap WHERE mid=?", (mid,))
        self.vs.s.db.commit()
        return closure
    def search(self, query, user_id="u", limit=5):
        return self.m.search(query, filters={"user_id": user_id}, top_k=limit)
