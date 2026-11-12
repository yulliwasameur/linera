import os, sys, tempfile, shutil, sqlite3, pytest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lf.store import LineageStore
from lf.vindex import ShardedIndex
from lf.keydag import Forgotten

def _mk(B=4):
    d = tempfile.mkdtemp()
    s = LineageStore(os.path.join(d, "m.db"), os.path.join(d, "st.json"))
    return d, s, ShardedIndex(s, dim=8, shard_size=B)

def _v(i):
    r = np.random.default_rng(i).normal(size=8).astype(np.float32); return r / np.linalg.norm(r)

def test_search_returns_nearest():
    d, s, ix = _mk()
    tags = [s.put(f"o{i}") for i in range(10)]
    for i, t in enumerate(tags): ix.add(t, _v(i))
    assert ix.search(_v(7), k=1)[0][1] == tags[7]

def test_forget_removes_descendant_vectors_keeps_survivors():
    d, s, ix = _mk(B=3)
    a, b = s.put("a"), s.put("b")
    r = s.put("r", "derived", [a, b]); g = s.put("g", "derived", [b])
    for i, t in enumerate([a, b, r, g]): ix.add(t, _v(i))
    ix.forget(a)
    hits = {t for _, t in ix.search(_v(2), k=10)}
    assert a not in hits and r not in hits and {b, g} <= hits

def test_stale_shard_copy_is_dead_after_forget():
    d, s, ix = _mk(B=8)
    a, b = s.put("a"), s.put("b"); r = s.put("r", "derived", [a])
    for i, t in enumerate([a, b, r]): ix.add(t, _v(i))
    backup = os.path.join(d, "backup.db"); s.db.commit(); shutil.copy(s.path, backup)   # backup before forget
    ix.forget(a); s.close()
    thief = LineageStore(backup, os.path.join(d, "st.json"))   # old backup + current (punctured) state
    tix = ShardedIndex(thief, dim=8, shard_size=8)
    with pytest.raises(Forgotten):
        tix._read(0)

def test_rebuilt_shard_readable_after_reload():
    d, s, ix = _mk(B=8)
    a, b, c = s.put("a"), s.put("b"), s.put("c")
    for i, t in enumerate([a, b, c]): ix.add(t, _v(i))
    ix.forget(b); s.close()
    s2 = LineageStore(os.path.join(d, "m.db"), os.path.join(d, "st.json"))
    ix2 = ShardedIndex(s2, dim=8, shard_size=8)
    members, v = ix2._read(0)
    assert sorted(members) == sorted([a, c]) and v.shape == (2, 8)
