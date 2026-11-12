import os, sys, tempfile, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lf.pprf import PPRF, Punctured
from lf.store import LineageStore
from lf.keydag import Forgotten

def test_pprf_puncture_preserves_other_points():
    f = PPRF(seed=b"\x01" * 32)
    pts = [0, 1, 2, 3, 12345, 2**32 - 1, 777777]
    before = {x: f.eval(x) for x in pts}
    f.puncture(12345)
    with pytest.raises(Punctured):
        f.eval(12345)
    for x in pts:
        if x != 12345:
            assert f.eval(x) == before[x]
    assert len(f.nodes) <= 32

def test_pprf_serialization_roundtrip():
    f = PPRF(); f.puncture(5); f.puncture(99)
    g = PPRF.loads(f.dumps())
    assert g.eval(6) == f.eval(6)
    with pytest.raises(Punctured):
        g.eval(99)

def test_lineage_closure_forgotten_and_others_kept():
    d = tempfile.mkdtemp()
    s = LineageStore(os.path.join(d, "m.db"), os.path.join(d, "state.json"))
    a = s.put("A: address 12 rue X"); b = s.put("B: likes trains")
    r = s.put("summary of A and B", "derived", [a, b])
    f = s.put("fact from summary", "derived", [r])
    g = s.put("fact from B only", "derived", [b])
    s.forget(a, delete_rows=False)
    for t in (a, r, f):
        with pytest.raises(Forgotten):
            s.get(t)
    assert s.get(b) == "B: likes trains" and s.get(g) == "fact from B only"

def test_forward_security_after_state_theft():
    d = tempfile.mkdtemp()
    db, st = os.path.join(d, "m.db"), os.path.join(d, "state.json")
    s = LineageStore(db, st)
    a = s.put("SECRET-CANARY"); r = s.put("derived SECRET-CANARY", "derived", [a])
    s.forget(a, delete_rows=False); s.close()
    thief = LineageStore(db, st)  # attacker loads the stolen db + current state
    assert not thief.readable(a) and not thief.readable(r)
    assert b"SECRET-CANARY" not in open(db, "rb").read()

def test_rotation_keeps_survivors_and_shrinks_state():
    d = tempfile.mkdtemp()
    s = LineageStore(os.path.join(d, "m.db"), os.path.join(d, "state.json"))
    tags = [s.put(f"obj {i}") for i in range(50)]
    for t in tags[:25]:
        s.forget(t)
    big = s.dag.pprf.size_bytes()
    n = s.rotate()
    assert n == 25 and s.dag.pprf.size_bytes() < big
    assert all(s.get(t) == f"obj {i}" for i, t in enumerate(tags) if i >= 25)

def test_tamper_lineage_detected():
    d = tempfile.mkdtemp()
    s = LineageStore(os.path.join(d, "m.db"), os.path.join(d, "state.json"))
    a = s.put("a"); b = s.put("b"); r = s.put("r", "derived", [a])
    s.db.execute("UPDATE obj SET parents='[%d]' WHERE tag=?" % b, (r,)); s.db.commit()
    s2 = LineageStore(s.path, s.state_path)
    with pytest.raises(Exception):
        s2.get(r)

def test_deep_lineage_chain_cold_cache():
    """Regression: key derivation must not recurse (chains can be thousands of objects deep)."""
    from lf.keydag import KeyDAG
    d = KeyDAG(PPRF(seed=b"\x02" * 32)); prev = None
    for t in range(1, 5001):
        d.register(t, [prev] if prev else []); prev = t
    k = d.key(5000); d.drop_cache()
    assert d.key(5000) == k
    d.forget(1)
    with pytest.raises(Forgotten):
        d.key(5000)
