"""E4: system costs of LinEra from 10^4 to 10^6 objects (no LLM).

Synthetic workload shaped like the LLM log of phase A: sessions of S source objects followed by D derived
objects whose sound parents are the session's sources plus R memories retrieved from a recent window of
earlier derived objects. Vectors are random unit vectors of dimension 384 (bge-small size).
Measures: write cost (incl. vector index), read latency, search latency (all shards decrypted and cached),
forget latency (puncture + closure + shard rebuilds), lineage closure size, PPRF state size, compaction
(rotate) time, database size.
Usage: python3 -I eval/scale.py --sizes 10000,100000,1000000 --out results/results_scale.json
"""
import argparse, json, os, random, sys, tempfile, time, platform
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lf.store import LineageStore
from lf.vindex import ShardedIndex
from lf.keydag import Forgotten

SEED = 20261109

def run(n, regime="chained", S=22, D=9, R=3, window=60, dim=384, shard=256, n_forget=100, n_get=1000):
    """regime=chained: the writer retrieves earlier *derived* memories (as in our LLM log);
    regime=flat: it retrieves earlier *source* turns, so every derived object has only source parents."""
    rnd = random.Random(SEED); rng = np.random.default_rng(SEED)
    d = tempfile.mkdtemp(prefix=f"scale{n}_")
    st = LineageStore(os.path.join(d, "m.db"), os.path.join(d, "st.json")); ix = ShardedIndex(st, dim, shard)
    tags, derived, sources = [], [], []
    t0 = time.perf_counter()
    while len(tags) < n:
        sess = []
        for _ in range(S):
            v = rng.standard_normal(dim).astype(np.float32); v /= np.linalg.norm(v)
            t = st.put("source text " + "x" * 120, "source", (), commit=False); ix.add(t, v, commit=False)
            sess.append(t); tags.append(t); sources.append(t)
        pool = (derived if regime == "chained" else sources[:-S])[-window:]
        ret = rnd.sample(pool, min(R, len(pool))) if pool else []
        for _ in range(D):
            v = rng.standard_normal(dim).astype(np.float32); v /= np.linalg.norm(v)
            t = st.put("derived fact " + "y" * 80, "derived", sess + ret, commit=False); ix.add(t, v, commit=False)
            tags.append(t); derived.append(t)
        if len(tags) % 3100 < S + D:
            ix.flush()
    ix.flush(); st.db.commit()
    write_ms = (time.perf_counter() - t0) * 1e3 / len(tags)
    db_mb = os.path.getsize(os.path.join(d, "m.db")) / 2**20
    # reads (cold key cache)
    st.dag.drop_cache(); sample = rnd.sample(tags, n_get)
    t0 = time.perf_counter()
    for t in sample: st.get(t)
    get_ms = (time.perf_counter() - t0) * 1e3 / n_get
    # search (first = decrypt all shards, then warm)
    q = rng.standard_normal(dim).astype(np.float32); q /= np.linalg.norm(q)
    ix.state_reset_cache(); t0 = time.perf_counter(); ix.search(q, 10); cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(10): ix.search(q, 10)
    warm_ms = (time.perf_counter() - t0) * 1e2
    # forgets of random sources
    fs = rnd.sample(sources, n_forget); clos, tms = [], []
    for t in fs:
        if t not in st.dag.parents: continue
        t0 = time.perf_counter(); c = ix.forget(t); tms.append((time.perf_counter() - t0) * 1e3); clos.append(len(c) - 1)
    state_kb = st.dag.pprf.size_bytes() / 1024
    alive = len(st.tags())
    t0 = time.perf_counter(); kept = st.rotate(); rot_s = time.perf_counter() - t0
    state_after_kb = st.dag.pprf.size_bytes() / 1024
    st.close()
    return {"regime": regime, "objects": len(tags), "write_ms_per_object": round(write_ms, 4), "db_MiB": round(db_mb, 1),
            "get_ms_cold": round(get_ms, 4), "search_first_s": round(cold, 3), "search_warm_ms": round(warm_ms, 2),
            "forget_ms_median": round(float(np.median(tms)), 2), "forget_ms_p95": round(float(np.percentile(tms, 95)), 2),
            "closure_descendants_median": float(np.median(clos)), "closure_descendants_p95": float(np.percentile(clos, 95)),
            "forgets": len(tms), "objects_alive_after": alive, "pprf_state_KiB_after_forgets": round(state_kb, 1),
            "rotate_s": round(rot_s, 2), "rotate_kept": kept, "pprf_state_KiB_after_rotate": round(state_after_kb, 2)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,100000"); ap.add_argument("--regimes", default="flat,chained"); ap.add_argument("--out", default="results/results_scale.json")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out["machine"] = {"cpu": platform.processor() or "Intel Xeon 2.1 GHz", "cores": os.cpu_count(), "python": platform.python_version()}
    for n in [int(x) for x in a.sizes.split(",")]:
        for rg in a.regimes.split(","):
            r = run(n, rg); out[f"{rg}:{n}"] = r; print(rg, n, r, flush=True)
        json.dump(out, open(a.out, "w"), indent=2)
