"""E1 + E2: lineage completeness and post-theft forensics (synthetic agent, no LLM).

A simulated agent runs S sessions. Each session ingests sources (a fraction carry a unique canary),
recalls a few older memories, and writes derived memories (summaries/facts) whose text copies snippets
from what is in context -- the way an LLM summary propagates facts. Then the user forgets every
canary source. The attacker obtains the database file and the current secret state, and tries to
recover each canary from (a) objects still decryptable with the stolen state, (b) raw file bytes.
Usage: python3 -I eval/canary.py --sessions 200 --seed 20261109 --out eval/results_canary.json
"""
import argparse, json, os, random, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lf.store import LineageStore
from lf.lineage import ContextTracker
from baselines.stores import PlainLogical, PerObjectKey, EpochKey

def run(scheme, sessions, seed, canary_rate=0.1):
    rnd = random.Random(seed)
    d = tempfile.mkdtemp()
    db, st = os.path.join(d, "m.db"), os.path.join(d, "state.json")
    S = {"lineage": LineageStore, "plain_logical": PlainLogical,
         "per_object_key": PerObjectKey, "epoch_key": EpochKey}[scheme](db, st)
    tr = ContextTracker(S)
    canaries, all_tags, text = {}, [], {}
    t0 = time.perf_counter(); n_writes = 0
    for s in range(sessions):
        if scheme == "epoch_key" and s % 20 == 0:
            S.set_epoch(s // 20)
        tr.new_session()
        for _ in range(rnd.randint(1, 3)):                       # new inputs
            body = f"user note {rnd.random():.6f}"
            if rnd.random() < canary_rate:
                c = f"CANARY-{s}-{rnd.randrange(10**8):08d}"; body += " " + c
            t = tr.ingest(body); all_tags.append(t); text[t] = body; n_writes += 1
            if "CANARY" in body: canaries[c] = t
        for t in rnd.sample(all_tags, k=min(len(all_tags), rnd.randint(0, 3))):   # retrieval
            if S.readable(t):
                tr.recall(t)
        ctx_text = " ".join(text[t] for t in tr.ctx if t in text)
        for _ in range(rnd.randint(1, 2)):                       # derived memories copy context snippets
            words = ctx_text.split()
            k = min(len(words), rnd.randint(4, 12))
            body = "summary: " + " ".join(rnd.sample(words, k)) + " " + " ".join(w for w in words if w.startswith("CANARY") and rnd.random() < 0.7)
            t = tr.write(body); all_tags.append(t); text[t] = body; n_writes += 1
    write_ms = (time.perf_counter() - t0) * 1e3 / max(1, n_writes)
    # forget all canary sources
    t1 = time.perf_counter()
    lost = set()
    for c, t in canaries.items():
        lost |= set(S.forget(t))
    forget_ms = (time.perf_counter() - t1) * 1e3 / max(1, len(canaries))
    S.close()
    # attacker: reload with stolen state (for encrypted schemes the in-memory key state is what is stolen)
    if scheme == "lineage":
        A = LineageStore(db, st); readable = lambda t: A.readable(t); get = A.get
    else:
        A = S; A.db = __import__("sqlite3").connect(db); readable = A.readable; get = A.get
    recovered_decrypt = set()
    for t in all_tags:
        if readable(t):
            body = get(t)
            for c in canaries:
                if c in body: recovered_decrypt.add(c)
    raw = open(db, "rb").read()
    recovered_raw = {c for c in canaries if c.encode() in raw}
    survivors = [t for t in all_tags if readable(t)]
    res = {
        "objects": len(all_tags), "canaries": len(canaries),
        "canary_recovered_by_decryption": len(recovered_decrypt),
        "canary_recovered_from_raw_bytes": len(recovered_raw),
        "canary_survival_rate": round(len(recovered_decrypt | recovered_raw) / max(1, len(canaries)), 4),
        "objects_still_readable": len(survivors),
        "blast_radius_fraction": round(1 - len(survivors) / len(all_tags), 4),
        "write_ms_per_object": round(write_ms, 3), "forget_ms_per_request": round(forget_ms, 3),
    }
    if scheme == "lineage":
        res["state_bytes_after_forgets"] = A.dag.pprf.size_bytes()
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=200); ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", default="eval/results_canary.json")
    a = ap.parse_args()
    out = {"params": vars(a), "results": {s: run(s, a.sessions, a.seed) for s in
           ["plain_logical", "epoch_key", "per_object_key", "lineage"]}}
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=2))
