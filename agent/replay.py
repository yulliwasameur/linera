"""Phase B: replay the LLM log into each storage scheme, forget, attack, and measure (E1, E2, E3-retrieval, E5).

No LLM call here: every derived memory is the text the LLM actually wrote in phase A, and the lineage
rules only decide which parents are recorded.
  sound      parents(d) = all turns of d's session + the earlier memories retrieved into the context
  attributed parents(d) = the 3 context items most similar to d (embedding cosine); no guarantee
Usage: python3 -I agent/replay.py --log runs/locomo_log.jsonl --out results/results_locomo.json
"""
import argparse, json, os, random, re, sys, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.common import Embedder, SEED
from agent import memory as M
from agent.build_log import TEMPLATES, canary_word

def load_log(path):
    recs = [json.loads(l) for l in open(path)]
    recs.sort(key=lambda r: (r["conv"], r["session"]))
    return recs

def texts_of(recs):
    T = {}
    for r in recs:
        for t in r["turns"]:
            T[f"{r['conv']}:{t['id']}"] = f"[{r['date']}] {t['speaker']}: {t['text']}"
        for did, txt in r["derived"]:
            T[did] = txt
    return T

def embed_all(T, emb, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); E = dict(zip(z["ids"], z["vecs"]))
        if all(k in E for k in T): return E
    ids = list(T); V = emb.many([T[i] for i in ids])
    np.savez(cache, ids=np.array(ids, dtype=object), vecs=V)
    return dict(zip(ids, V))

def parents_of(r, did, mode, E, k_attr=3):
    ctx = [f"{r['conv']}:{t['id']}" for t in r["turns"]] + list(r["retrieved"])
    if mode == "sound": return ctx
    sims = sorted(((float(E[c] @ E[did]), c) for c in ctx), reverse=True)
    return [c for _, c in sims[:k_attr]]

def build(recs, scheme, mode, E, dim, epoch_len=5):
    mem = M.make(scheme, dim); tag = {}
    t0 = time.perf_counter(); n = 0
    for i, r in enumerate(recs):
        if scheme == "epoch": mem.set_epoch((r["conv"], (r["session"] - 1) // epoch_len))
        for t in r["turns"]:
            oid = f"{r['conv']}:{t['id']}"
            tag[oid] = mem.put(f"[{r['date']}] {t['speaker']}: {t['text']}", "source", (), E[oid]); n += 1
        for did, txt in r["derived"]:
            ps = [tag[p] for p in parents_of(r, did, mode, E)]
            tag[did] = mem.put(txt, "derived", ps, E[did]); n += 1
    mem.commit()
    return mem, tag, (time.perf_counter() - t0) * 1e3 / n

def lineage_closure_struct(recs, mode, E):
    """Structural DAG (no crypto): returns parents map over object ids."""
    P = {}
    for r in recs:
        for t in r["turns"]: P[f"{r['conv']}:{t['id']}"] = []
        for did, _ in r["derived"]: P[did] = parents_of(r, did, mode, E)
    ch = collections.defaultdict(list)
    for o, ps in P.items():
        for p in ps: ch[p].append(o)
    return P, ch

def closure(ch, roots):
    out, st = set(roots), list(roots)
    while st:
        for c in ch.get(st.pop(), ()):
            if c not in out: out.add(c); st.append(c)
    return out

# ------------------------------------------------------------------ E1 + E2
def canary_attack(recs, T, E, emb, schemes, n_decoys=99):
    can = [(r["conv"], r["canary"]) for r in recs if r["canary"]]
    can_ids = [f"{c}:{x['id']}" for c, x in can]
    propagated = [x for c, x in can if any(re.search(r"(?<![0-9])" + x["num"] + r"(?![0-9])", txt) or x["word"] in txt
                  for r in recs if r["conv"] == c for _, txt in r["derived"])]
    # decoy candidate sets for the vector identification attack (attacker knows the template)
    rnd = random.Random(SEED + 7); cand = {}
    for (c, x), cid in zip(can, can_ids):
        true = T[cid].split(": ", 1)[1]
        tpl = next(tp for tp in TEMPLATES if tp.format(num=x["num"], word=x["word"]) in true)
        spk = T[cid].split("] ", 1)[1].split(":")[0]
        cs = [true] + [tpl.format(num=str(rnd.randrange(10000, 99999)), word=canary_word(rnd)) for _ in range(n_decoys)]
        cand[cid] = emb.many(cs)
    KW = [["door code", "building manager"], ["library card", "book-club", "book club"],
          ["vet", "confirmation number", "clinic"], ["locker", "trainer"]]
    _, ch_true = lineage_closure_struct(recs, "sound", E)        # ground-truth provenance = what the LLM saw
    topic = {}
    for (c, x), cid in zip(can, can_ids):
        ti = next(i for i, tp in enumerate(TEMPLATES) if tp.format(num=x["num"], word=x["word"]) in T[cid])
        desc = closure(ch_true, [cid]) - {cid}
        topic[cid] = (desc, KW[ti])
    topic_prop = sum(1 for cid, (desc, kw) in topic.items() if any(k in T[o].lower() for o in desc for k in kw))
    out = {"canaries": len(can), "canaries_propagated_to_derived": len(propagated),
           "canaries_topic_propagated_to_derived": topic_prop, "schemes": {}}
    for scheme, mode in schemes:
        mem, tag, wms = build(recs, scheme, mode, E, emb.dim)
        t0 = time.perf_counter(); lost = set()
        for cid in can_ids: lost |= mem.forget(tag[cid])
        fms = (time.perf_counter() - t0) * 1e3 / len(can_ids)
        mem.commit(); mem.close()
        texts, vecs = mem.thief()
        raw = b"".join(open(f, "rb").read() for f in mem.files())
        rec_txt, rec_raw, rec_vec, rec_topic = set(), set(), set(), set()
        # whole-token matching: a 5-digit secret must not match inside the 10-digit decimal tags stored in clear
        def has(x, s):
            pat = rb"(?<![0-9])" + x["num"].encode() + rb"(?![0-9])|" + re.escape(x["word"].encode())
            return re.search(pat, s if isinstance(s, bytes) else s.encode()) is not None
        oid_of = {v: k for k, v in tag.items()}
        for cid, (desc, kw) in topic.items():
            if any(oid_of.get(t) in desc and any(k in s.lower() for k in kw) for t, s in texts.items()): rec_topic.add(cid)
        for (c, x), cid in zip(can, can_ids):
            if any(has(x, s) for s in texts.values()): rec_txt.add(cid)
            if has(x, raw): rec_raw.add(cid)
            if len(vecs):
                score = (cand[cid] @ vecs.T).max(axis=1)           # best match of each candidate
                if int(np.argmax(score)) == 0: rec_vec.add(cid)
        prop_ids = {f"{c}:{x['id']}" for c, x in can if x in propagated}
        key = scheme if scheme != "linera" else f"linera_{mode}"
        out["schemes"][key] = {
            "recovered_by_decryption": len(rec_txt), "recovered_from_raw_bytes": len(rec_raw),
            "identified_from_vectors": len(rec_vec), "topic_leak_in_descendants": len(rec_topic),
            "any_channel": len(rec_txt | rec_raw | rec_vec | rec_topic),
            "recovered_by_decryption_among_propagated": len(rec_txt & prop_ids),
            "objects": len(tag), "objects_lost": len(tag) - len(texts) if scheme != "plain_logical" else len(lost),
            "write_ms_per_object": round(wms, 3), "forget_ms_per_request": round(fms, 3)}
        print(key, out["schemes"][key], flush=True)
    return out

# ------------------------------------------------------------------ E3 (retrieval) + E5
def questions(path):
    data = json.load(open(path)); Q = []
    for ci, c in enumerate(data):
        for q in c["qa"]:
            if q["category"] == 5 or not q.get("evidence"): continue
            ev = [f"{ci}:{e.strip()}" for e in q["evidence"]]
            Q.append({"conv": ci, "q": q["question"], "a": str(q["answer"]), "cat": q["category"], "ev": ev})
    return Q

def forget_sets(recs, rates, seed):
    src = [f"{r['conv']}:{t['id']}" for r in recs for t in r["turns"] if not t["id"].startswith("C")]
    rnd = random.Random(seed); rnd.shuffle(src)
    return {p: set(src[: int(round(p * len(src)))]) for p in rates}

def retrieval_utility(recs, T, E, emb, Q, rates, seeds, k=10):
    sess_of = {}
    for r in recs:
        for did, _ in r["derived"]: sess_of[did] = {f"{r['conv']}:{t['id']}" for t in r["turns"]}
    all_ids = list(T); conv_of = {i: int(i.split(":")[0]) for i in all_ids}
    kind = {i: ("derived" if i in sess_of else "source") for i in all_ids}
    qv = emb.many(["Represent this sentence for searching relevant passages: " + q["q"] for q in Q])
    V = np.vstack([E[i] for i in all_ids]); conv_arr = np.array([conv_of[i] for i in all_ids])
    kind_arr = np.array([kind[i] == "derived" for i in all_ids])
    res = {}
    structs = {m: lineage_closure_struct(recs, m, E) for m in ("sound", "attributed")}
    for seed in seeds:
        fs = forget_sets(recs, rates, SEED + seed)
        for p in [0.0] + list(rates):
            F = fs.get(p, set())
            lost = {"none": set(), "no_lineage": set(F),
                    "sound": closure(structs["sound"][1], F), "attributed": closure(structs["attributed"][1], F)}
            for cond, L in lost.items():
                if p == 0.0 and cond != "none": continue
                if p > 0 and cond == "none": continue
                for view in ("facts_only", "hybrid"):
                    alive = np.array([i not in L for i in all_ids]) & (kind_arr if view == "facts_only" else True)
                    hits = n = 0
                    for q, v in zip(Q, qv):
                        if any(e in F for e in q["ev"]): continue          # evidence itself was forgotten
                        m = alive & (conv_arr == q["conv"])
                        sc = np.where(m, V @ v, -9); top = np.argsort(-sc)[:k]
                        ok = False
                        for j in top:
                            i = all_ids[j]
                            if sc[j] <= -9: break
                            if i in q["ev"] or (i in sess_of and sess_of[i] & set(q["ev"])): ok = True; break
                        hits += ok; n += 1
                    key = f"p={p}|{cond}|{view}"
                    res.setdefault(key, []).append(hits / max(1, n))
                    res.setdefault(f"p={p}|{cond}|{view}|n_questions", []).append(n)
                nd = sum(1 for i in all_ids if kind[i] == "derived")
                res.setdefault(f"p={p}|{cond}|derived_lost", []).append(sum(1 for i in L if kind[i] == "derived") / nd)
                res.setdefault(f"p={p}|{cond}|objects_lost", []).append(len(L) / len(all_ids))
    return {k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4), "runs": len(v)} for k, v in res.items()}

def lineage_stats(recs, E):
    out = {}
    for m in ("sound", "attributed"):
        P, ch = lineage_closure_struct(recs, m, E)
        der = [o for o in P if P[o]]
        src = [o for o in P if not P[o] and not o.split(":")[1].startswith("C")]
        sizes = [len(closure(ch, [s])) - 1 for s in src]       # descendants of one forgotten turn
        anc = []
        par = P
        for d in der:                                            # number of source ancestors of each derived object
            seen, st = set(), [d]
            while st:
                for p in par[st.pop()]:
                    if p not in seen: seen.add(p); st.append(p)
            anc.append(sum(1 for x in seen if not par[x]))
        out[m] = {"derived": len(der), "sources": len(src), "mean_parents": round(float(np.mean([len(P[d]) for d in der])), 2),
                  "mean_source_ancestors": round(float(np.mean(anc)), 1), "median_source_ancestors": float(np.median(anc)),
                  "single_forget_descendants_mean": round(float(np.mean(sizes)), 2),
                  "single_forget_descendants_p95": float(np.percentile(sizes, 95)),
                  "single_forget_descendants_max": int(max(sizes))}
    # what-if (structural only): no write-time retrieval -> parents are the session's own turns
    P = {}
    for r in recs:
        for t in r["turns"]: P[f"{r['conv']}:{t['id']}"] = []
        for did, _ in r["derived"]: P[did] = [f"{r['conv']}:{t['id']}" for t in r["turns"]]
    ch = collections.defaultdict(list)
    for o, ps in P.items():
        for p in ps: ch[p].append(o)
    src = [o for o in P if not P[o] and not o.split(":")[1].startswith("C")]
    sizes = [len(closure(ch, [s])) - 1 for s in src]
    out["whatif_session_local"] = {"single_forget_descendants_mean": round(float(np.mean(sizes)), 2),
                                   "single_forget_descendants_p95": float(np.percentile(sizes, 95)),
                                   "single_forget_descendants_max": int(max(sizes))}
    # what-if (structural only): flat derivation -- the writer is shown the *turns* of the sessions it
    # retrieved instead of their summaries, so parents are sources only and chains cannot form
    sess_turns = {(r["conv"], r["session"]): [f"{r['conv']}:{t['id']}" for t in r["turns"]] for r in recs}
    P = {}
    for r in recs:
        for t in r["turns"]: P[f"{r['conv']}:{t['id']}"] = []
        extra = [x for rid in r["retrieved"] for x in sess_turns[(int(rid.split(":")[0]), int(rid.split(":")[1]))]]
        for did, _ in r["derived"]: P[did] = [f"{r['conv']}:{t['id']}" for t in r["turns"]] + extra
    ch = collections.defaultdict(list)
    for o, ps in P.items():
        for p in ps: ch[p].append(o)
    sizes = [len(closure(ch, [s])) - 1 for s in src]
    chs = {"sound": lineage_closure_struct(recs, "sound", E)[1], "attributed": lineage_closure_struct(recs, "attributed", E)[1],
           "flat": ch}
    der_ids = {d for r in recs for d, _ in r["derived"]}
    rad = {}
    for seed in (0, 1, 2):
        fs = forget_sets(recs, [0.01, 0.05, 0.10], SEED + seed)
        for p_, F in fs.items():
            for name, c in chs.items():
                L = closure(c, F)
                rad.setdefault(f"{name}@{p_}", []).append(len(L & der_ids) / len(der_ids))
    out["derived_lost_at_rates"] = {k: round(float(np.mean(v)), 4) for k, v in rad.items()}
    out["whatif_flat"] = {"single_forget_descendants_mean": round(float(np.mean(sizes)), 2),
                          "single_forget_descendants_p95": float(np.percentile(sizes, 95)),
                          "single_forget_descendants_max": int(max(sizes))}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/locomo_log.jsonl"); ap.add_argument("--data", default="../data/locomo10.json")
    ap.add_argument("--out", default="results/results_locomo.json"); ap.add_argument("--parts", default="lineage,canary,utility")
    ap.add_argument("--qfile", default="", help="questions json (LongMemEval subset); default: LoCoMo --data")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    recs = load_log(a.log); T = texts_of(recs); emb = Embedder()
    E = embed_all(T, emb, a.log.replace(".jsonl", "_emb.npz"))
    convs = sorted({r["conv"] for r in recs})
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out["meta"] = {"seed": SEED, "log": os.path.basename(a.log), "conversations": convs, "sessions": len(recs),
                   "sources": sum(len(r["turns"]) for r in recs), "derived": sum(len(r["derived"]) for r in recs),
                   "embedder": Embedder.name, "llm_tokens_prompt": sum(r["usage"]["prompt_tokens"] for r in recs),
                   "llm_tokens_completion": sum(r["usage"]["completion_tokens"] for r in recs),
                   "llm_seconds": round(sum(r["secs"] for r in recs), 1)}
    parts = a.parts.split(",")
    if "lineage" in parts:
        out["E5_lineage"] = lineage_stats(recs, E); print(json.dumps(out["E5_lineage"], indent=1))
    if "canary" in parts:
        out["E1_E2_canary"] = canary_attack(recs, T, E, emb, [("plain_logical", "sound"), ("per_object", "sound"),
                                            ("epoch", "sound"), ("linera", "attributed"), ("linera", "sound")])
    if "utility" in parts:
        Q = [q for q in (json.load(open(a.qfile)) if a.qfile else questions(a.data)) if q["conv"] in convs]
        out["E3_retrieval"] = {"questions": len(Q), "k": 10,
                               "results": retrieval_utility(recs, T, E, emb, Q, [0.01, 0.05, 0.10], [0, 1, 2])}
        print(json.dumps(out["E3_retrieval"], indent=1))
    json.dump(out, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
