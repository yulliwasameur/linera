"""E3 with the real LLM: answer questions from what survives in memory after forgetting.

Memory view = derived memories only (mem0-like "facts" store). Conditions at forget rate p:
  none          no forgetting (upper bound)
  no_lineage    only the forgotten turns disappear (= logical deletion, per-object or epoch keys)
  attributed    LinEra, attributed lineage (closure over the attributed DAG)
  sound         LinEra, sound lineage (closure over the sound DAG)
  sound_rd      sound + re-derivation: every session that lost derived memories is re-summarised by the
                same LLM from its surviving turns (+3 retrieved surviving memories); new memories get sound parents
Only questions whose evidence turns all survive are scored (the others are *meant* to become unanswerable).
Metric: token F1 with the LoCoMo normalisation; also the rate of "unknown" answers.
Usage: python3 -I agent/qa.py --log runs/locomo_log.jsonl --convs 0,1 --n 120 --out results/results_qa_locomo.json
"""
import argparse, hashlib, json, os, random, re, string, sys, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.common import Embedder, LLM, SEED
from agent.build_log import PROMPT, parse
from agent.build_window import WINDOW_PROMPT, parse_facts
from agent.replay import load_log, texts_of, embed_all, lineage_closure_struct, closure, forget_sets, questions

QA_PROMPT = """Memories (oldest dates may matter):
{mem}

Question: {q}
Answer the question in a few words, based on the memories above."""

def norm(s):
    s = str(s).lower().replace(",", "")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()

def f1(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g: return float(p == g)
    common = collections.Counter(p) & collections.Counter(g); ns = sum(common.values())
    if ns == 0: return 0.0
    pr, rc = ns / len(p), ns / len(g); return 2 * pr * rc / (pr + rc)

def save_cache(cache, path):
    tmp = path + ".tmp"; json.dump(cache, open(tmp, "w")); os.replace(tmp, path)

def rederive(recs, lost, E, emb, llm, cache, key, cache_path=None):
    """Re-summarise every session that lost derived memories. Returns {new_id: (text, vec, conv)} and #calls."""
    new, calls = {}, 0
    for r in recs:
        mine = [d for d, _ in r["derived"]]
        if not any(d in lost for d in mine): continue
        turns = [t for t in r["turns"] if f"{r['conv']}:{t['id']}" not in lost]
        if not turns: continue
        stext = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
        if r.get("writer", "").startswith("window"):            # windowed flat writer: no retrieval, facts only
            ck = f"{key}|{r['conv']}:{r['session']}"; calls += 1
            if ck not in cache:
                cache[ck] = llm.chat(WINDOW_PROMPT.format(date=r["date"], turns=stext), max_tokens=160)[0]
                if cache_path: save_cache(cache, cache_path)
            for i, f in enumerate(parse_facts(cache[ck])):
                txt = f"[{r['date']}] {f}"; new[f"{r['conv']}:{r['session']}:R{i}"] = (txt, emb.one(txt), r["conv"])
            continue
        # retrieval over surviving earlier memories of this conversation (originals + re-derived)
        pool = [(d, txt) for rr in recs if rr["conv"] == r["conv"] and rr["session"] < r["session"]
                for d, txt in rr["derived"] if d not in lost]
        pool += [(d, v[0]) for d, v in new.items() if v[2] == r["conv"] and int(d.split(":")[1]) < r["session"]]
        retrieved = []
        if pool:
            q = emb.one(stext[:2000]); vs = emb.many([t for _, t in pool])
            retrieved = [pool[i] for i in np.argsort(-(vs @ q))[:3]]
        mem = "Earlier memories:\n" + "\n".join(f"- {t}" for _, t in retrieved) + "\n\n" if retrieved else ""
        ck = f"{key}|{r['conv']}:{r['session']}"
        calls += 1
        if ck not in cache:
            out, _ = llm.chat(PROMPT.format(mem=mem, date=r["date"], turns=stext), max_tokens=320)
            cache[ck] = out
            if cache_path: save_cache(cache, cache_path)
        summ, facts = parse(cache[ck])
        items = ([f"[{r['date']}] {summ}"] if summ else []) + [f"[{r['date']}] {f}" for f in facts]
        for i, txt in enumerate(items):
            new[f"{r['conv']}:{r['session']}:R{i}"] = (txt, emb.one(txt), r["conv"])
    return new, calls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/locomo_log.jsonl"); ap.add_argument("--data", default="../data/locomo10.json")
    ap.add_argument("--qfile", default="", help="json list of questions (LongMemEval); default: LoCoMo --data")
    ap.add_argument("--convs", default="0,1"); ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--rates", default="0.01,0.1"); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="results/results_qa_locomo.json")
    ap.add_argument("--conds", default="none,no_lineage,attributed,sound,sound_rd")
    ap.add_argument("--rd_rates", default="", help="rates at which sound_rd is run (default: all rates)")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    convs = [int(x) for x in a.convs.split(",")]; rates = [float(x) for x in a.rates.split(",")]
    recs = [r for r in load_log(a.log) if r["conv"] in convs]
    ctag = '-'.join(map(str, convs))                         # cache name; hashed when too long for the file system
    if len(ctag) > 64: ctag = f"{len(convs)}h" + hashlib.sha1(ctag.encode()).hexdigest()[:12]
    T = texts_of(recs); emb = Embedder(); E = embed_all(T, emb, a.log.replace(".jsonl", f"_emb_c{ctag}.npz"))
    Q = json.load(open(a.qfile)) if a.qfile else questions(a.data)
    Q = [q for q in Q if q["conv"] in convs]
    fs = forget_sets(recs, rates, SEED)                      # seed 0 forget sets (nested)
    Fmax = fs[max(rates)]
    elig = [q for q in Q if not any(e in Fmax for e in q["ev"]) and all(e in T for e in q["ev"])]
    rnd = random.Random(SEED); qs = sorted(rnd.sample(elig, min(a.n, len(elig))), key=lambda q: (q["conv"], q["q"]))
    llm = LLM()
    cache_path = a.out.replace(".json", "_llmcache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    derived = {d: (txt, E[d], r["conv"]) for r in recs for d, txt in r["derived"]}
    structs = {m: lineage_closure_struct(recs, m, E)[1] for m in ("sound", "attributed")}
    want = a.conds.split(",")
    rd_rates = [float(x) for x in a.rd_rates.split(",")] if a.rd_rates else rates
    conds = [("none", 0.0)] + [(c, p) for p in rates for c in ("no_lineage", "attributed", "sound", "sound_rd")
                               if c in want and not (c == "sound_rd" and p not in rd_rates)]
    prev = json.load(open(a.out)).get("conditions", {}) if os.path.exists(a.out) else {}
    out = {"meta": {"seed": SEED, "convs": convs, "questions_eligible": len(elig), "questions_scored": len(qs),
                    "k": a.k, "rates": rates}, "conditions": {k: v for k, v in prev.items()}}
    for cond, p in conds:
        F = fs.get(p, set())
        L = {"none": set(), "no_lineage": set(F)}.get(cond)
        if L is None: L = closure(structs["attributed" if cond == "attributed" else "sound"], F)
        mem = {d: v for d, v in derived.items() if d not in L}
        rd_calls = 0
        if cond == "sound_rd":
            t0 = time.time(); new, rd_calls = rederive(recs, L, E, emb, llm, cache, f"rd|{p}", cache_path)
            json.dump(cache, open(cache_path, "w")); mem.update(new); rd_secs = time.time() - t0
        scores, unk = [], 0
        for q in qs:
            pool = [(d, v) for d, v in mem.items() if v[2] == q["conv"]]
            if pool:
                qv = emb.query(q["q"]); M = np.vstack([v[1] for _, v in pool])
                top = [pool[i][1][0] for i in np.argsort(-(M @ qv))[: a.k]]
            else:
                top = []
            prompt = QA_PROMPT.format(mem="\n".join(f"- {t}" for t in top) or "(none)", q=q["q"])
            ck_old = f"qa2|{cond}|{p}|{q['conv']}|{q['q']}"
            ck = "qa3|" + hashlib.sha256(prompt.encode()).hexdigest()      # identical prompt => identical answer
            if ck not in cache:
                cache[ck] = cache[ck_old] if ck_old in cache else llm.chat(prompt, max_tokens=32)[0]
                if len(cache) % 10 == 0: save_cache(cache, cache_path)
            ans = cache[ck].strip()
            unk += any(w in ans.lower() for w in ("unknown", "not mentioned", "not specified", "no information", "does not mention", "not provided"))
            scores.append(f1(ans, q["a"]))
        json.dump(cache, open(cache_path, "w"))
        nd = sum(1 for r in recs for _ in r["derived"])
        res = {"rate": p, "f1": round(float(np.mean(scores)), 4), "f1_sd_bootstrap": round(float(np.std(
                   [np.mean(np.random.default_rng(SEED + b).choice(scores, len(scores))) for b in range(1000)])), 4),
               "unknown_rate": round(unk / len(qs), 4), "derived_alive": len(mem), "derived_original": nd,
               "rederive_llm_calls": rd_calls}
        if cond == "sound_rd": res["rederive_seconds"] = round(rd_secs, 1)
        out["conditions"][f"{cond}@{p}"] = res
        print(cond, p, res, flush=True)
        json.dump(out, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
