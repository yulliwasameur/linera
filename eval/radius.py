"""E5 (complement): forgetting radius versus writer fan-in, on the same conversations (no LLM call).

For each log and lineage rule: mean number of source ancestors of a derived memory (fan-in, transitively),
share of derived memories erased at forget rates 1/5/10% (3 seeds), and the (1-p)^f prediction.
Usage: python3 -I eval/radius.py --out results/results_radius.json
"""
import argparse, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.common import SEED
from agent.replay import load_log, texts_of, lineage_closure_struct, closure, forget_sets

def emb_cache(log):
    z = np.load(log.replace(".jsonl", "_emb.npz"), allow_pickle=True); return dict(zip(z["ids"], z["vecs"]))

def analyse(recs, E, rules):
    out = {}
    der = [d for r in recs for d, _ in r["derived"]]
    for rule in rules:
        if rule == "session_local":
            P = {}
            for r in recs:
                for t in r["turns"]: P[f"{r['conv']}:{t['id']}"] = []
                for d, _ in r["derived"]: P[d] = [f"{r['conv']}:{t['id']}" for t in r["turns"]]
            ch = collections.defaultdict(list)
            for o, ps in P.items():
                for p in ps: ch[p].append(o)
        else:
            P, ch = lineage_closure_struct(recs, rule, E)
        fan = []
        for d in der:
            seen, st = set(), [d]
            while st:
                for p in P[st.pop()]:
                    if p not in seen: seen.add(p); st.append(p)
            fan.append(sum(1 for x in seen if not P[x]))
        res = {"mean_source_ancestors": round(float(np.mean(fan)), 2),
               "fan_hist": {str(k): int(v) for k, v in zip(*np.unique(fan, return_counts=True))}}
        for p in (0.01, 0.05, 0.10):
            vals, pred = [], []
            for seed in (0, 1, 2):
                F = forget_sets(recs, [p], SEED + seed)[p]
                L = closure(ch, F); vals.append(sum(1 for d in der if d in L) / len(der))
            res[f"erased@{p}"] = round(float(np.mean(vals)), 4)
            res[f"predicted@{p}"] = round(float(np.mean([1 - (1 - p) ** f for f in fan])), 4)
        out[rule] = res
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/results_radius.json"); a = ap.parse_args()
    out = {}
    sess = load_log("runs/locomo_log.jsonl"); Es = emb_cache("runs/locomo_log.jsonl")
    out["session_writer_conv0-4"] = analyse(sess, Es, ["sound", "attributed", "session_local"])
    s01 = [r for r in sess if r["conv"] in (0, 1)]
    out["session_writer_conv0-1"] = analyse(s01, Es, ["sound", "attributed", "session_local"])
    for w in (2, 4, 8, 16):
        lg = f"runs/locomo_win{w}_log.jsonl"
        if os.path.exists(lg) and os.path.exists(lg.replace(".jsonl", "_emb.npz")):
            win = load_log(lg); Ew = emb_cache(lg); key = f"window{w}_writer_conv0-1"
            out[key] = analyse(win, Ew, ["sound", "attributed"])
            out[key]["derived"] = sum(len(r["derived"]) for r in win); out[key]["llm_calls"] = len(win)
    lg = "runs/locomo_llama3b_log.jsonl"
    if os.path.exists(lg) and os.path.exists(lg.replace(".jsonl", "_emb.npz")):
        L3 = load_log(lg); E3 = emb_cache(lg)
        out["llama3b_session_writer_conv0-1"] = analyse(L3, E3, ["sound", "attributed", "session_local"])
        out["llama3b_session_writer_conv0-1"]["derived"] = sum(len(r["derived"]) for r in L3)
    out["session_writer_conv0-1"]["derived"] = sum(len(r["derived"]) for r in s01)
    # extended campaign: Qwen2.5-7B (vLLM, H100), all 10 LoCoMo conversations, session writer and 4-turn windows
    for key, lg, rules in [("qwen7b_session_writer_conv0-9", "runs/locomo_qwen7b_log.jsonl", ["sound", "attributed", "session_local"]),
                           ("qwen7b_window4_writer_conv0-9", "runs/locomo_qwen7b_win4_log.jsonl", ["sound", "attributed"])]:
        if os.path.exists(lg) and os.path.exists(lg.replace(".jsonl", "_emb.npz")):
            R7 = load_log(lg); E7 = emb_cache(lg)
            out[key] = analyse(R7, E7, rules)
            out[key]["derived"] = sum(len(r["derived"]) for r in R7); out[key]["llm_calls"] = len(R7)
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
