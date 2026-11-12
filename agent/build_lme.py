"""Phase A on a LongMemEval-S subset (CPU budget).

Subset (seeded): 6 questions for each of multi-session, temporal-reasoning, knowledge-update,
single-session-user (abstention, preference and assistant-side questions are excluded: we keep user turns
only, and preference answers are rubric-scored). Each question has its own user history made of its
evidence sessions plus one random distractor session from its haystack, in date order. User turns only,
each cut to 500 characters. Evidence turns = user turns marked has_answer.
Usage: python3 -I agent/build_lme.py --data ../data/longmemeval_s_cleaned.json
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.common import Embedder, LLM, SEED
from agent.build_log import PROMPT, parse

TYPES = ["multi-session", "temporal-reasoning", "knowledge-update", "single-session-user"]

def subset(data, per_type=6):
    rnd = random.Random(SEED); out = []
    for ty in TYPES:
        pool = [x for x in data if x["question_type"] == ty and "_abs" not in x["question_id"]]
        out += rnd.sample(pool, per_type)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/longmemeval_s_cleaned.json")
    ap.add_argument("--out", default="runs/lme_log.jsonl"); ap.add_argument("--qout", default="runs/lme_questions.json")
    ap.add_argument("--per_type", type=int, default=6)
    a = ap.parse_args(); os.makedirs("runs", exist_ok=True)
    data = json.load(open(a.data)); sub = subset(data, a.per_type); del data
    rnd = random.Random(SEED + 1); Q = []; plan = []
    for ci, x in enumerate(sub):
        ids, dates, sess = x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]
        ans = [i for i, s in enumerate(ids) if s in x["answer_session_ids"]]
        others = [i for i in range(len(ids)) if i not in ans]
        chosen = sorted(ans + [rnd.choice(others)], key=lambda i: dates[i])
        ev = []
        for sn, i in enumerate(chosen, 1):
            turns = []
            for j, t in enumerate(sess[i]):
                if t["role"] != "user": continue
                tid = f"S{sn}:{j}"; turns.append({"id": tid, "speaker": "User", "text": t["content"][:500]})
                if t.get("has_answer"): ev.append(f"{ci}:{tid}")
            if i in ans and not any(e.startswith(f"{ci}:S{sn}:") for e in ev):
                ev += [f"{ci}:{t['id']}" for t in turns]
            plan.append((ci, sn, dates[i], turns))
        Q.append({"conv": ci, "q": x["question"] + f" (asked on {x['question_date']})", "a": str(x["answer"]),
                  "cat": x["question_type"], "ev": ev, "qid": x["question_id"]})
    json.dump(Q, open(a.qout, "w"), indent=1)
    done = set()
    if os.path.exists(a.out):
        done = {(r["conv"], r["session"]) for r in map(json.loads, open(a.out))}
    emb, llm = Embedder(), LLM()
    derived = {}
    if os.path.exists(a.out):
        for r in map(json.loads, open(a.out)):
            for did, txt in r["derived"]: derived.setdefault(r["conv"], []).append((did, txt, emb.one(txt)))
    for ci, sn, date, tl in plan:
        if (ci, sn) in done: continue
        stext = "\n".join(f"{t['speaker']}: {t['text']}" for t in tl)
        retrieved = []
        if derived.get(ci):
            q = emb.one(stext[:2000])
            retrieved = [(d, t) for _, d, t in sorted(((float(v @ q), d, t) for d, t, v in derived[ci]), reverse=True)[:3]]
        mem = "Earlier memories:\n" + "\n".join(f"- {t}" for _, t in retrieved) + "\n\n" if retrieved else ""
        t0 = time.time()
        out, usage = llm.chat(PROMPT.format(mem=mem, date=date, turns=stext), max_tokens=320)
        summ, facts = parse(out)
        ds = ([(f"{ci}:{sn}:S", f"[{date}] {summ}")] if summ else []) + [(f"{ci}:{sn}:F{i}", f"[{date}] {f}") for i, f in enumerate(facts)]
        rec = {"conv": ci, "session": sn, "date": date, "turns": tl, "retrieved": [d for d, _ in retrieved],
               "derived": ds, "canary": None, "raw": out, "usage": usage, "secs": round(time.time() - t0, 2)}
        with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
        for did, txt in ds: derived.setdefault(ci, []).append((did, txt, emb.one(txt)))
        print(ci, sn, len(ds), rec["secs"], flush=True)

if __name__ == "__main__":
    main()
