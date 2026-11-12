"""Phase A: run a real local LLM as the memory writer over a long-term conversation benchmark.

For every session, in order, the agent sees (i) the session turns and (ii) the top-3 earlier derived
memories retrieved by embedding similarity, then writes one SUMMARY and up to 8 FACTS (mem0-style
extraction). We log exactly what was in the model context, so that every lineage rule can be replayed
offline in phase B without calling the LLM again. Canary turns (unique secrets) are planted in a fixed,
seeded subset of sessions to measure how far a forgotten secret propagates.

Usage: python3 -I agent/build_log.py --data data/locomo10.json --out runs/locomo_log.jsonl
"""
import argparse, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.common import Embedder, LLM, SEED

TEMPLATES = [
    "Oh, before I forget: my new apartment door code is {num}, and the building manager is called {word}.",
    "Random thing, but my library card number changed to {num} and my book-club nickname is now {word}.",
    "I booked the vet for next week, confirmation number {num}, at the clinic on {word} Street.",
    "My new gym locker is number {num}; the trainer there, {word}, is really nice.",
]
SYL = ["bram", "quil", "dor", "vex", "mar", "lun", "thal", "kest", "orin", "zeph", "wyn", "cal", "brix", "fen", "gor", "sil"]

def canary_word(rnd):
    return "".join(rnd.choice(SYL) for _ in range(3)).capitalize()

def locomo_sessions(conv):
    n = 1
    while f"session_{n}" in conv:
        yield n, conv.get(f"session_{n}_date_time", ""), conv[f"session_{n}"]
        n += 1

PROMPT = """You maintain the long-term memory of a personal assistant.
{mem}Conversation session ({date}):
{turns}

Write the memory update in exactly this format:
SUMMARY: <2-3 sentences>
FACTS:
- <one standalone fact per line, with names and dates when known; at most 8 facts>"""

def parse(txt):
    summ, facts, mode = "", [], None
    for line in txt.splitlines():
        l = line.strip().strip("*").strip()
        if not l: continue
        if l.upper().startswith("SUMMARY"):
            summ = l.split(":", 1)[1].strip() if ":" in l else ""; mode = "s"; continue
        if l.upper().startswith("FACTS"):
            mode = "f"; continue
        if mode == "f" and (l.startswith("-") or re.match(r"^\d+[.)]", l)):
            f = re.sub(r"^(-|\d+[.)])\s*", "", l).strip()
            if f: facts.append(f)
        elif mode == "s" and not facts:
            summ = (summ + " " + l).strip()
    return summ, facts[:8]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--out", default="runs/locomo_log.jsonl")
    ap.add_argument("--canary_rate", type=float, default=0.25); ap.add_argument("--convs", default="")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["conv"], r["session"]))
    data = json.load(open(a.data)); rnd = random.Random(SEED)
    emb, llm = Embedder(), LLM()
    convs = [int(x) for x in a.convs.split(",")] if a.convs else range(len(data))
    used_nums = set()
    for ci in range(len(data)):
        conv = data[ci]["conversation"]; A, B = conv["speaker_a"], conv["speaker_b"]
        derived = []   # (id, text, vec) of earlier derived memories in this conversation
        prior = {}
        if os.path.exists(a.out):
            for line in open(a.out):
                r = json.loads(line)
                if r["conv"] == ci: prior[r["session"]] = r
        for sn, date, turns in locomo_sessions(conv):
            # canary planting is decided for every session (even resumed ones) so the RNG stream is stable
            plant = rnd.random() < a.canary_rate
            if plant:
                while True:
                    num = str(rnd.randrange(10000, 99999))
                    if num not in used_nums: used_nums.add(num); break
                word = canary_word(rnd); tpl = rnd.choice(TEMPLATES); spk = rnd.choice([A, B])
                pos = rnd.randrange(0, len(turns) + 1)
            if ci not in convs: continue
            if (ci, sn) in done:
                r = prior[sn]
                for did, txt in r["derived"]:
                    derived.append((did, txt, emb.one(txt)))
                continue
            tl = [{"id": t["dia_id"], "speaker": t["speaker"], "text": t["text"]} for t in turns]
            canary = None
            if plant:
                cid = f"C{ci}-{sn}"
                tl.insert(pos, {"id": cid, "speaker": spk, "text": tpl.format(num=num, word=word)})
                canary = {"id": cid, "num": num, "word": word}
            stext = "\n".join(f"{t['speaker']}: {t['text']}" for t in tl)
            retrieved = []
            if derived:
                q = emb.one(stext[:2000])
                sims = sorted(((float(v @ q), did, txt) for did, txt, v in derived), reverse=True)[:3]
                retrieved = [(did, txt) for _, did, txt in sims]
            mem = "Earlier memories:\n" + "\n".join(f"- {t}" for _, t in retrieved) + "\n\n" if retrieved else ""
            t0 = time.time()
            out, usage = llm.chat(PROMPT.format(mem=mem, date=date, turns=stext), max_tokens=320)
            summ, facts = parse(out)
            ds = []
            if summ: ds.append((f"{ci}:{sn}:S", f"[{date}] {summ}"))
            ds += [(f"{ci}:{sn}:F{i}", f"[{date}] {f}") for i, f in enumerate(facts)]
            rec = {"conv": ci, "session": sn, "date": date, "turns": tl, "retrieved": [d for d, _ in retrieved],
                   "derived": ds, "canary": canary, "raw": out, "usage": usage, "secs": round(time.time() - t0, 2)}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            for did, txt in ds:
                derived.append((did, txt, emb.one(txt)))
            print(ci, sn, len(ds), rec["secs"], flush=True)

if __name__ == "__main__":
    main()
