"""Phase A, windowed flat writer: the same turns as the session log (canaries included), but the LLM extracts
facts from windows of W consecutive turns, with no retrieval of earlier memories. Under the sound rule each
fact then has at most W parents, all of them source turns, so the forgetting radius is bounded.
Records use session = 100*session + window index, so that every downstream script works unchanged.
Usage: python3 -I agent/build_window.py --src runs/locomo_log.jsonl --convs 0,1 --w 4 --out runs/locomo_win4_log.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.common import LLM

WINDOW_PROMPT = """You maintain the long-term memory of a personal assistant.
Conversation excerpt ({date}):
{turns}

List up to 3 standalone facts worth remembering from this excerpt, with names and dates when known.
Write exactly this format, and write "FACTS:" followed by nothing if there is nothing worth remembering:
FACTS:
- <fact>"""

def parse_facts(txt):
    out = []
    for l in txt.splitlines():
        l = l.strip().strip("*").strip()
        if l.startswith("-"):
            f = l[1:].strip()
            if f: out.append(f)
    return out[:3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/locomo_log.jsonl"); ap.add_argument("--convs", default="0,1")
    ap.add_argument("--w", type=int, default=4); ap.add_argument("--out", default="runs/locomo_win4_log.jsonl")
    a = ap.parse_args(); convs = {int(x) for x in a.convs.split(",")}
    src = sorted((json.loads(l) for l in open(a.src)), key=lambda r: (r["conv"], r["session"]))
    done = {(r["conv"], r["session"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    llm = LLM()
    for r in src:
        if r["conv"] not in convs: continue
        for wi in range(0, len(r["turns"]), a.w):
            sn = r["session"] * 100 + wi // a.w
            if (r["conv"], sn) in done: continue
            tl = r["turns"][wi: wi + a.w]
            t0 = time.time()
            out, usage = llm.chat(WINDOW_PROMPT.format(date=r["date"], turns="\n".join(f"{t['speaker']}: {t['text']}" for t in tl)), max_tokens=160)
            ds = [(f"{r['conv']}:{sn}:F{i}", f"[{r['date']}] {f}") for i, f in enumerate(parse_facts(out))]
            canary = r["canary"] if r["canary"] and any(t["id"] == r["canary"]["id"] for t in tl) else None
            rec = {"conv": r["conv"], "session": sn, "date": r["date"], "turns": tl, "retrieved": [], "derived": ds,
                   "canary": canary, "raw": out, "usage": usage, "secs": round(time.time() - t0, 2), "writer": f"window{a.w}"}
            with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
            print(r["conv"], sn, len(ds), rec["secs"], flush=True)

if __name__ == "__main__":
    main()
