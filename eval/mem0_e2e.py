"""End-to-end case study on mem0 2.2.1 with the local LLM (Qwen2.5-1.5B-Instruct) as mem0's extractor.

Same user messages go through
  (a) vanilla mem0: default Qdrant on-disk store + history SQLite on disk; the user's erasure request is
      served the way mem0 offers it: search the memories about the topic and delete() each match;
  (b) mem0 + LinEra (integrations/mem0_linera.py): forget_message() on the canary messages.
Then the attacker copies every file of the memory directory (and, for LinEra, the current secret state)
and looks for the canary secrets in: decryptable records (LinEra) and raw bytes of every file.
Usage: python3 -I eval/mem0_e2e.py --out results/results_mem0.json
"""
import argparse, glob, json, os, sys, tempfile, time
os.environ["MEM0_TELEMETRY"] = "False"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.common import LLM, SEED
from mem0.llms.base import LLMBase

MSGS = [
    "Hi! I just moved to Lyon for a new job at a robotics startup.",
    "My new apartment door code is 74219 and the building manager is called Quilbramdor.",
    "I play the cello on weekends and I'm learning a Bach suite.",
    "My sister Lina is visiting next month; she's vegetarian.",
    "The vet appointment for my cat Pixel has confirmation number 55120.",
    "I prefer short answers in the morning, I'm usually rushed.",
]
CANARIES = {1: ["74219", "Quilbramdor"], 4: ["55120"]}   # message index -> secrets
TOPICS = {1: "apartment door code building manager", 4: "vet appointment confirmation number cat"}

class LocalLLM(LLMBase):
    def __init__(self, llm):
        self.llm = llm; self.config = type("C", (), {"model": "qwen2.5-1.5b-instruct-q4_k_m"})()
    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kw):
        fmt = None
        if response_format and (response_format == "json" or (isinstance(response_format, dict) and "json" in str(response_format.get("type", "")))):
            fmt = {"type": "json_object"}
        return self.llm.chat_messages(messages, max_tokens=700, json_mode=fmt is not None)[0]

EMB = {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384}}

def scan(dirpath, secrets):
    hits = {}
    for f in glob.glob(os.path.join(dirpath, "**", "*"), recursive=True):
        if os.path.isfile(f):
            b = open(f, "rb").read()
            for s in secrets:
                if s.encode() in b: hits.setdefault(s, []).append(os.path.relpath(f, dirpath))
    return hits

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/results_mem0.json")
    ap.add_argument("--messages", default="", help="optional JSON file: list of user messages and canary indices"); a = ap.parse_args()
    global MSGS, CANARIES, TOPICS
    if a.messages:
        sc = json.load(open(a.messages))
        MSGS = sc["messages"]; CANARIES = {int(k): v for k, v in sc["canaries"].items()}
        TOPICS = {int(k): v for k, v in sc["topics"].items()}
    from mem0 import Memory
    from integrations.mem0_linera import LinEraMemory
    llm = LocalLLM(LLM(n_ctx=16384)); secrets = [s for v in CANARIES.values() for s in v]; out = {"messages": len(MSGS)}
    # ---------------- (a) vanilla mem0
    d = tempfile.mkdtemp(prefix="mem0_vanilla_")
    cfg = {"embedder": EMB, "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(d, "qdrant"),
           "collection_name": "mem", "embedding_model_dims": 384, "on_disk": True}},
           "history_db_path": os.path.join(d, "history.db"),
           "llm": {"provider": "openai", "config": {"model": "local", "api_key": "x", "openai_base_url": "http://127.0.0.1:9/v1"}}}
    m = Memory.from_config(cfg); m.llm = llm
    t0 = time.time()
    for msg in MSGS: m.add([{"role": "user", "content": msg}], user_id="alice")
    t_add = time.time() - t0
    before = [x["memory"] for x in m.get_all(filters={"user_id": "alice"})["results"]]
    deleted = []
    for i, topic in TOPICS.items():                     # erasure request served with mem0's own API
        for r in m.search(topic, filters={"user_id": "alice"}, top_k=5)["results"]:
            if r["score"] >= 0.5 and r["id"] not in deleted:
                m.delete(r["id"]); deleted.append(r["id"])
    after = [x["memory"] for x in m.get_all(filters={"user_id": "alice"})["results"]]
    del m
    out["vanilla"] = {"memories_before": len(before), "memories_deleted": len(deleted), "memories_after": len(after),
                      "secrets_in_live_memories_after": sorted({s for s in secrets for x in after if s in x}),
                      "secrets_in_files_after": scan(d, secrets), "add_seconds": round(t_add, 1),
                      "memories_text_before": before, "memories_text_after": after}
    print(json.dumps(out["vanilla"], indent=1), flush=True)
    # ---------------- (b) mem0 + LinEra
    d2 = tempfile.mkdtemp(prefix="mem0_linera_")
    lm = LinEraMemory(d2, {"embedder": EMB, "llm": cfg["llm"]}, dim=384); lm.m.llm = llm
    tags = []; t0 = time.time()
    for msg in MSGS: tags.append(lm.add_message(msg, user_id="alice")[0])
    t_add = time.time() - t0
    before = [p["data"] for p in (lm.vs._payload(t) for t in lm.vs.idmap.values())]
    lost = set()
    for i in CANARIES: lost |= lm.forget_message(tags[i])
    lm.vs.s.close()
    from lf.store import LineageStore
    from lf.keydag import Forgotten
    thief = LineageStore(os.path.join(d2, "linera.db"), os.path.join(d2, "linera.state"))
    readable = []
    for t in thief.tags():
        try: readable.append(thief.get(t))
        except Forgotten: pass
    out["linera"] = {"memories_before": len(before), "objects_in_closures": len(lost),
                     "readable_objects_after_theft": len(readable),
                     "secrets_in_decryptable_records": sorted({s for s in secrets for x in readable if s in x}),
                     "secrets_in_files_after": scan(d2, secrets), "add_seconds": round(t_add, 1),
                     "memories_text_before": before,
                     "memories_text_after": [json.loads(x).get("data") for x in readable if '"data"' in x]}
    print(json.dumps(out["linera"], indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
