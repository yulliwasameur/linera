"""mem0 2.2.1 + LinEra back-end, infer=False path (no LLM needed). The infer=True path with a local LLM is
exercised by eval/mem0_e2e.py."""
import os, sys, tempfile
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEM0_TELEMETRY", "False")
mem0 = pytest.importorskip("mem0")
from integrations.mem0_linera import LinEraMemory

CFG = {"llm": {"provider": "openai", "config": {"model": "local", "api_key": "x", "openai_base_url": "http://127.0.0.1:9/v1"}},
       "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384}}}

def test_forget_message_kills_memories_written_from_it():
    d = tempfile.mkdtemp()
    m = LinEraMemory(d, CFG, dim=384)
    t1, _ = m.add_message("My locker code at the gym is 48213.", infer=False)
    t2, _ = m.add_message("I like hiking in the Alps.", infer=False)
    before = [r["memory"] for r in m.search("locker code")["results"]]
    assert any("48213" in x for x in before)
    m.forget_message(t1)
    after = [r["memory"] for r in m.search("locker code")["results"]]
    assert not any("48213" in x for x in after)
    assert any("Alps" in r["memory"] for r in m.search("hiking")["results"])
    # mem0 history is not on disk; the only database is LinEra's
    raw = open(os.path.join(d, "linera.db"), "rb").read()
    assert b"48213" not in raw
