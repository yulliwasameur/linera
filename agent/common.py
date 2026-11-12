"""Shared components: seed, embedder (bge-small-en-v1.5, 384-d, normalized) and the LLM.

LLM back-ends, chosen with the environment variable LF_BACKEND:
  llamacpp (default)  local GGUF file in LF_MODEL (Qwen2.5-1.5B-Instruct Q4_K_M by default), CPU
  vllm                Hugging Face model id or path in LF_MODEL, GPU (Hedy, Juliet, Jean Zay, LUMI-G)
  anthropic           Claude API, model id in LF_MODEL (e.g. claude-haiku-5-5); key read by the SDK from
                      ANTHROPIC_API_KEY in the user's own environment
All back-ends use temperature 0 and report token usage in the same format.
"""
import os
import numpy as np

SEED = 20261109
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.environ.get("LF_BACKEND", "llamacpp")
MODEL = os.environ.get("LF_MODEL", os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf"))

class Embedder:
    name = "BAAI/bge-small-en-v1.5"
    dim = 384
    def __init__(self):
        from fastembed import TextEmbedding
        self.m = TextEmbedding(self.name)
    def many(self, texts):
        if not texts: return np.zeros((0, self.dim), np.float32)
        v = np.array(list(self.m.embed(list(texts))), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    def one(self, text):
        return self.many([text])[0]
    def query(self, text):
        # bge query instruction, as recommended for retrieval
        return self.one("Represent this sentence for searching relevant passages: " + text)

class LLM:
    def __init__(self, n_ctx=4096):
        self.backend = BACKEND
        if BACKEND == "llamacpp":
            from llama_cpp import Llama
            self.m = Llama(MODEL, n_ctx=n_ctx, n_threads=os.cpu_count(), seed=SEED, verbose=False)
        elif BACKEND == "vllm":
            from vllm import LLM as VLLM, SamplingParams
            self.m = VLLM(model=MODEL, seed=SEED, max_model_len=max(n_ctx, 16384),
                          gpu_memory_utilization=float(os.environ.get("LF_GPU_MEM", "0.85")))
            self._sp = SamplingParams
        elif BACKEND == "anthropic":
            import anthropic
            self.m = anthropic.Anthropic()
        else:
            raise ValueError(f"unknown LF_BACKEND {BACKEND}")

    def chat_messages(self, messages, max_tokens=256, json_mode=False):
        if self.backend == "llamacpp":
            r = self.m.create_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens, seed=SEED,
                                              response_format={"type": "json_object"} if json_mode else None)
            return r["choices"][0]["message"]["content"], r["usage"]
        if self.backend == "vllm":
            out = self.m.chat(messages, self._sp(temperature=0.0, max_tokens=max_tokens, seed=SEED), use_tqdm=False)[0]
            u = {"prompt_tokens": len(out.prompt_token_ids), "completion_tokens": len(out.outputs[0].token_ids)}
            u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
            return out.outputs[0].text, u
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        r = self.m.messages.create(model=MODEL, max_tokens=max_tokens, temperature=0.0, system=system, messages=msgs)
        u = {"prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens}
        u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text"), u

    def chat(self, prompt, max_tokens=256, system="You are a careful assistant."):
        return self.chat_messages([{"role": "system", "content": system}, {"role": "user", "content": prompt}], max_tokens)
