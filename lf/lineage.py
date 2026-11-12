"""Lineage capture rule for an agent loop.

mode="sound"     : parents(d) = every object whose plaintext is in the LLM context when d is written
                   (sound by construction: the model cannot write what it never saw, parametric knowledge aside)
mode="attributed": parents(d) = objects selected by an attribution function (precision > recall; no guarantee)
"""
from __future__ import annotations

class ContextTracker:
    def __init__(self, store, mode="sound", attribute=None):
        self.store, self.mode, self.attribute = store, mode, attribute
        self.ctx: list[int] = []

    def observe(self, tag: int):
        if tag not in self.ctx:
            self.ctx.append(tag)

    def ingest(self, content: str, kind="source") -> int:
        t = self.store.put(content, kind=kind, parents=())
        self.observe(t)
        return t

    def recall(self, tag: int) -> str:
        txt = self.store.get(tag)
        self.observe(tag)
        return txt

    def write(self, content: str, kind="derived") -> int:
        if self.mode == "attributed" and self.attribute:
            parents = self.attribute(content, list(self.ctx))
        else:
            parents = list(self.ctx)
        t = self.store.put(content, kind=kind, parents=parents)
        self.observe(t)
        return t

    def new_session(self):
        self.ctx = []
