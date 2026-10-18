"""索引与问答编排。"""
from __future__ import annotations

from typing import Any

from .llm import answer as llm_answer
from .store import VectorStore


class RagPipeline:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()

    def ingest(self, doc_id: str, title: str, text: str, tags: list[str] | None = None) -> dict:
        doc = self.store.add_document(doc_id, title, text, tags)
        return {"doc_id": doc.doc_id, "title": doc.title, "chunks": len(doc.chunks)}

    def query(self, question: str, top_k: int = 4, doc_id: str | None = None) -> dict[str, Any]:
        contexts = self.store.search(question, top_k=top_k, doc_id=doc_id)
        result = llm_answer(question, contexts)
        result["contexts"] = contexts
        return result

    def stats(self) -> dict[str, int]:
        return self.store.stats()
