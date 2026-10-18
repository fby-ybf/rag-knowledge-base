"""内存向量库：相似度检索、元数据过滤、pickle 持久化。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import TfidfEmbedder


@dataclass
class Document:
    doc_id: str
    title: str
    chunks: list[str]
    metadatas: list[dict[str, Any]] = field(default_factory=list)


class VectorStore:
    """把文档分块向量化后存入内存，支持余弦相似度 Top-K 检索。"""

    def __init__(self) -> None:
        self.embedder = TfidfEmbedder()
        self._texts: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None
        self.documents: dict[str, Document] = {}

    # ---------- 写入 ----------
    def add_document(self, doc_id: str, title: str, text: str, tags: list[str] | None = None) -> Document:
        from .chunker import chunk_text

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("文档内容为空，无法入库")
        metas = [{"doc_id": doc_id, "title": title, "chunk_index": i, "tags": tags or []} for i in range(len(chunks))]
        doc = Document(doc_id=doc_id, title=title, chunks=chunks, metadatas=metas)
        self.documents[doc_id] = doc
        self._rebuild()
        return doc

    def _rebuild(self) -> None:
        texts, metas = [], []
        for doc in self.documents.values():
            texts.extend(doc.chunks)
            metas.extend(doc.metadatas)
        self._texts, self._metas = texts, metas
        self._matrix = self.embedder.fit_transform(texts) if texts else None

    # ---------- 查询 ----------
    def search(self, query: str, top_k: int = 4, doc_id: str | None = None) -> list[dict[str, Any]]:
        if self._matrix is None or not self._texts:
            return []
        qv = self.embedder.transform([query])
        scores = self._matrix @ qv[0]
        order = np.argsort(-scores)
        hits: list[dict[str, Any]] = []
        for idx in order:
            meta = self._metas[int(idx)]
            if doc_id and meta["doc_id"] != doc_id:
                continue
            hits.append(
                {
                    "text": self._texts[int(idx)],
                    "score": round(float(scores[int(idx)]), 4),
                    "doc_id": meta["doc_id"],
                    "title": meta["title"],
                    "chunk_index": meta["chunk_index"],
                }
            )
            if len(hits) >= top_k:
                break
        return hits

    # ---------- 运维 ----------
    def stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "chunks": len(self._texts)}

    def save(self, path: str | Path) -> None:
        payload = {doc_id: asdict_safe(doc) for doc_id, doc in self.documents.items()}
        Path(path).write_bytes(pickle.dumps(payload))

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        payload = pickle.loads(p.read_bytes())
        self.documents = {k: Document(**v) for k, v in payload.items()}
        self._rebuild()
        return True


def asdict_safe(doc: Document) -> dict[str, Any]:
    return {"doc_id": doc.doc_id, "title": doc.title, "chunks": doc.chunks, "metadatas": doc.metadatas}
