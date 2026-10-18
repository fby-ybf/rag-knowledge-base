"""FastAPI 路由。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .llm import is_online
from .pipeline import RagPipeline

app = FastAPI(title="rag-knowledge-base", version="1.0.0")
pipeline = RagPipeline()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_PATH = DATA_DIR / "index.pkl"
if INDEX_PATH.exists():
    pipeline.store.load(INDEX_PATH)


class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="文档唯一 ID")
    title: str = Field(..., description="文档标题")
    text: str = Field(..., description="纯文本内容")
    tags: list[str] = []


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    doc_id: str | None = None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "llm_mode": "online" if is_online() else "offline"}


@app.get("/stats")
def stats() -> dict:
    return pipeline.stats()


@app.post("/documents")
def ingest(req: IngestRequest) -> dict:
    try:
        result = pipeline.ingest(req.doc_id, req.title, req.text, req.tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    DATA_DIR.mkdir(exist_ok=True)
    pipeline.store.save(INDEX_PATH)
    return result


@app.post("/query")
def query(req: QueryRequest) -> dict:
    return pipeline.query(req.question, top_k=req.top_k, doc_id=req.doc_id)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
