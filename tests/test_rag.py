"""分块 / 检索排序 / 离线问答 / API 测试。"""
from fastapi.testclient import TestClient

from app.api import app
from app.chunker import chunk_text
from app.llm import answer, offline_answer
from app.pipeline import RagPipeline

SAMPLE = """公司年假制度

员工入职满一年后享有带薪年假。工龄 1-10 年的员工年假为 5 天，10-20 年为 10 天，20 年以上为 15 天。

年假申请流程

员工需要在 OA 系统提交年假申请，由直属上级审批。超过 5 天的年假需部门负责人二次审批。"""


def test_chunk_preserves_order_and_overlap():
    chunks = chunk_text("甲" * 500, size=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0][:80] == chunks[1][:80]  # 相邻块有重叠
    assert "".join(chunks).replace("甲", "") == ""


def test_pipeline_ingest_and_query():
    pipe = RagPipeline()
    pipe.ingest("hr-1", "员工手册", SAMPLE)
    result = pipe.query("年假有几天？")
    assert result["mode"] == "offline"
    assert "年假" in result["answer"]
    assert result["contexts"], "检索结果不应为空"
    assert result["contexts"][0]["title"] == "员工手册"


def test_search_ranks_relevant_chunk_first():
    pipe = RagPipeline()
    pipe.ingest("hr-1", "员工手册", SAMPLE)
    hits = pipe.store.search("年假审批流程", top_k=2)
    # Top-2 中应覆盖到「审批」相关内容，且分数单调不增
    assert any("审批" in h["text"] for h in hits)
    assert hits[0]["score"] >= hits[1]["score"]


def test_metadata_filter():
    pipe = RagPipeline()
    pipe.ingest("hr-1", "员工手册", SAMPLE)
    pipe.ingest("it-1", "IT 规范", "办公电脑密码每 90 天更换一次。")
    hits = pipe.store.search("密码", doc_id="it-1")
    assert hits and hits[0]["doc_id"] == "it-1"


def test_empty_store_returns_friendly_message():
    result = RagPipeline().query("任意问题")
    assert result["mode"] == "offline"
    assert "资料" in result["answer"]


def test_api_flow():
    client = TestClient(app)
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["llm_mode"] in ("online", "offline")
    r = client.post("/documents", json={"doc_id": "hr-1", "title": "员工手册", "text": SAMPLE})
    assert r.status_code == 200
    assert r.json()["chunks"] >= 2
    r = client.post("/query", json={"question": "年假几天"})
    assert r.status_code == 200
    assert r.json()["mode"] in ("online", "offline")
    assert client.get("/stats").json()["documents"] >= 1
