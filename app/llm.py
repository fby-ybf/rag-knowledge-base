"""LLM 客户端：OpenAI 兼容接口；未配置 Key 时进入离线模式。"""
from __future__ import annotations

import os

import httpx

SYSTEM_PROMPT = (
    "你是企业知识库问答助手。只依据下面给出的资料回答，资料中没有的信息要明确说不知道，"
    "并在结尾列出引用编号。"
)


def _config() -> tuple[str | None, str | None, str]:
    return (
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


def is_online() -> bool:
    base, key, _ = _config()
    return bool(base and key)


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        blocks.append(f"[{i}] 来源：{ctx['title']}（片段 {ctx['chunk_index']}）\n{ctx['text']}")
    return f"资料：\n{chr(10).join(blocks) or '（无）'}\n\n问题：{question}"


def answer(question: str, contexts: list[dict]) -> dict:
    """返回 {'mode': 'online'|'offline', 'answer': str, 'citations': [...]}"""
    citations = [
        {"index": i, "title": c["title"], "chunk_index": c["chunk_index"], "score": c["score"]}
        for i, c in enumerate(contexts, start=1)
    ]
    base, key, model = _config()
    if not (base and key):
        return {"mode": "offline", "answer": offline_answer(question, contexts), "citations": citations}

    try:
        resp = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, contexts)},
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return {"mode": "online", "answer": content, "citations": citations}
    except Exception as exc:  # 网络/鉴权失败时降级，保证链路可用
        return {
            "mode": "offline",
            "answer": offline_answer(question, contexts) + f"\n\n（在线模型调用失败：{exc}）",
            "citations": citations,
        }


def offline_answer(question: str, contexts: list[dict]) -> str:
    """无 LLM 时的模板回答：摘录最相关片段并标注来源。"""
    if not contexts:
        return "知识库中还没有相关资料，请先上传文档。"
    top = contexts[0]
    body = " ".join(c["text"][:120] for c in contexts[:2])
    cites = "、".join(f"[{i}]《{c['title']}》片段{c['chunk_index']}" for i, c in enumerate(contexts, start=1))
    return (
        f"根据知识库中与「{question}」最相关的资料（最高相关度 {top['score']}）：\n{body}……\n\n"
        f"以上内容摘自：{cites}。\n（当前为离线模式，答案由检索片段拼接而成，未调用大模型）"
    )
